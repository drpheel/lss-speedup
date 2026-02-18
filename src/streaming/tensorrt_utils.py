"""TensorRT conversion helpers and compatibility patching."""

import importlib
import os
import traceback

import torch

from .config import TRT_DEBUG_TRACEBACK, TENSORRT_WORKSPACE_MB, USE_FP16

try:
    from torch2trt import TRTModule, torch2trt

    HAS_TORCH2TRT = True
except Exception:
    HAS_TORCH2TRT = False

_TORCH2TRT_PATCHED = False


def _apply_torch2trt_conv_dims_patch() -> None:
    global _TORCH2TRT_PATCHED
    if _TORCH2TRT_PATCHED or not HAS_TORCH2TRT:
        return

    try:
        import tensorrt as trt

        t2t = importlib.import_module("torch2trt.torch2trt")
        nc = importlib.import_module("torch2trt.converters.native_converters")

        def _flatten_ints(v):
            if isinstance(v, (list, tuple)):
                out = []
                for item in v:
                    out.extend(_flatten_ints(item))
                return out
            try:
                return [int(v)]
            except Exception:
                return [int(float(v))]

        def _normalize_nd_param(v, ndim, default):
            vals = _flatten_ints(v)
            if len(vals) == 0:
                vals = [int(default)]
            if len(vals) == 1:
                return tuple([vals[0]] * ndim)
            if len(vals) >= ndim:
                return tuple(vals[:ndim])
            return tuple(vals + [vals[-1]] * (ndim - len(vals)))

        def _to_trt_dims(vals):
            vals = _flatten_ints(vals)
            if len(vals) == 1:
                return trt.Dims((vals[0],))
            if len(vals) == 2 and hasattr(trt, "DimsHW"):
                return trt.DimsHW(vals[0], vals[1])
            return trt.Dims(tuple(vals))

        def convert_conv2d3d_patched(ctx):
            input = nc.get_arg(ctx, "input", pos=0, default=None)
            weight = nc.get_arg(ctx, "weight", pos=1, default=None)
            bias = nc.get_arg(ctx, "bias", pos=2, default=None)
            stride = nc.get_arg(ctx, "stride", pos=3, default=1)
            padding = nc.get_arg(ctx, "padding", pos=4, default=0)
            dilation = nc.get_arg(ctx, "dilation", pos=5, default=1)
            groups = nc.get_arg(ctx, "groups", pos=6, default=1)

            input_trt = nc.add_missing_trt_tensors(ctx.network, [input])[0]
            output = ctx.method_return
            input_dim = input.dim() - 2

            out_channels = int(weight.shape[0])
            kernel_size = _normalize_nd_param(tuple(weight.shape[2:]), input_dim, default=1)
            stride = _normalize_nd_param(stride, input_dim, default=1)
            padding = _normalize_nd_param(padding, input_dim, default=0)
            dilation = _normalize_nd_param(dilation, input_dim, default=1)

            kernel = weight.detach().cpu().numpy()
            bias_np = bias.detach().cpu().numpy() if bias is not None else None

            if input_dim == 1:
                kernel_size = kernel_size + (1,)
                stride = stride + (1,)
                padding = padding + (0,)
                dilation = dilation + (1,)
                unsqueeze_layer = ctx.network.add_shuffle(input_trt)
                nc.set_layer_precision(ctx, unsqueeze_layer)
                unsqueeze_layer.reshape_dims = tuple([0] * input.ndim) + (1,)
                conv_input = unsqueeze_layer.get_output(0)
            else:
                conv_input = input_trt

            conv_layer = ctx.network.add_convolution_nd(
                input=conv_input,
                num_output_maps=out_channels,
                kernel_shape=kernel_size,
                kernel=kernel,
                bias=bias_np,
            )
            conv_layer.stride_nd = _to_trt_dims(stride)
            conv_layer.padding_nd = _to_trt_dims(padding)
            conv_layer.dilation_nd = _to_trt_dims(dilation)
            if groups is not None:
                conv_layer.num_groups = groups

            if input_dim == 1:
                squeeze_layer = ctx.network.add_shuffle(conv_layer.get_output(0))
                nc.set_layer_precision(ctx, squeeze_layer)
                squeeze_layer.reshape_dims = tuple([0] * input.ndim)
                output._trt = squeeze_layer.get_output(0)
            else:
                output._trt = conv_layer.get_output(0)

        for key in (
            "torch.nn.functional.conv1d",
            "torch.nn.functional.conv2d",
            "torch.nn.functional.conv3d",
        ):
            if key in t2t.CONVERTERS:
                t2t.CONVERTERS[key]["converter"] = convert_conv2d3d_patched

        _TORCH2TRT_PATCHED = True
        print("Applied torch2trt conv dims compatibility patch for TensorRT 8.5.")
    except Exception as exc:
        print(f"Could not apply torch2trt compatibility patch: {exc}")


def maybe_build_single_input_tensorrt(module, sample_input, device, engine_path: str, label: str):
    if device.type != "cuda":
        print("TensorRT disabled: CUDA device not available.")
        return module, False
    if not HAS_TORCH2TRT:
        print("TensorRT disabled: torch2trt is not installed. Falling back to PyTorch.")
        return module, False

    _apply_torch2trt_conv_dims_patch()

    if os.path.exists(engine_path):
        try:
            trt_model = TRTModule()
            trt_model.load_state_dict(torch.load(engine_path))
            trt_model.to(device)
            trt_model.eval()
            print(f"Loaded {label} TensorRT engine from {engine_path}")
            return trt_model, True
        except Exception as exc:
            print(f"Failed to load {label} TensorRT engine, rebuilding. Reason: {exc}")
            if TRT_DEBUG_TRACEBACK:
                traceback.print_exc()

    print(f"Building {label} TensorRT engine. This can take a while on first run...")
    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    try:
        if t0 is not None:
            t0.record()
        trt_model = torch2trt(
            module,
            [sample_input],
            fp16_mode=USE_FP16,
            max_workspace_size=TENSORRT_WORKSPACE_MB * (1 << 20),
        )
        torch.save(trt_model.state_dict(), engine_path)
        if t1 is not None:
            t1.record()
            torch.cuda.synchronize()
            dt_sec = t0.elapsed_time(t1) / 1000.0
            print(f"{label} TensorRT engine built and saved to {engine_path} in {dt_sec:.1f}s")
        else:
            print(f"{label} TensorRT engine built and saved to {engine_path}")
        return trt_model, True
    except Exception as exc:
        print(f"{label} TensorRT build failed; falling back to PyTorch. Reason: {exc}")
        if TRT_DEBUG_TRACEBACK:
            traceback.print_exc()
        return module, False
