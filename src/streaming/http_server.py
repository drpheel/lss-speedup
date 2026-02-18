"""HTTP MJPEG server and handler factory."""

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from .config import FRAME_INTERVAL
from .eval_metrics import render_eval_html
from .state import SharedState


def build_handler(shared_state: SharedState):
    class MJPEGHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            routes = {
                "/cam": self._handle_cam_stream,
                "/bev": self._handle_bev_stream,
                "/eval": self._handle_eval_html,
                "/eval.json": self._handle_eval_json,
            }
            handler = routes.get(self.path)
            if handler is None:
                self.send_error(404)
                self.end_headers()
                return
            handler()

        def _stream_mjpeg(self, frame_getter):
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                frame_data = frame_getter()
                if frame_data is None:
                    time.sleep(0.01)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame_data))
                    self.end_headers()
                    self.wfile.write(frame_data)
                    self.wfile.write(b"\r\n")
                    time.sleep(FRAME_INTERVAL)
                except BrokenPipeError:
                    break

        def _handle_cam_stream(self):
            self._stream_mjpeg(shared_state.get_cam_frame)

        def _handle_bev_stream(self):
            self._stream_mjpeg(shared_state.get_bev_frame)

        def _handle_eval_html(self):
            body = render_eval_html(shared_state.get_eval_stats()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_eval_json(self):
            body = json.dumps(shared_state.get_eval_stats(), indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return MJPEGHandler


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads."""
