"""Application composition: start inference thread and HTTP server."""

import threading

from .config import HOST_IP, PORT
from .http_server import ThreadedHTTPServer, build_handler
from .inference import run_inference_loop
from .state import SharedState


def main() -> None:
    shared_state = SharedState()

    thread = threading.Thread(target=run_inference_loop, args=(shared_state,), daemon=True)
    thread.start()

    print(f"Starting MJPEG Server at http://{HOST_IP}:{PORT}...")
    print(f"  - Camera Stream: http://<JETSON_IP>:{PORT}/cam")
    print(f"  - BEV Stream:    http://<JETSON_IP>:{PORT}/bev")

    handler_cls = build_handler(shared_state)
    server = ThreadedHTTPServer((HOST_IP, PORT), handler_cls)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
