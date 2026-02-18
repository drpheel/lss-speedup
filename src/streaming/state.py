"""Thread-safe shared runtime state."""

import threading
from typing import Dict, Optional


class SharedState:
    def __init__(self):
        self._cam_jpeg: Optional[bytes] = None
        self._bev_jpeg: Optional[bytes] = None
        self._eval_stats: Dict[str, object] = {}
        self._frame_lock = threading.Lock()
        self._eval_lock = threading.Lock()

    def set_frames(self, cam_jpeg: bytes, bev_jpeg: bytes) -> None:
        with self._frame_lock:
            self._cam_jpeg = cam_jpeg
            self._bev_jpeg = bev_jpeg

    def get_cam_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._cam_jpeg

    def get_bev_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._bev_jpeg

    def set_eval_stats(self, stats: Dict[str, object]) -> None:
        with self._eval_lock:
            self._eval_stats = dict(stats)

    def get_eval_stats(self) -> Dict[str, object]:
        with self._eval_lock:
            if self._eval_stats:
                return dict(self._eval_stats)
        return {"ready": False, "message": "Eval not started yet."}
