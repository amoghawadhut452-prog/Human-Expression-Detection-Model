
import os
import time
from pathlib import Path
from typing import List
import urllib.request

import cv2
import numpy as np

EMOTIONS = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt']

def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download_file(urls: List[str], dst: Path, min_bytes: int = 1024) -> Path:
    """
    Try a list of URLs until one succeeds. Ensures file size > min_bytes.
    """
    ensure_dir(dst.parent)
    for u in urls:
        try:
            print(f"[download] Fetching {u} -> {dst}")
            urllib.request.urlretrieve(u, str(dst))
            if dst.exists() and dst.stat().st_size >= min_bytes:
                print(f"[download] OK: {dst.name} ({dst.stat().st_size} bytes)")
                return dst
            else:
                print(f"[download] File too small from {u}, retrying...")
        except Exception as e:
            print(f"[download] Failed {u}: {e}")
    raise RuntimeError(f"Could not download {dst.name} from any provided URL")

class FPSMeter:
    def __init__(self, alpha: float = 0.9):
        self.t0 = None
        self.ema = None
        self.alpha = alpha

    def update(self) -> float:
        t = time.time()
        if self.t0 is None:
            self.t0 = t
            return 0.0
        dt = t - self.t0
        self.t0 = t
        fps = 1.0 / dt if dt > 0 else 0.0
        self.ema = fps if self.ema is None else self.alpha * self.ema + (1 - self.alpha) * fps
        return fps

    def current(self) -> float:
        return 0.0 if self.ema is None else float(self.ema)

def draw_label(img, text, x, y):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y-h-baseline-4), (x+w+4, y+2), (0,0,0), -1)
    cv2.putText(img, text, (x+2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
