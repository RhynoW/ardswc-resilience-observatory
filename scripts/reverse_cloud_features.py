"""雲遮罩重做第 1 步：為兩個影像日期各算 4 個 10 m 特徵（≈9×9 像素盒式平均），存 poc/reverse/cloudfeat4_<date>.npy（H×W×4 float16）：
  0 飽和度 (max−min)/max；1 偏紅度 (R−B)/(R+B)（裸土偏褐 >0，雲偏白藍 ≈0 或 <0）；2 亮度 V=max(R,G,B)；3 局部紋理＝V 的盒內標準差。
之後在 reverse_cloud_rule.py 以這些特徵試規則（不需重讀圖磚）。
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import reverse_detect as R  # noqa: E402
from landslide_incremental import grid  # noqa: E402


def main():
    tf, (H, W) = grid()
    fps = json.load(open(R.POC / "wayback_footprints.json", encoding="utf-8"))
    tx, ty, px, py = R.cell_index(H, W, tf)
    key = tx.astype(np.int64) * 1_000_000 + ty
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    starts, ends = np.concatenate([[0], bounds]), np.concatenate([bounds, [len(ks)]])
    for dt in R.DATES.values():
        out = R.OUT / f"cloudfeat4_{dt}.npy"
        if out.exists():
            continue
        d = R.POC / "wb_tiles" / dt
        F = np.full((H * W, 4), np.nan, np.float16)
        for s, e in zip(starts, ends):
            p = d / f"{int(ks[s] // 1_000_000)}_{int(ks[s] % 1_000_000)}.jpg"
            if not p.exists():
                continue
            img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (256, 256):
                continue
            f = img.astype(np.float32)
            b, g, r = f[..., 0], f[..., 1], f[..., 2]
            v = f.max(axis=2)
            sat = (v - f.min(axis=2)) / (v + 1e-6)
            rb = (r - b) / (r + b + 1e-6)
            m1 = cv2.blur(v, (9, 9))
            sd = np.sqrt(np.maximum(cv2.blur(v * v, (9, 9)) - m1 * m1, 0))
            idx = order[s:e]
            ys, xs = py[idx], px[idx]
            F[idx, 0], F[idx, 1], F[idx, 2], F[idx, 3] = cv2.blur(sat, (9, 9))[ys, xs], cv2.blur(rb, (9, 9))[ys, xs], m1[ys, xs], sd[ys, xs]
        np.save(out, F.reshape(H, W, 4))
        print(time.strftime("%H:%M:%S"), dt, "done", flush=True)


if __name__ == "__main__":
    main()
