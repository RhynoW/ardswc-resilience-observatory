"""Sentinel-2（Copernicus WMTS，TRUE_COLOR，8 位元）取 POC 全範圍，重採樣到 10 m TM2 分析格網。
日期：2024-04-04（雲量 0.9%，凱米颱風前）與 2025-03-25（雲量 0.1%，約與 113 年度圖層影像期相近）。
圖磚左下角有 Copernicus 浮水印，視為無效格（不補洞）。輸出 poc/s2/rgb_<date>.npy（H×W×3 uint8）、valid_<date>.npy。
注意：WMTS 為 8 位元顯示用影像（非反射率），指標只用比值，門檻不可直接沿用 Wayback。
用法：python scripts/s2_poc_fetch.py
"""
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import sentinel_assist as S  # noqa: E402
from landslide_incremental import RES, grid  # noqa: E402

OUT = REPO / "data" / "biggis_interp" / "poc" / "s2"
DATES = ["20240404", "20250325"]
WORLD = 2 * math.pi * S.R_MERC
P = WORLD / (S.TILE_PX * 2 ** (S.TILE_MATRIX - 1))     # m/px


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tf, (H, W) = grid()
    to_m = Transformer.from_crs(3826, 3857, always_xy=True)
    xs = tf.c + (np.arange(W) + 0.5) * RES
    ys = tf.f - (np.arange(H) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    mx, my = to_m.transform(X.ravel(), Y.ravel())
    gx = (np.asarray(mx) + WORLD / 2) / P
    gy = (WORLD / 2 - np.asarray(my)) / P
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    col, row = ix // S.TILE_PX, iy // S.TILE_PX
    px, py = (ix % S.TILE_PX).astype(np.int32), (iy % S.TILE_PX).astype(np.int32)
    tiles = sorted(set(zip(row.tolist(), col.tolist())))
    log("需要圖磚", len(tiles))
    iid = S.instance_id()
    wm = np.zeros((S.TILE_PX, S.TILE_PX), bool)
    wm[S.TILE_PX - S.WATERMARK_BOX_PX[1]:, :S.WATERMARK_BOX_PX[0]] = True
    key = row * 100000 + col
    order = np.argsort(key, kind="stable")
    ks = key[order]
    b = np.flatnonzero(np.diff(ks)) + 1
    starts, ends = np.concatenate([[0], b]), np.concatenate([b, [len(ks)]])
    for ymd in DATES:
        if (OUT / f"rgb_{ymd}.npy").exists():
            continue

        def get(rc):
            try:
                return rc, S._fetch_tile(iid, ymd, "TRUE_COLOR", rc[0], rc[1])
            except Exception as e:  # noqa: BLE001
                log("失敗", rc, repr(e)[:80])
                return rc, None
        with ThreadPoolExecutor(4) as ex:
            got = dict(ex.map(get, tiles))
        log(ymd, "取得", sum(v is not None for v in got.values()), "/", len(tiles))
        rgb = np.zeros((H * W, 3), np.uint8)
        valid = np.zeros(H * W, bool)
        for s, e in zip(starts, ends):
            k = int(ks[s])
            t = got.get((k // 100000, k % 100000))
            if t is None:
                continue
            idx = order[s:e]
            rgb[idx] = t[py[idx], px[idx]][:, ::-1] if t.shape[2] == 3 else t[py[idx], px[idx], :3][:, ::-1]
            valid[idx] = ~wm[py[idx], px[idx]]
        np.save(OUT / f"rgb_{ymd}.npy", rgb.reshape(H, W, 3))
        np.save(OUT / f"valid_{ymd}.npy", valid.reshape(H, W))
        log(ymd, "完成；有效比例", round(float(valid.mean()), 3))


if __name__ == "__main__":
    main()
