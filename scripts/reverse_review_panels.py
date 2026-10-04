"""反向流程：「影像偵測有、官方無」人工（目視）複核樣本。
取基準 A 在 2023-11-02 的偵測連通塊，排除與官方 2022／2023 圖層（膨脹 3 格）有任何重疊者，依面積三層各抽 10 個（固定亂數種子），
每個輸出一張盲測圖（亂數編號 R01…；左：2023-11-02 影像加輪廓，中：同影像無輪廓，右：2022-08-26 影像），不顯示任何官方資訊。
key.json 存編號對照；verdicts.json 由判讀者填；用法：python scripts/reverse_review_panels.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from landslide_incremental import RES, grid  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "reverse" / "review"
Z, SEED, PER = 17, 20261004, 10
D23, D22 = "2023-11-02", "2022-08-26"
HALF = 160                       # 裁切半徑（像素，z17 約 1.2 m → 約 190 m）
to3857 = Transformer.from_crs(3826, 3857, always_xy=True)
N = 2 ** Z
W2 = 20037508.342789244


def to_px(x, y):
    mx, my = to3857.transform(x, y)
    return (np.asarray(mx) + W2) / (2 * W2) * N * 256, (W2 - np.asarray(my)) / (2 * W2) * N * 256


def crop(date, cx, cy, poly_xy=None):
    gx, gy = to_px(cx, cy)
    x0, y0 = int(gx) - HALF, int(gy) - HALF
    img = np.zeros((2 * HALF, 2 * HALF, 3), np.uint8)
    for tx in range(x0 // 256, (x0 + 2 * HALF) // 256 + 1):
        for ty in range(y0 // 256, (y0 + 2 * HALF) // 256 + 1):
            p = POC / "wb_tiles" / date / f"{tx}_{ty}.jpg"
            if not p.exists():
                continue
            t = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if t is None:
                continue
            ax, ay = tx * 256 - x0, ty * 256 - y0
            sx0, sy0, sx1, sy1 = max(ax, 0), max(ay, 0), min(ax + 256, 2 * HALF), min(ay + 256, 2 * HALF)
            if sx1 > sx0 and sy1 > sy0:
                img[sy0:sy1, sx0:sx1] = t[sy0 - ay:sy1 - ay, sx0 - ax:sx1 - ax]
    out = img.copy()
    if poly_xy is not None:
        for pts in poly_xy:
            px, py = to_px(pts[:, 0], pts[:, 1])
            cv2.polylines(out, [np.stack([px - x0, py - y0], 1).round().astype(np.int32)], True, (0, 255, 255), 1)
    return out, img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tf, shape = grid()
    det = np.load(POC / "reverse" / f"bare_A_{D23}.npy")
    region = np.load(POC / "reverse" / "region.npy")
    Zm = np.load(POC / "incremental_masks.npz")
    off = ndi.binary_dilation(Zm["2022"] | Zm["2023"], iterations=3)
    lab, n = ndi.label(det & region, structure=np.ones((3, 3)))
    area = np.bincount(lab.ravel(), minlength=n + 1) * 0.01
    ov = np.bincount(lab[off].ravel(), minlength=n + 1)
    cand = [i for i in range(1, n + 1) if ov[i] == 0 and area[i] >= 0.1]
    strata = {"<0.5ha": [i for i in cand if area[i] < 0.5], "0.5–2ha": [i for i in cand if 0.5 <= area[i] <= 2], ">2ha": [i for i in cand if area[i] > 2]}
    print({k: len(v) for k, v in strata.items()}, "全部偵測塊", n, "無官方重疊", len(cand))
    rng = np.random.default_rng(SEED)
    pick = []
    for k, v in strata.items():
        pick += [(k, int(i)) for i in rng.choice(v, min(PER, len(v)), replace=False)]
    order = rng.permutation(len(pick))
    key = {}
    for j, o in enumerate(order, 1):
        k, i = pick[o]
        rid = f"R{j:02d}"
        m = (lab == i)
        rr, cc = np.nonzero(m)
        cx, cy = tf.c + (cc.mean() + 0.5) * RES, tf.f - (rr.mean() + 0.5) * RES
        cs, _ = cv2.findContours(np.pad(m, 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [np.array([[tf.c + (p[0][0] - 1) * RES, tf.f - (p[0][1] - 1) * RES] for p in c]) for c in cs]
        a, b = crop(D23, cx, cy, polys)
        _, c = crop(D22, cx, cy)
        panel = np.concatenate([a, b, c], 1)
        cv2.putText(panel, rid, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{rid}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        key[rid] = {"stratum": k, "area_ha": round(float(area[i]), 2), "cx": cx, "cy": cy}
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完成", len(key), "張 →", OUT)


if __name__ == "__main__":
    main()
