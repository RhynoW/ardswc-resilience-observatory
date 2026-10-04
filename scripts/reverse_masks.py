"""反向流程補強：雲遮罩＋河道遮罩，並重評估（基準 A）。
雲：像素 V≥150 且飽和度 <0.25（白灰霧狀），盒式平均到 10 m，≥0.4 判為雲，再膨脹 3 格（含雲影邊緣）。
河道：D8 流量累積 ≥ ACC_HA 公頃的河網，膨脹 DIL 格（20 m 格網）後最近鄰對位到 10 m 格網。
用目視複核的 29 個樣本檢查：被遮罩移除的偵測塊比例（雲 5、河床／河谷 7，崩塌 1＝R05 不應被移除）；
並重算全評估範圍的像素召回與精確率下限，量化遮罩的代價。
輸出：poc/reverse/cloud_<date>.npy、river.npy、eval_masks.json
用法：python scripts/reverse_masks.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import reverse_detect as R  # noqa: E402
from landslide_incremental import RES, grid  # noqa: E402

ACC_HA, DIL = 200.0, 5        # 20 m 格網上河網膨脹 5 格＝約 100 m
CLOUD_V, CLOUD_S, CLOUD_FRAC, CLOUD_DIL = 150, 0.25, 0.4, 3


def cloud_pixel(img):
    f = img.astype(np.float32)
    mx, mn = f.max(axis=2), f.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    return ((mx >= CLOUD_V) & (sat < CLOUD_S)).astype(np.float32), mx


def main():
    tf, (H, W) = grid()
    fps = json.load(open(R.POC / "wayback_footprints.json", encoding="utf-8"))
    tx, ty, px, py = R.cell_index(H, W, tf)
    cloud = {}
    orig = R.pixel_bare
    for y, dt in R.DATES.items():
        f = R.OUT / f"cloudfeat_{dt}.npy"
        if f.exists():
            c = np.load(f)
        else:
            R.pixel_bare = cloud_pixel
            try:
                c, _, _ = R.build_features(dt, tx, ty, px, py, H, W, fps)
            finally:
                R.pixel_bare = orig
            np.save(f, c)
        cloud[y] = ndi.binary_dilation(np.nan_to_num(c, nan=0.0) >= CLOUD_FRAC, iterations=CLOUD_DIL)
        np.save(R.OUT / f"cloud_{dt}.npy", cloud[y])
    del tx, ty, px, py
    # 河道遮罩
    with rasterio.open(R.POC / "su_work" / "acc.tif") as s:
        acc = s.read(1)
        left, top = s.bounds.left, s.bounds.top
    riv20 = ndi.binary_dilation(acc >= ACC_HA * 1e4 / 400.0, iterations=DIL)
    xs = tf.c + (np.arange(W) + 0.5) * RES
    ys = tf.f - (np.arange(H) + 0.5) * RES
    cj = np.clip(((xs - left) / 20.0).astype(int), 0, riv20.shape[1] - 1)
    ri = np.clip(((top - ys) / 20.0).astype(int), 0, riv20.shape[0] - 1)
    river = riv20[np.ix_(ri, cj)]
    np.save(R.OUT / "river.npy", river)
    # 重評估
    region = np.load(R.OUT / "region.npy")
    Z = np.load(R.POC / "incremental_masks.npz")
    gt = {y: Z[y] for y in R.DATES}
    out = {"params": {"ACC_HA": ACC_HA, "DIL": DIL, "CLOUD_V": CLOUD_V, "CLOUD_S": CLOUD_S, "CLOUD_FRAC": CLOUD_FRAC, "CLOUD_DIL": CLOUD_DIL},
           "cloud_frac_of_region": {y: round(float(cloud[y][region].mean()), 4) for y in R.DATES}, "river_frac_of_region": round(float(river[region].mean()), 4)}
    for y, dt in R.DATES.items():
        d0 = np.load(R.OUT / f"bare_A_{dt}.npy")
        row = {"A": R.metrics(d0, gt[y], region)}
        d1 = d0 & ~cloud[y]
        row["A+cloud"] = R.metrics(d1, gt[y], region)
        d2 = d1 & ~river
        row["A+cloud+river"] = R.metrics(d2, gt[y], region)
        row["gt_components_A+cloud+river"] = R.comp_recall(d2, gt[y], region)
        out[y] = row
    # 複核樣本
    K = json.load(open(R.OUT / "review" / "key.json", encoding="utf-8"))
    V = json.load(open(R.OUT / "review" / "verdicts.json", encoding="utf-8"))
    d23 = np.load(R.OUT / f"bare_A_{R.DATES['2023']}.npy")
    lab, n = ndi.label(d23 & region, structure=np.ones((3, 3)))
    res = {}
    for rid, k in K.items():
        c, r = int((tf.f - k["cy"]) // RES), int((k["cx"] - tf.c) // RES)
        l = lab[c, r] if lab[c, r] > 0 else None
        if l is None:  # 質心可能落在塊外：取最近的標籤
            win = lab[max(c - 15, 0):c + 16, max(r - 15, 0):r + 16]
            vals = win[win > 0]
            l = int(np.bincount(vals).argmax()) if len(vals) else None
        if l is None:
            res[rid] = None
            continue
        m = lab == l
        res[rid] = {"class": V[rid]["class"], "cloud_frac": round(float(cloud["2023"][m].mean()), 2), "river_frac": round(float(river[m].mean()), 2)}
    out["review_sample"] = res
    removed = {}
    for rid, v in res.items():
        if v is None:
            continue
        removed.setdefault(v["class"], []).append(round(max(v["cloud_frac"], v["river_frac"]), 2))
    out["review_removed_max_frac_by_class"] = removed
    (R.OUT / "eval_masks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "review_sample"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
