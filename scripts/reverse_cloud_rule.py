"""雲遮罩重做第 2 步：區塊層級規則。
偵測塊（基準 A 的連通塊）若有 ≥FRAC 的格子「像雲」（飽和度 <SAT 且偏紅度 <RB；特徵見 reverse_cloud_features.py）即整塊移除。
理由：雲塊整塊低飽和、近中性色；崩塌地內部只有少數格子偏淺，不會整塊達標。規則門檻用 5 個雲樣本與官方崩塌像素的分布決定（探索見註解），
不是以官方圖層召回最大化調出。輸出 poc/reverse/eval_cloudrule.json 與 bare_AC_<date>.npy（移除雲塊後的偵測）。
用法：python scripts/reverse_cloud_rule.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import reverse_detect as R  # noqa: E402
from landslide_incremental import RES, grid  # noqa: E402

SAT, RB, FRAC, SD = 0.14, 0.07, 0.5, 9


def remove_cloud_blocks(det, F, region):
    like = (~np.isnan(F).any(-1)) & (F[..., 0] < SAT) & (F[..., 1] < RB) & (F[..., 3] < SD)
    lab, n = ndi.label(det, structure=np.ones((3, 3)))
    tot = np.bincount(lab.ravel(), minlength=n + 1)
    cl = np.bincount(lab[like].ravel(), minlength=n + 1)
    bad = np.zeros(n + 1, bool)
    bad[1:] = cl[1:] / np.maximum(tot[1:], 1) >= FRAC
    return det & ~bad[lab], int(bad.sum())


def main():
    tf, shape = grid()
    region = np.load(R.OUT / "region.npy")
    Z = np.load(R.POC / "incremental_masks.npz")
    out = {"params": {"SAT": SAT, "RB": RB, "FRAC": FRAC}}
    kept = {}
    for y, dt in R.DATES.items():
        F = np.load(R.OUT / f"cloudfeat4_{dt}.npy").astype(np.float32)
        d0 = np.load(R.OUT / f"bare_A_{dt}.npy")
        d1, nbad = remove_cloud_blocks(d0, F, region)
        np.save(R.OUT / f"bare_AC_{dt}.npy", d1)
        kept[y] = d1
        out[y] = {"A": R.metrics(d0, Z[y], region), "A+cloudblocks": R.metrics(d1, Z[y], region), "gt_components": R.comp_recall(d1, Z[y], region),
                  "removed_blocks": nbad, "removed_ha": round(float((d0 & ~d1).sum() * 0.01), 1)}
        print(y, "removed blocks", nbad, "ha", out[y]["removed_ha"], "recall", out[y]["A"]["recall"], "→", out[y]["A+cloudblocks"]["recall"],
              "prec", out[y]["A"]["precision_lower_bound"], "→", out[y]["A+cloudblocks"]["precision_lower_bound"], flush=True)
    # 複核樣本
    K = json.load(open(R.OUT / "review" / "key.json", encoding="utf-8"))
    V = json.load(open(R.OUT / "review" / "verdicts.json", encoding="utf-8"))
    d23 = np.load(R.OUT / f"bare_A_{R.DATES['2023']}.npy")
    lab, _ = ndi.label(d23 & region, structure=np.ones((3, 3)))
    res = {}
    for rid, k in K.items():
        c, r = int((tf.f - k["cy"]) // RES), int((k["cx"] - tf.c) // RES)
        l = lab[c, r]
        if l == 0:
            w = lab[max(c - 15, 0):c + 16, max(r - 15, 0):r + 16]
            v = w[w > 0]
            l = int(np.bincount(v).argmax()) if len(v) else 0
        m = lab == l
        res.setdefault(V[rid]["class"], []).append(round(float(1 - kept["2023"][m].sum() / max(m.sum(), 1)), 2))
    out["review_removed_frac_by_class"] = res
    print(res)
    (R.OUT / "eval_cloudrule.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
