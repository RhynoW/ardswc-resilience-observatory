"""參照崩塌塊「命中」門檻敏感度：重疊面積比例 ≥10%／20%／50%（及 ≥0.1 ha 以上塊），依面積分層。
輸出 poc/reverse/eval_hit_threshold.json。用法：python scripts/reverse_hit_threshold.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import reverse_detect as R  # noqa: E402

TH = (0.1, 0.2, 0.5)


def main():
    region = np.load(R.OUT / "region.npy")
    Z = np.load(R.POC / "incremental_masks.npz")
    out = {}
    for y, dt in R.DATES.items():
        det = np.load(R.OUT / f"bare_A_{dt}.npy") & region
        g = (Z[y] & region).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
        hit = np.bincount(lab[det].ravel(), minlength=n)
        res = {}
        for t in TH:
            row = {"<0.5ha": [0, 0], "0.5–2ha": [0, 0], ">2ha": [0, 0]}
            for i in range(1, n):
                a = st[i, cv2.CC_STAT_AREA] * 0.01
                if a < 0.1:
                    continue
                k = "<0.5ha" if a < 0.5 else ("0.5–2ha" if a <= 2 else ">2ha")
                row[k][1] += 1
                row[k][0] += hit[i] / st[i, cv2.CC_STAT_AREA] >= t
            res[str(t)] = {k: {"hit": int(v[0]), "n": v[1], "rate": round(v[0] / v[1], 3)} for k, v in row.items()}
        out[y] = res
        print(y, {t: {k: v["rate"] for k, v in r.items()} for t, r in res.items()})
    (R.OUT / "eval_hit_threshold.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
