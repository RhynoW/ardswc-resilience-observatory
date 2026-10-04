"""反向流程補充：空間緩衝評估。5 km 棋盤格相鄰格互為訓練／測試，邊界兩側會有空間自相關；
此腳本在每個格邊界兩側各排除 BUF 公尺（訓練與測試皆排除），重選 B 的 ExG 門檻並評估，與無緩衝結果並列。
輸出：data/biggis_interp/poc/reverse/eval_buffer.json
用法：python scripts/reverse_buffer_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import reverse_detect as R  # noqa: E402
from landslide_incremental import RES, grid  # noqa: E402

BUFS = [0, 500, 1000, 2000]


def main():
    tf, (H, W) = grid()
    feats = {}
    for y, dt in R.DATES.items():
        z = np.load(R.OUT / f"feat_{dt}.npz")
        feats[y] = (z["exg"], z["bright"], z["valid"])
    gt = {y: v for y, v in ((y, np.load(R.POC / "incremental_masks.npz")[y]) for y in R.DATES)}
    region0 = np.load(R.OUT / "region.npy")
    train0 = np.load(R.OUT / "train.npy")
    xs = tf.c + (np.arange(W) + 0.5) * RES
    ys = tf.f - (np.arange(H) + 0.5) * RES
    dx = np.minimum(xs % R.BLOCK_M, R.BLOCK_M - xs % R.BLOCK_M)
    dy = np.minimum(ys % R.BLOCK_M, R.BLOCK_M - ys % R.BLOCK_M)
    edge = np.minimum(dy[:, None], dx[None, :])       # 到最近格邊界的距離

    def det_for(y, T):
        ex, br, va = feats[y]
        return R.post(((np.nan_to_num(ex, nan=1.0) < T) & (br >= 0.5)) & va)

    out = {}
    for buf in BUFS:
        keep = region0 & (edge >= buf)
        tr, te = keep & train0, keep & ~train0
        best = None
        for T in np.arange(-0.10, 0.25, 0.02):
            f1 = []
            for y in R.DATES:
                m = R.metrics(det_for(y, T), gt[y], tr)
                f1.append(2 * m["tp"] / max(2 * m["tp"] + m["fp"] + m["fn"], 1))
            s = float(np.mean(f1))
            if best is None or s > best[1]:
                best = (round(float(T), 2), s)
        r = {"T_selected": best[0], "train_km2": round(float(tr.sum() * 1e-4), 1), "test_km2": round(float(te.sum() * 1e-4), 1)}
        for y in R.DATES:
            d = det_for(y, best[0])
            r[y] = {"test": R.metrics(d, gt[y], te), "A_fixed_test": R.metrics(det_for(y, R.T_FIXED), gt[y], te)}
        out[str(buf)] = r
        print(buf, r["T_selected"], r["train_km2"], r["test_km2"],
              {y: (r[y]["test"]["recall"], r[y]["test"]["iou"], r[y]["A_fixed_test"]["recall"]) for y in R.DATES}, flush=True)
    (R.OUT / "eval_buffer.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
