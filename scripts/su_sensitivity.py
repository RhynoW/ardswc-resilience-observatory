"""Slope Unit 參數敏感度：STREAM_HA（河網門檻）× MIN_SU_HA（最小單元）逐一改變，與基準（10, 1）比較。
各變體在 data/biggis_interp/poc/su_sens/<tag>/ 重算（不覆蓋正式 su_labels.npz）。
比較：POC 內 SU 數、面積分位數、「曾有 ≥0.5 ha 裸露」的 SU 數、四年聯集裸露中被這些 SU 涵蓋的比例、
      裸露面積前 100 名 SU 的聯集與基準的 Jaccard（排序穩定度）、基準前 100 名中有幾個在變體中仍有 ≥50% 面積落在變體前 100 名內。
用法：python scripts/su_sensitivity.py run   # 重算各變體（每個數分鐘）
      python scripts/su_sensitivity.py eval  # 比較
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SENS = REPO / "data" / "biggis_interp" / "poc" / "su_sens"
VARIANTS = {"s10_m1": (10, 1), "s5_m1": (5, 1), "s20_m1": (20, 1), "s10_m05": (10, 0.5), "s10_m2": (10, 2)}


def run():
    for tag, (s, m) in VARIANTS.items():
        d = SENS / tag
        if (d / "su_labels.npz").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        code = (f"import sys; sys.path.insert(0, r'{REPO / 'scripts'}'); import slope_units as S; from pathlib import Path; "
                f"d=Path(r'{d}'); S.POC=d; S.WORK=d/'su_work'; S.STREAM_HA={s}; S.MIN_SU_HA={m}; S.main()")
        print(tag, flush=True)
        subprocess.run([sys.executable, "-c", code], check=True, env={**__import__('os').environ, "PYTHONUTF8": "1"})
        # 釋放空間：中間水文檔不需保留
        for f in (d / "su_work").glob("*.tif"):
            f.unlink()


def labels_on_grid(tag, tf, shape, RES):
    L = np.load(SENS / tag / "su_labels.npz")
    key20, sx0, sy0 = L["key20"], float(L["x0"]), float(L["y0"])
    h, w = shape
    x = tf.c + (np.arange(w) + 0.5) * RES
    y = tf.f - (np.arange(h) + 0.5) * RES
    cols = np.floor((x - sx0) / 20.0).astype(int)
    rows = np.floor((sy0 - y) / 20.0).astype(int)
    ok = ((rows >= 0) & (rows < key20.shape[0]))[:, None] & ((cols >= 0) & (cols < key20.shape[1]))[None, :]
    suk = np.full(shape, -1, np.int64)
    rr, cc = np.nonzero(ok)
    suk[rr, cc] = key20[rows[rr], cols[cc]]
    return suk


def evaluate():
    sys.path.insert(0, str(REPO / "scripts"))
    from landslide_incremental import RES, grid
    tf, shape = grid()
    Z = np.load(REPO / "data" / "biggis_interp" / "poc" / "incremental_masks.npz")
    ever = np.zeros(shape, bool)
    for k in Z.files:
        ever |= Z[k]
    out, top = {}, {}
    for tag in VARIANTS:
        suk = labels_on_grid(tag, tf, shape, RES)
        v = suk >= 0
        u, inv = np.unique(suk[v], return_inverse=True)
        area = np.bincount(inv) * 0.01
        bare = np.bincount(inv, weights=ever[v].astype(float)) * 0.01
        sel = bare >= 0.5
        t100 = np.argsort(-bare)[:100]
        mask_top = np.zeros(shape, bool)
        lut = np.zeros(len(u), bool); lut[t100] = True
        mask_top[v] = lut[inv]
        top[tag] = (mask_top, suk, u, bare)
        out[tag] = {"n_su": int(len(u)), "area_ha_p5_50_95": np.percentile(area, [5, 50, 95]).round(1).tolist(),
                    "n_su_ge0.5ha_bare": int(sel.sum()), "share_ever_bare_in_those_su": round(float(bare[sel].sum() / max(bare.sum(), 1e-9)), 3)}
    base = top["s10_m1"][0]
    for tag in VARIANTS:
        m = top[tag][0]
        out[tag]["top100_union_jaccard_vs_base"] = round(float((m & base).sum() / max((m | base).sum(), 1)), 3)
        print(tag, out[tag], flush=True)
    (SENS / "sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"run": run, "eval": evaluate}[sys.argv[1]]()
