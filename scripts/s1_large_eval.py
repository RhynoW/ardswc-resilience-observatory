"""Sentinel-1「大崩塌查核」定位驗證：只看 ≥2 ha 的塊（連通塊層級），把 S1 變化偵測當作官方事件判釋「漏判查核」。
對 4 個區域（花蓮、A 太魯閣、B 中部卡努、C 嘉義凱米）、各軌道的 VH，K=2／3／4（K 事先固定、只報敏感度，不依事件標籤擇優）：
  參照大塊：事件窗口內事件目錄多邊形聯集的連通塊（8 連通，膨脹 1 格合併相鄰碎塊），面積 ≥ MIN_HA（預設 2 ha）。
  偵測大塊：S1 新增候選（d<中位數−K×MAD，3×3 開運算）連通塊（膨脹 1 格合併）面積 ≥ MIN_HA。
  召回（大塊）＝ 被偵測重疊 ≥10% 面積的參照大塊比例（塊數與面積）。
  精確（大塊）＝ 偵測大塊中，與事件多邊形（含 30 m）重疊 ≥20% 的比例；其餘稱「目錄外大塊」＝漏判查核候選（需人工複核，不等於漏判）。
  對照：事件平移 3 km 後同樣的精確比例（隨機巧合水準）。
限制：參照為衛星判釋（非真值，可能漏掉大崩塌——這正是查核要找的）；單一事件對；MIN_HA、重疊門檻為事先固定。
用法：python scripts/s1_large_eval.py → data/biggis_interp/areas/s1_large_eval.json
"""
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import area_run as AR  # noqa: E402
import area_run_s1 as ARS  # noqa: E402
import s2_area as A  # noqa: E402
import s1_area as S1  # noqa: E402
import reverse_detect as R  # noqa: E402
from landslide_incremental import mask  # noqa: E402

MIN_HA = 2.0
MIN_CELLS = int(MIN_HA / 0.01)
KS = (2.0, 3.0, 4.0)


def comps(m, min_cells):
    lab, n = ndi.label(ndi.binary_dilation(m, iterations=1) & True, structure=np.ones((3, 3)))
    # 以原始遮罩計面積，膨脹僅用於合併
    area = np.bincount(lab[m].ravel(), minlength=n + 1)
    keep = area >= min_cells
    keep[0] = False
    return lab, keep, area


def run_area(tag):
    if tag == "H":
        A.NAME = "hualien_barrier_lake"
        S1.PRE_WIN, S1.POST_WIN = ("2025-05-15", "2025-07-05"), ("2025-10-01", "2025-10-16")
    else:
        AR.setup(tag)
        S1.PRE_WIN, S1.POST_WIN = ARS.WIN[tag]
    tf, (H, W) = A.grid()
    shape = (H, W)
    steep = A.slope(tf, H, W) >= R.SLOPE_MIN
    ev = A.events() if tag != "H" else A.events()
    bb = gpd.GeoSeries([A.box(*A.BBOX)], crs=4326).to_crs(3826).iloc[0]
    ev = {k: g for k, g in ev.items() if g.intersects(bb).any()}
    keys = sorted({p.stem.split("_")[0] for p in (A.AREA / "s1").glob("*_vh.npy") if not p.stem.startswith("new")})
    res = {}
    for key in keys:
        S1.ORBITS = {key: None}
        pre, dpre = S1.stack(key, "pre", "vh")
        post, dpost = S1.stack(key, "post", "vh")
        if pre is None or post is None:
            continue
        lo, hi = max(dpre), min(dpost)
        sel = [g for k, g in ev.items() if lo < k[1] <= hi]
        if not sel:
            continue
        pol = gpd.GeoSeries(pd.concat(sel).values, crs=3826)
        ev_raw = mask(gpd.GeoDataFrame(geometry=pol.values, crs=3826), tf, shape)
        ev_d = ndi.binary_dilation(ev_raw, iterations=3)
        ev_s = ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=pol.translate(xoff=3000.0).values, crs=3826), tf, shape), iterations=3)
        with np.errstate(all="ignore"):
            d = 10 * np.log10(np.maximum(post, 1e-6)) - 10 * np.log10(np.maximum(pre, 1e-6))
        region = np.isfinite(d) & steep & (pre > 1e-4) & (post > 1e-4)
        med = float(np.median(d[region]))
        mad = float(np.median(np.abs(d[region] - med))) * 1.4826
        rlab, rkeep, rarea = comps(ev_raw & region, MIN_CELLS)
        ref_ids = np.flatnonzero(rkeep)
        out = {"pre": dpre[-1], "post": dpost[0], "ref_large_n": int(len(ref_ids)), "ref_large_ha": round(float(rarea[ref_ids].sum() * 0.01), 1), "ref_total_ha": round(float((ev_raw & region).sum() * 0.01), 1), "by_K": {}}
        for K in KS:
            new = R.post((d < med - K * mad) & region) & region
            new = ndi.median_filter(new.astype(np.uint8), size=3).astype(bool) & region
            dlab, dkeep, darea = comps(new, MIN_CELLS)
            det_ids = np.flatnonzero(dkeep)
            hit = 0
            hit_ha = 0.0
            for i in ref_ids:
                m = rlab == i
                ov = (new & m).sum() / max((ev_raw & region & m).sum(), 1)
                if ov >= 0.10:
                    hit += 1
                    hit_ha += (ev_raw & region & m).sum() * 0.01
            inev = 0
            outside = []
            chance = 0
            for i in det_ids:
                m = (dlab == i) & new
                f = (m & ev_d).sum() / max(m.sum(), 1)
                if f >= 0.2:
                    inev += 1
                else:
                    outside.append(round(float(m.sum() * 0.01), 1))
                if (m & ev_s).sum() / max(m.sum(), 1) >= 0.2:
                    chance += 1
            out["by_K"][str(K)] = {"det_large_n": int(len(det_ids)), "det_large_ha": round(float(darea[det_ids].sum() * 0.01), 1), "recall_large_n": f"{hit}/{len(ref_ids)}",
                                   "recall_large_frac": round(hit / len(ref_ids), 3) if len(ref_ids) else None, "recall_large_area_frac": round(hit_ha / max(float(rarea[ref_ids].sum() * 0.01), 1e-9), 3),
                                   "precision_large": round(inev / len(det_ids), 3) if len(det_ids) else None, "precision_large_n": f"{inev}/{len(det_ids)}",
                                   "chance_precision_shift3km": round(chance / len(det_ids), 3) if len(det_ids) else None,
                                   "outside_catalog_n": len(outside), "outside_catalog_ha": round(float(sum(outside)), 1), "outside_catalog_sizes_ha": sorted(outside, reverse=True)[:10]}
        res[key] = out
        print(tag, key, json.dumps(out, ensure_ascii=False), flush=True)
    return res


if __name__ == "__main__":
    out = {"MIN_HA": MIN_HA, "areas": {}}
    for tag in ("H", "A", "B", "C"):
        out["areas"][tag] = run_area(tag)
    (REPO / "data" / "biggis_interp" / "areas" / "s1_large_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
