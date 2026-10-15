"""Slope Unit v2：集水區重疊法（正、反 DEM 集水區疊合；依陳振宇委員 2026-10-08 建議）。
v1（slope_units.py）以「河段子集水區＋沿流向分左右岸」切半坡面，單元偏窄、常不連續。v2 改用文獻常見的集水區重疊法：
  1. 正 DEM：填窪 → D8 → 累積 → 河網（門檻 STREAM_HA）→ 河段子集水區（邊界是稜線）。
  2. 反 DEM（zmax − DEM）：同樣流程；反 DEM 的「河網」就是原地形的稜線，子集水區邊界是谷線。
  3. 兩組子集水區疊合（取交集），每個組合再依 4 連通拆成連續區塊 → 以稜線與谷線為界的坡面。
  4. 小於 MIN_SU_HA 的區塊併入共用邊界最長的相鄰單元（反覆數輪），所以每個單元都是單一連續區塊。
有效坡面（≥15°）：同一 DEM 的坡度先做 3×3 平均再取 ≥15°，每個單元內移除 <EFF_MIN_HA 的有效碎塊、填補 <EFF_MIN_HA 的緩坡孔洞
  （v1 未平滑、未去碎屑，有效坡面破碎）；原始單元幾何保留不變。
SU_ID：SU2-<gx>-<gy>，取單元內最低格的 TWD97 20 m 格網索引；版本字串記錄 DEM、解析度、門檻、最小面積、坡度門檻與平滑。
用法：python scripts/slope_units_v2.py [--sens]   （--sens 只跑門檻敏感度，輸出統計不存標籤）
輸出：poc/su_labels_v2.npz（key20 原始、key_eff 有效）、poc/su_meta_v2.json、poc/su_work_v2/（WhiteboxTools 中間檔）
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from skimage.measure import label as sklabel

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
WORK = POC / "su_work_v2"
DEM_V1 = POC / "su_work" / "dem.tif"      # 與 v1 同一份裁切（POC＋6 km 緩衝）
STREAM_HA = float(os.environ.get("SU2_STREAM_HA", 20.0))
MIN_SU_HA = float(os.environ.get("SU2_MIN_HA", 2.0))
SLOPE_MIN = 15.0
EFF_MIN_HA = 0.2
PX_HA = 400.0 / 1e4


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def basins(src_tif, tag, stream_ha):
    import whitebox
    w = whitebox.WhiteboxTools()
    w.set_verbose_mode(False)
    d = str(WORK) + os.sep
    f, p, a, s, b = (d + f"{tag}_{x}.tif" for x in ("filled", "d8", "acc", "streams", "sub"))
    if not os.path.exists(f):
        w.breach_depressions_least_cost(src_tif, f, dist=200, fill=True)
    if not os.path.exists(p):
        w.d8_pointer(f, p)
        w.d8_flow_accumulation(p, a, out_type="cells", pntr=True)
    s = d + f"{tag}_streams_{stream_ha:g}.tif"
    b = d + f"{tag}_sub_{stream_ha:g}.tif"
    if not os.path.exists(b):
        w.extract_streams(a, s, threshold=stream_ha * 1e4 / 400.0)
        w.subbasins(p, s, b)
    with rasterio.open(b) as r:
        return r.read(1).astype(np.int64)


def merge_small(lab, min_px, rounds=12):
    """lab：0..n-1 有效、-1 無效。小區塊併入共用邊界最長的鄰居。"""
    for it in range(rounds):
        n = int(lab.max()) + 1
        cnt = np.bincount(lab[lab >= 0], minlength=n)
        small = (cnt > 0) & (cnt < min_px)
        if not small.any():
            break
        pa = np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()])
        pb = np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()])
        m = (pa != pb) & (pa >= 0) & (pb >= 0)
        a = np.concatenate([pa[m], pb[m]])
        b = np.concatenate([pb[m], pa[m]])
        k = small[a]
        a, b = a[k], b[k]
        if len(a) == 0:
            break
        code, c = np.unique(a * n + b, return_counts=True)
        ua, ub = code // n, code % n
        # 每個小區塊取共用邊界最多者；同分取面積較大者
        order = np.lexsort((-cnt[ub], -c, ua))
        ua, ub = ua[order], ub[order]
        first = np.r_[True, ua[1:] != ua[:-1]]
        tgt = np.arange(n)
        tgt[ua[first]] = ub[first]
        # 兩個小區塊互指：只讓編號大的併入編號小的
        mutual = (tgt[tgt] == np.arange(n)) & (tgt != np.arange(n))
        tgt[mutual & (np.arange(n) < tgt)] = np.arange(n)[mutual & (np.arange(n) < tgt)]
        for _ in range(20):
            tgt = tgt[tgt]
        lab = np.where(lab >= 0, tgt[np.clip(lab, 0, None)], -1)
        _, lab_c = np.unique(lab, return_inverse=True)
        lab = lab_c.reshape(lab.shape) - (1 if (lab < 0).any() else 0)
        log(f"  合併第 {it + 1} 輪：小區塊 {int(small.sum())} 個 → 單元 {int(lab.max()) + 1}")
    return lab


def build(stream_ha, min_ha):
    WORK.mkdir(parents=True, exist_ok=True)
    with rasterio.open(DEM_V1) as r:
        dem = r.read(1).astype(np.float32)
        prof = r.profile
        tf = r.transform
    valid = dem > -9990
    inv_tif = WORK / "inv.tif"
    if not inv_tif.exists():
        inv = np.where(valid, float(dem[valid].max()) - dem, -9999.0).astype(np.float32)
        with rasterio.open(inv_tif, "w", **prof) as d_:
            d_.write(inv, 1)
    s_norm = basins(str(DEM_V1), "norm", stream_ha)
    s_inv = basins(str(inv_tif), "inv", stream_ha)
    ok = valid & (s_norm > 0) & (s_inv > 0)
    combo = np.where(ok, s_norm * (int(s_inv.max()) + 1) + s_inv, 0)
    lab = sklabel(combo, background=0, connectivity=1) - 1     # 同組合再依 4 連通拆開
    log(f"疊合後區塊 {int(lab.max()) + 1}（門檻 {stream_ha} ha）")
    lab = merge_small(lab, int(round(min_ha / PX_HA)))
    return dem, valid, tf, lab


def stats(lab):
    n = int(lab.max()) + 1
    cnt = np.bincount(lab[lab >= 0], minlength=n)
    # 周長（格邊數）：相鄰不同標籤或無效的邊
    per = np.zeros(n)
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = a != b
        for x in (a[m], b[m]):
            x = x[x >= 0]
            per += np.bincount(x, minlength=n)
    edge = np.zeros_like(lab, dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    per += np.bincount(lab[edge & (lab >= 0)], minlength=n)
    A = cnt * 400.0
    P = per * 20.0
    pp = 4 * np.pi * A / np.maximum(P, 1) ** 2
    w = 2 * A / np.maximum(P, 1)
    return {"n": n, "area_ha_q": np.percentile(cnt * PX_HA, [5, 25, 50, 75, 95]).round(1).tolist(),
            "width_m_median": round(float(np.median(w))), "share_width_lt80": round(float((w < 80).mean()), 3),
            "pp_median": round(float(np.median(pp)), 3)}


def effective(dem, valid, lab):
    gy, gx = np.gradient(np.where(valid, dem, np.nan), 20.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    sm = ndi.uniform_filter(np.nan_to_num(slope), 3)
    eff = (sm >= SLOPE_MIN) & valid & (lab >= 0)
    mpx = int(round(EFF_MIN_HA / PX_HA))
    # 單元內：移除小有效碎塊、填補小緩坡孔洞（以「有效＋單元標籤」的連通區塊判斷，碎塊不跨單元）
    k_eff = np.where(eff, lab + 1, 0)
    cc = sklabel(k_eff, background=0, connectivity=1)
    sz = np.bincount(cc.ravel())
    eff &= ~((cc > 0) & (sz[cc] < mpx))
    k_hole = np.where((lab >= 0) & ~eff, lab + 1, 0)
    hc = sklabel(k_hole, background=0, connectivity=1)
    hs = np.bincount(hc.ravel())
    eff |= (hc > 0) & (hs[hc] < mpx)
    return np.where(eff, lab, -1), slope


def main():
    if "--sens" in sys.argv:
        out = {}
        for s_ha in (10.0, 20.0, 50.0):
            _, _, _, lab = build(s_ha, MIN_SU_HA)
            out[f"S{s_ha:g}_M{MIN_SU_HA:g}"] = stats(lab)
            log(out[f"S{s_ha:g}_M{MIN_SU_HA:g}"])
        (WORK / "sensitivity_v2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return
    dem, valid, tf, lab = build(STREAM_HA, MIN_SU_HA)
    key_eff, slope = effective(dem, valid, lab)
    n = int(lab.max()) + 1
    idx = np.arange(n)
    # SU_ID：單元內最低格
    rr, cc = np.nonzero(lab >= 0)
    order = np.lexsort((dem[rr, cc], lab[rr, cc]))
    ls = lab[rr, cc][order]
    first = np.r_[True, ls[1:] != ls[:-1]]
    low = {int(l_): (int(r_), int(c_)) for l_, r_, c_ in zip(ls[first], rr[order][first], cc[order][first])}
    ver = f"SUv2-D20m-WO-S{STREAM_HA:g}-M{MIN_SU_HA:g}-E15s3-F{EFF_MIN_HA:g}"
    area = np.bincount(lab[lab >= 0], minlength=n) * PX_HA
    eff_area = np.bincount(key_eff[key_eff >= 0], minlength=n) * PX_HA
    s_all = ndi.mean(np.nan_to_num(slope), lab + 1, idx + 1)
    s_eff = ndi.mean(np.nan_to_num(slope), key_eff + 1, idx + 1)
    units = []
    for i in range(n):
        r0, c0 = low[i]
        sid = f"SU2-{int((tf.c + (c0 + 0.5) * 20) // 20)}-{int((tf.f - (r0 + 0.5) * 20) // 20)}"
        units.append({"su_id": sid, "su_uid": f"{sid}@{ver}", "key": i, "area_ha": round(float(area[i]), 2), "eff_area_ha": round(float(eff_area[i]), 2),
                      "eff_frac": round(float(eff_area[i] / max(area[i], 1e-9)), 3), "mean_slope_deg_full": round(float(s_all[i]), 1),
                      "mean_slope_deg": round(float(s_eff[i]), 1) if eff_area[i] > 0 else None})
    assert len({u["su_id"] for u in units}) == n
    st = stats(lab)
    st_eff_parts = None
    np.savez_compressed(POC / "su_labels_v2.npz", key20=lab.astype(np.int32), key_eff=key_eff.astype(np.int32), x0=tf.c, y0=tf.f, res=20.0)
    meta = {"params": {"su_version": ver, "method": "集水區重疊法（正 DEM 與反 DEM 的河段子集水區疊合，4 連通拆塊，小區塊併入共用邊界最長的鄰居）",
                       "dtm": "data/dtm/tw_dtm20.tif（20 m；與 v1 同一裁切）", "stream_ha": STREAM_HA, "min_su_ha": MIN_SU_HA, "slope_min_deg": SLOPE_MIN,
                       "slope_smoothing": "3x3 平均", "eff_min_fragment_ha": EFF_MIN_HA, "origin_xy": [tf.c, tf.f], "shape": list(lab.shape),
                       "built": time.strftime("%Y-%m-%d"), "crosswalk_note": "與 v1 以空間重疊對應（su_crosswalk_v1_v2.json）"},
            "stats": st, "n_units": n, "units": units}
    (POC / "su_meta_v2.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    log("完成", ver, st)


if __name__ == "__main__":
    main()
