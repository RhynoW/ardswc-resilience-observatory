# -*- coding: utf-8 -*-
"""
事件模式：颱風豪雨後的「雷達先行、光學補證」新生崩塌／堰塞湖候選偵測（Google Earth Engine，離線算好存 JSON＋縮圖）。

對應案例：2025 馬太鞍溪上游（水保署 7/22 以雷達發現「河道突然變一片黑」，7/25 雲散後光學確認堰塞湖與 ≥500 公頃崩塌）。
現有工具（gee_trial.py）只有「乾季 Sentinel-2 dNBR、年度對比」，雲多時完全不能用，且沒有水體；這支補三件事：

  1. 雷達（Sentinel-1 GRD VV，穿雲）：事件前 --pre-days 天 median 當基準，事件後逐次過境各算
       新增暗區 = VV < --dark-db 且基準期不暗（排除原本就暗的河道／海／陰影）、ΔVV > --delta-db、
       且坡度 < --max-slope（排除雷達陰影）。逐次過境輸出面積（公頃），可看水體／堰塞湖何時出現、是否擴大。
     限制：VV 暗 ≠ 水（平滑裸地、雷達陰影、疊掩區也暗；崩塌裸地常使後向散射「變亮」而非變暗，此處只看暗區）。
  2. 光學（Sentinel-2 SR + Cloud Score+）逐日可用性：AOI 內無雲比例，讓人知道「什麼時候能用光學補證」。
  3. 光學補證（事件窗內有效像素足夠才算）：
       植生損失 dNBR > --nbr-thresh 且事件前 NBR 為植生（≥ --veg-nbr）→ 崩塌／裸露候選面積（公頃）
       新增水體 NDWI(B3,B8) > 0 且事件前不是水 → 堰塞湖／淹水候選面積（公頃）
     事件前基準：事件前 --opt-pre-days 天 Cloud Score+ 遮雲後 median；事件後：事件日起 --opt-post-days 天內 median。
     雙期有效像素比例 < --min-valid 一律標 insufficient、不給數值（與 gee_trial 一致）。

所有輸出都是候選訊號，非判釋成果；面積受閾值（未校正）與有效像素影響，需人工覆核（review_waitlist 流程）。

輸出：data/event_watch/<name>/result.json、sar_new_dark_<date>.png、optical_loss.png、optical_water.png
認證／專案：同 gee_trial（GEE_PROJECT 環境變數或 --project；專案 ID 不寫進倉庫）。

用法：
  python scripts/gee_event_watch.py --name mataian_2025 --center 121.36,23.65 --half-km 12 \\
         --event 2025-07-22 --post-days 30 [--project <id>]
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "data" / "event_watch"

S1 = "COPERNICUS/S1_GRD"
S2 = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
DEM = "USGS/SRTMGL1_003"          # 只用來算坡度（遮雷達陰影）；30 m


def init_ee(project):
    try:
        import ee
    except ImportError:
        sys.exit("缺少 earthengine-api：pip install earthengine-api")
    try:
        ee.Initialize(project=project or os.environ.get("GEE_PROJECT"))
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Earth Engine 初始化失敗：{e}\n請設 GEE_PROJECT 或 --project，並先 earthengine authenticate。")
    return ee


def area_ha(ee, mask, geom, scale):
    """mask（0/1 影像）為 1 的面積（m²）；回傳 ee.Number。"""
    r = mask.rename("m").multiply(ee.Image.pixelArea()).reduceRegion(
        ee.Reducer.sum(), geom, scale, maxPixels=1e10, bestEffort=True)
    return r.get("m")


def top_clusters(ee, mask, geom, n=8, scale=30, min_ha=10):
    """mask 的連通塊（8 鄰接）依面積排序，回傳前 n 個 [{area_ha, lon, lat}]——自動指出「事件在哪」。"""
    vec = mask.selfMask().rename("m").reduceToVectors(geometry=geom, scale=scale, maxPixels=1e10, bestEffort=True,
                                                      geometryType="polygon", eightConnected=True, labelProperty="m")
    vec = vec.map(lambda f: f.set({"ha": f.geometry().area(30).divide(1e4)})).filter(ee.Filter.gte("ha", min_ha)).sort("ha", False).limit(n)
    out = []
    for f in vec.getInfo()["features"]:
        ring = f["geometry"]["coordinates"][0]
        lon = sum(p[0] for p in ring) / len(ring); lat = sum(p[1] for p in ring) / len(ring)
        out.append({"area_ha": round(f["properties"]["ha"], 1), "lon": round(lon, 4), "lat": round(lat, 4)})
    return out


def hotspot_context(clusters, path, radius_km=3.0):
    """把事件偵測到的連通塊對到本系統的歷史複發熱點（這是 BigGIS 沒有的：事件落在「常出事」還是「全新」的地方）。"""
    import math
    if not Path(path).exists():
        return
    hs = json.loads(Path(path).read_text(encoding="utf-8"))

    def km(a, b, c, d):
        p = math.pi / 180
        x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
        return 12742 * math.asin(math.sqrt(x))
    for c in clusters:
        near = min(hs, key=lambda h: km(c["lat"], c["lon"], h["lat"], h["lon"]))
        dk = km(c["lat"], c["lon"], near["lat"], near["lon"])
        c["nearest_hotspot"] = {"rank": near["rank"], "dist_km": round(dk, 1), "n_independent_events": near["n_independent_events"],
                                "years": near["years"]}
        c["history"] = "recurrent" if dk <= radius_km else "no_recorded_hotspot_nearby"


def save_thumb(ee, vis_img, geom, path, dim=900):
    url = vis_img.getThumbURL({"region": geom, "dimensions": dim, "format": "png"})
    with urllib.request.urlopen(url, timeout=180) as r:
        Path(path).write_bytes(r.read())


def sar_stage(ee, geom, ev, args, out):
    slope = ee.Terrain.slope(ee.Image(DEM))
    flat = slope.lt(args.max_slope)

    def col(a, b):
        return (ee.ImageCollection(S1).filterBounds(geom).filterDate(str(a), str(b))
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).select("VV"))

    smooth = lambda im: im.focal_median(30, "circle", "meters")      # 抑制斑點雜訊（像素 10 m）
    pre_col = col(ev - dt.timedelta(days=args.pre_days), ev)
    n_pre = pre_col.size().getInfo()
    if n_pre < 3:
        return {"status": "insufficient", "reason": f"事件前 {args.pre_days} 天只有 {n_pre} 景 S1"}
    pre = smooth(pre_col.median())
    pre_dark = pre.lt(args.dark_db)
    post = col(ev, ev + dt.timedelta(days=args.post_days))
    n = post.size().getInfo()
    lst = post.toList(n)
    rows, best = [], None
    for i in range(n):
        im = ee.Image(lst.get(i))
        info = im.getInfo()["properties"]
        date = dt.datetime.fromtimestamp(info["system:time_start"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
        orbit = f'{info.get("orbitProperties_pass", "?")}{info.get("relativeOrbitNumber_start", "")}'
        v = smooth(im)
        new_dark = v.lt(args.dark_db).And(pre_dark.Not()).And(pre.subtract(v).gt(args.delta_db)).And(flat)
        cov = v.mask().reduceRegion(ee.Reducer.mean(), geom, 40, maxPixels=1e9, bestEffort=True).get("VV")
        ha = area_ha(ee, new_dark.unmask(0), geom, 10)
        cov, ha = ee.List([cov, ha]).getInfo()
        cov, ha = cov or 0, (ha or 0) / 1e4
        rows.append({"date": date, "pass": orbit, "coverage": round(cov, 3), "new_dark_ha": round(ha, 1)})
        print(f"  S1 {date} {orbit:>12} 覆蓋 {cov:.2f}  新增暗區 {ha:.1f} ha", flush=True)
        if cov > 0.8 and (best is None or ha > best[0]):
            best = (ha, date, new_dark)
    if best:
        _, date, m = best
        base = pre.visualize(min=-25, max=0)
        save_thumb(ee, base.blend(m.selfMask().visualize(palette=["ff0000"])), geom, out / f"sar_new_dark_{date}.png")
    return {"status": "ok", "n_pre": n_pre, "acquisitions": rows,
            "params": {"dark_db": args.dark_db, "delta_db": args.delta_db, "max_slope": args.max_slope, "pre_days": args.pre_days},
            "note": "面積只含 AOI 被該次過境覆蓋的部分（見 coverage）；coverage < 1 的列不可與滿覆蓋列直接比較"}


def optical_stage(ee, geom, ev, args, out):
    def col(a, b):
        return (ee.ImageCollection(S2).filterBounds(geom).filterDate(str(a), str(b))
                .linkCollection(ee.ImageCollection(CS_PLUS), ["cs_cdf"]))

    allc = col(ev - dt.timedelta(days=1), ev + dt.timedelta(days=args.post_days + 1))
    n_all = allc.size().getInfo()
    lst = allc.toList(n_all)
    byday = {}
    for i in range(n_all):
        im = ee.Image(lst.get(i))
        t = im.get("system:time_start").getInfo()
        clear = im.select("cs_cdf").gte(args.cs_thresh).reduceRegion(
            ee.Reducer.mean(), geom, 60, maxPixels=1e9, bestEffort=True).get("cs_cdf").getInfo() or 0
        k = dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
        byday[k] = max(byday.get(k, 0), round(clear, 3))
    avail = [{"date": k, "clear_frac_best_tile": v} for k, v in sorted(byday.items())]
    for a in avail:
        print(f"  S2 {a['date']} 無雲比例（最佳 tile）{a['clear_frac_best_tile']:.2f}", flush=True)

    mask_cs = lambda im: im.updateMask(im.select("cs_cdf").gte(args.cs_thresh))
    pre_c = col(ev - dt.timedelta(days=args.opt_pre_days), ev).map(mask_cs)
    post_c = col(ev, ev + dt.timedelta(days=args.opt_post_days)).map(mask_cs)
    n_pre, n_post = pre_c.size().getInfo(), post_c.size().getInfo()
    res = {"availability": avail, "n_pre": n_pre, "n_post": n_post}
    if n_pre < 2 or n_post < 1:
        res.update(status="insufficient", reason=f"光學影像不足（事件前 {n_pre} 景、事件後 {n_post} 景）")
        return res
    a, b = pre_c.median(), post_c.median()
    nbr = lambda im: im.normalizedDifference(["B8", "B12"])
    ndwi = lambda im: im.normalizedDifference(["B3", "B8"])
    valid = a.select("B8").mask().And(b.select("B8").mask())
    vf = valid.reduceRegion(ee.Reducer.mean(), geom, 40, maxPixels=1e9, bestEffort=True).get("B8").getInfo() or 0
    res["valid_frac"] = round(vf, 3)
    if vf < args.min_valid:
        res.update(status="insufficient", reason=f"雙期有效像素僅 {vf:.0%}（雲遮）")
        return res
    elev_ok = ee.Image(DEM).gte(args.min_elev)           # 排除平地農田（收割／翻耕會被誤當植生損失）
    loss = elev_ok.And(nbr(a).gte(args.veg_nbr)).And(nbr(a).subtract(nbr(b)).gt(args.nbr_thresh)).And(valid)
    water = elev_ok.And(ndwi(b).gt(0)).And(ndwi(a).lte(0)).And(valid)
    L = area_ha(ee, loss.unmask(0), geom, 20)
    W = area_ha(ee, water.unmask(0), geom, 20)
    L, W = ee.List([L, W]).getInfo()
    res["loss_clusters"] = top_clusters(ee, loss, geom)
    res["water_clusters"] = top_clusters(ee, water, geom)
    for k in ("loss_clusters", "water_clusters"):
        hotspot_context(res[k], args.hotspots)
    res.update(status="ok", loss_ha=round((L or 0) / 1e4, 1), new_water_ha=round((W or 0) / 1e4, 1),
               params={"nbr_thresh": args.nbr_thresh, "veg_nbr": args.veg_nbr, "cs_thresh": args.cs_thresh,
                       "min_elev": args.min_elev, "opt_pre_days": args.opt_pre_days, "opt_post_days": args.opt_post_days})
    rgb = b.visualize(bands=["B4", "B3", "B2"], min=0, max=3000)
    save_thumb(ee, rgb.blend(loss.selfMask().visualize(palette=["ff0000"])), geom, out / "optical_loss.png")
    save_thumb(ee, rgb.blend(water.selfMask().visualize(palette=["00aaff"])), geom, out / "optical_water.png")
    return res


def main():
    ap = argparse.ArgumentParser(description="雷達先行、光學補證的事件偵測（GEE）")
    ap.add_argument("--name", required=True)
    ap.add_argument("--center", required=True, help="lon,lat")
    ap.add_argument("--half-km", type=float, default=12.0)
    ap.add_argument("--event", required=True, help="事件日期 YYYY-MM-DD；其前為基準窗、其後為偵測窗")
    ap.add_argument("--post-days", type=int, default=30)
    ap.add_argument("--pre-days", type=int, default=60, help="S1 基準窗長")
    ap.add_argument("--opt-pre-days", type=int, default=90)
    ap.add_argument("--opt-post-days", type=int, default=30)
    ap.add_argument("--dark-db", type=float, default=-18.0)
    ap.add_argument("--delta-db", type=float, default=3.0)
    ap.add_argument("--max-slope", type=float, default=15.0)
    ap.add_argument("--nbr-thresh", type=float, default=0.2)
    ap.add_argument("--veg-nbr", type=float, default=0.1)
    ap.add_argument("--min-elev", type=float, default=300.0, help="光學損失／水體只計此高程（m）以上，排除平地農田")
    ap.add_argument("--cs-thresh", type=float, default=0.6)
    ap.add_argument("--min-valid", type=float, default=0.6)
    ap.add_argument("--hotspots", default=str(REPO / "data" / "ardswc_hotspots" / "top100_consolidated.json"))
    ap.add_argument("--project", default=None)
    args = ap.parse_args()

    ee = init_ee(args.project)
    lon, lat = (float(x) for x in args.center.split(","))
    geom = ee.Geometry.Point([lon, lat]).buffer(args.half_km * 1000).bounds()
    ev = dt.date.fromisoformat(args.event)
    out = OUT_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"AOI {args.center} ±{args.half_km} km，事件 {ev}", flush=True)
    sar = sar_stage(ee, geom, ev, args, out)
    opt = optical_stage(ee, geom, ev, args, out)
    res = {"name": args.name, "center": [lon, lat], "half_km": args.half_km, "event": str(ev),
           "note": "候選訊號，非判釋成果。雷達暗區≠水；面積受未校正閾值與有效像素影響。", "sar": sar, "optical": opt}
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("光學：", opt.get("status"), opt.get("reason", ""), {k: opt.get(k) for k in ("loss_ha", "new_water_ha", "valid_frac")})
    for k in ("loss_clusters", "water_clusters"):
        print(k, opt.get(k))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
