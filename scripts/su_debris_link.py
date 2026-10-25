"""坡面單元與土石流潛勢溪流的連結（崩塌 → 土砂 → 溪流 → 下游的第一段串接）：只作參考資訊，不改候選層級。
資料：農業部農村發展及水土保持署 BigGIS 公開向量圖磚（gis.ardswc.gov.tw/api/ardswc/vectortiles/shp/{debris_flow,debris_watershed,debris_affect}），
     由 Rust_WMTS_Server 專案的 scripts/make_ardswc_geojson.py 抓 z12 圖磚合併成 GeoJSON（座標精度約 10 m、已簡化；2026-10-02 快照），
     複製到 data/biggis_interp/poc/debris/（gitignore）。屬性只有 popup_html，這裡解析回欄位。
每個 SUv2 原始坡面（su_original，EPSG:3826）：
  集水區：坡面落在哪條潛勢溪流的集水區內（重疊 ≥10% 坡面面積者，依重疊比例排序；坡面的土砂會進入這條溪流）
  影響範圍：坡面外推 100 m 是否碰到潛勢溪流的影響範圍（範圍內的保全住戶數）
  最近溪流：最近一條潛勢溪流的編號、名稱、風險與距離（m，自坡面邊界起算）
輸出：data/biggis_interp/poc/su_debris.json（su_id → 指標；含資料來源與快照日期）
用法：python scripts/su_debris_link.py
"""
import json
import re
import warnings
from pathlib import Path

import geopandas as gpd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
SRC = POC / "debris"
SU = POC / "su_export" / "slope_units_SUv2.gpkg"
BUF = 100.0
MIN_SHARE = 0.10
SNAPSHOT = "2026-10-02"


def load(name):
    g = gpd.read_file(SRC / f"{name}.geojson")
    rows = [dict(re.findall(r"([^<>：]+)：([^<]*)", h)) for h in g["popup_html"]]
    g = gpd.GeoDataFrame(rows, geometry=g.geometry.values, crs=4326).to_crs(3826)
    return g


def households(v):
    """保全住戶數；欄位缺或無法解析時回 None（不當成 0）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else int(f)


def main():
    su = gpd.read_file(SU, layer="su_original")[["su_id", "geometry"]]
    box = su.total_bounds
    clip = lambda g: g.cx[box[0] - 5000:box[2] + 5000, box[1] - 5000:box[3] + 5000].reset_index(drop=True)
    ws, af, fl = clip(load("debris_watershed")), clip(load("debris_affect")), clip(load("debris_flow"))
    su["a"] = su.area
    x = gpd.overlay(su[["su_id", "a", "geometry"]], ws[["編號", "geometry"]], how="intersection")
    x["share"] = x.area / x["a"]
    x = x[x["share"] >= MIN_SHARE].sort_values("share", ascending=False)
    in_ws = {sid: [[r["編號"], round(float(r["share"]), 2)] for _, r in g.iterrows()] for sid, g in x.groupby("su_id")}
    buf = gpd.GeoDataFrame(su[["su_id"]], geometry=su.buffer(BUF), crs=3826)
    j = gpd.sjoin(buf, af[["編號", "保全住戶", "geometry"]], predicate="intersects")
    hit_af = {sid: [[r["編號"], households(r["保全住戶"])] for _, r in g.drop_duplicates("編號").iterrows()] for sid, g in j.groupby("su_id")}
    near = gpd.sjoin_nearest(su[["su_id", "geometry"]], fl[["編號", "溪流", "風險", "geometry"]], how="left", distance_col="d")
    near = near.sort_values("d").drop_duplicates("su_id").set_index("su_id")
    out = {}
    for sid in su["su_id"]:
        n = near.loc[sid]
        out[sid] = {"ws": in_ws.get(sid, []), "affect": hit_af.get(sid, []),
                    "near": None if n.isna()["編號"] else {"id": n["編號"], "name": n["溪流"], "risk": n["風險"], "m": round(float(n["d"]))}}
    meta = {"source": "農業部農村發展及水土保持署 BigGIS 公開向量圖磚（土石流潛勢溪流、集水區、影響範圍）", "snapshot": SNAPSHOT,
            "precision": "約 10 m（z12 向量圖磚合併、已簡化）", "buffer_m": BUF, "min_share": MIN_SHARE, "n_su": len(out),
            "n_in_ws": sum(1 for v in out.values() if v["ws"]), "n_affect": sum(1 for v in out.values() if v["affect"]),
            "counts": {"streams": len(fl), "watersheds": len(ws), "affect": len(af)}}
    (POC / "su_debris.json").write_text(json.dumps({"meta": meta, "su": out}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
