"""坡面單元的道路與聚落暴露（OpenStreetMap）：回答「這個坡面如果崩了，會碰到什麼」，不改候選層級。
資料：Geofabrik 臺灣 OSM 擷取（taiwan-latest.osm.pbf，ODbL；頁面須標示「© OpenStreetMap contributors」），
     先切成 POC 範圍 data/biggis_interp/poc/osm/poc_osm.gpkg（roads＝highway 線、buildings＝building 面、places＝place 點）。
每個 SUv2 原始坡面（su_original，EPSG:3826）計算：
  坡面內、坡面外推 100 m 範圍內的道路長度（m，依道路等級分主要／一般／產業道路與小徑）
  坡面內、100 m 範圍內的建物數（建物面中心點落在範圍內）
  最近的聚落點（place=town/village/hamlet/isolated_dwelling/neighbourhood）名稱與距離（m，自坡面邊界起算）
OSM 由志工繪製，偏遠山區的建物與產業道路常缺漏；數字是「至少有這些」，不是完整清冊。
輸出：data/biggis_interp/poc/su_osm_exposure.json（su_id → 指標；含資料日期）
用法：python scripts/su_osm_exposure.py [--extract PBF]   # 有 --extract 時先從 PBF 重新切 POC 範圍
"""
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import geopandas as gpd
import pyogrio

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
OSM = POC / "osm" / "poc_osm.gpkg"
SU = POC / "su_export" / "slope_units_SUv2.gpkg"
BBOX = (120.55, 22.95, 120.95, 23.35)  # lon0, lat0, lon1, lat1（與 fetch_landslide_layers 相同）
BUF = 100.0
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
MINOR = {"unclassified", "residential", "living_street", "service", "road"}
TRACK = {"track", "path", "footway", "steps", "bridleway", "cycleway"}
PLACES = {"town", "village", "hamlet", "isolated_dwelling", "neighbourhood"}


def extract(pbf):
    OSM.parent.mkdir(parents=True, exist_ok=True)
    for lay, where, name in (("lines", "highway IS NOT NULL", "roads"), ("multipolygons", "building IS NOT NULL", "buildings"), ("points", "place IS NOT NULL", "places")):
        df = pyogrio.read_dataframe(pbf, layer=lay, bbox=BBOX, where=where)
        pyogrio.write_dataframe(df, OSM, layer=name)
        print(name, len(df))


def road_len(roads, zone):
    """zone: GeoDataFrame[su_id, geometry] → {su_id: {cls: m}}。"""
    x = gpd.overlay(roads[["cls", "geometry"]], zone, how="intersection", keep_geom_type=True)
    x["m"] = x.length
    out = {}
    for (sid, c), m in x.groupby(["su_id", "cls"])["m"].sum().items():
        out.setdefault(sid, {})[c] = round(float(m))
    return out


def count_in(pts, zone):
    j = gpd.sjoin(pts[["geometry"]], zone, predicate="within")
    return j.groupby("su_id").size().to_dict()


def main():
    if "--extract" in sys.argv:
        extract(sys.argv[sys.argv.index("--extract") + 1])
    su = gpd.read_file(SU, layer="su_original")[["su_id", "geometry"]]
    roads = gpd.read_file(OSM, layer="roads").to_crs(3826)
    roads["cls"] = roads["highway"].map(lambda h: "major" if h in MAJOR else "minor" if h in MINOR else "track" if h in TRACK else None)
    roads = roads[roads["cls"].notna()]
    bld = gpd.read_file(OSM, layer="buildings").to_crs(3826)
    bld = gpd.GeoDataFrame(geometry=bld.geometry.representative_point(), crs=3826)
    pl = gpd.read_file(OSM, layer="places").to_crs(3826)
    pl = pl[pl["place"].isin(PLACES)][["name", "place", "geometry"]].reset_index(drop=True)
    buf = su.copy()
    buf["geometry"] = su.buffer(BUF)
    r_in, r_buf = road_len(roads, su), road_len(roads, buf)
    b_in, b_buf = count_in(bld, su), count_in(bld, buf)
    near = gpd.sjoin_nearest(su, pl, how="left", distance_col="d")
    near = near.sort_values("d").drop_duplicates("su_id").set_index("su_id")
    out = {}
    for sid in su["su_id"]:
        n = near.loc[sid]
        out[sid] = {"road_in": r_in.get(sid, {}), "road_100": r_buf.get(sid, {}), "bld_in": int(b_in.get(sid, 0)), "bld_100": int(b_buf.get(sid, 0)),
                    "place": None if n.isna()["name"] else {"name": n["name"], "type": n["place"], "m": round(float(n["d"]))}}
    hit = sum(1 for v in out.values() if v["road_100"] or v["bld_100"])
    meta = {"source": "OpenStreetMap（Geofabrik taiwan-latest.osm.pbf）", "license": "ODbL；© OpenStreetMap contributors", "buffer_m": BUF,
            "extract_date": date.fromtimestamp(OSM.stat().st_mtime).isoformat(), "n_su": len(out), "n_road_or_bld_100": hit,
            "counts": {"roads": len(roads), "buildings": len(bld), "places": len(pl)}}
    (POC / "su_osm_exposure.json").write_text(json.dumps({"meta": meta, "su": out}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
