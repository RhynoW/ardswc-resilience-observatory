"""各地層的崩塌密度（粗尺度）：公開全臺地層面 × 高雄 POC ≥15° 有效坡面 × 2021–2024 年度崩塌圖層。
地層：geodac.ncku.edu.tw/SWCB_LLGIS/地質圖.kml（BigGIS 引用；1,576 個地層面，欄位 NAME_C／ABBREV／TIME_C，
     小比例尺區域地質圖，只有地層，沒有斷層、位態、崩積區；POC 內地層面中位數約 160 ha，遠大於坡面單元）。
     先存到 data/biggis_interp/poc/geology/地質圖.kml（gitignore）。
崩塌：fetch_landslide_layers.py 的 110–113 年度全臺崩塌地圖層（2021–2024），四年聯集。
分母：SUv2 有效坡面（slope_units_SUv2.gpkg 的 su_effective，坡度 ≥15°），所以平緩的河階、沖積層只算到其中的陡坡部分。
每個地層：有效坡面面積、崩塌面積、崩塌密度（崩塌面積 ÷ 有效坡面面積）、頻率比 FR（該地層密度 ÷ 全區密度，>1 表示比平均容易崩）、
     Sentinel-2 NDRE 持續下降坡面占比（坡面依有效坡面代表點歸到地層）。
有效坡面 <2 km² 的地層樣本太小，標註但不解讀。這是描述性統計，不是敏感度模型；斷層距離、順向坡要五萬分之一向量才能做。
輸出：data/biggis_interp/poc/geology/geology_landslide_density.json，並印出表格。
用法：python scripts/geology_landslide_density.py
"""
import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
KML = POC / "geology" / "地質圖.kml"
SU = POC / "su_export" / "slope_units_SUv2.gpkg"
YEARS = (2021, 2022, 2023, 2024)
MIN_KM2 = 2.0


def ring(txt):
    return [tuple(map(float, p.split(",")[:2])) for p in txt.split()]


def read_kml():
    t = KML.read_text(encoding="utf-8")
    rows = []
    for pm in t.split("<Placemark")[1:]:
        f = dict(re.findall(r"<td>([A-Z_0-9]+)</td>\s*<td>([^<]*)</td>", pm))
        polys = []
        for pg in re.findall(r"<Polygon>(.*?)</Polygon>", pm, re.S):
            outer = re.search(r"<outerBoundaryIs>.*?<coordinates>(.*?)</coordinates>", pg, re.S).group(1)
            inner = re.findall(r"<innerBoundaryIs>.*?<coordinates>(.*?)</coordinates>", pg, re.S)
            polys.append(Polygon(ring(outer), [ring(i) for i in inner]).buffer(0))
        # KML 的「頭嵙山層」頭字遺失成 ?，補回
        rows.append({"abbrev": f.get("ABBREV", ""), "name": f.get("NAME_C", "").replace("?嵙", "頭嵙"), "age": f.get("TIME_C", ""), "geometry": unary_union(polys)})
    return gpd.GeoDataFrame(rows, crs=4326).to_crs(3826)


def main():
    eff = gpd.read_file(SU, layer="su_effective")[["su_id", "geometry"]]
    poc = eff.union_all()
    geo = read_kml()
    geo = geo[geo.intersects(poc.envelope)]
    geo = geo.dissolve(by=["abbrev", "name", "age"], as_index=False)
    geo["geometry"] = geo.intersection(poc)
    geo = geo[~geo.is_empty]
    geo["eff_km2"] = geo.area / 1e6
    # 2021 圖層的 CRS 寫成自訂 TM2_TWD97（GRS80、中央經線 121°），與 EPSG:3826 相同，直接覆寫
    ls = pd.concat([gpd.read_file(POC / f"landslide_{y}.gpkg")[["geometry"]].set_crs(3826, allow_override=True) for y in YEARS])
    ls_u = ls.union_all().intersection(poc)
    geo["ls_ha"] = geo.intersection(ls_u).area / 1e4
    nd = json.load(open(POC / "s2ts" / "su_ndre.json", encoding="utf-8"))["su"]
    pts = gpd.GeoDataFrame(eff[["su_id"]], geometry=eff.representative_point(), crs=3826)
    pts = pts[pts["su_id"].isin(nd) & pts["su_id"].map(lambda s: nd.get(s, {}).get("base") is not None)]
    pts["drop"] = pts["su_id"].map(lambda s: bool(nd[s].get("drop")))
    j = gpd.sjoin(pts, geo[["abbrev", "geometry"]], predicate="within")
    cnt = j.groupby("abbrev")["drop"].agg(["size", "sum"])
    tot_eff, tot_ls = geo["eff_km2"].sum(), geo["ls_ha"].sum()
    dens_all = tot_ls / (tot_eff * 100)
    rows = []
    for _, r in geo.sort_values("eff_km2", ascending=False).iterrows():
        d = r["ls_ha"] / (r["eff_km2"] * 100)
        n_su, n_drop = (int(cnt.loc[r["abbrev"], "size"]), int(cnt.loc[r["abbrev"], "sum"])) if r["abbrev"] in cnt.index else (0, 0)
        rows.append({"abbrev": r["abbrev"], "name": r["name"], "age": r["age"], "eff_km2": round(r["eff_km2"], 1), "ls_ha": round(r["ls_ha"], 1),
                     "density": round(d, 4), "fr": round(d / dens_all, 2), "n_su": n_su, "n_drop": n_drop,
                     "drop_rate": round(n_drop / n_su, 3) if n_su else None, "small": bool(r["eff_km2"] < MIN_KM2)})
    n_su_all = int(cnt["size"].sum())
    out = {"meta": {"geology": "geodac.ncku.edu.tw/SWCB_LLGIS/地質圖.kml（小比例尺地層面）", "landslide": "全臺崩塌地圖層 %d–%d 聯集" % (YEARS[0], YEARS[-1]),
                    "denominator": "SUv2 有效坡面（≥15°）", "min_km2": MIN_KM2, "eff_km2": round(tot_eff, 1), "ls_ha": round(tot_ls, 1), "density": round(dens_all, 4),
                    "n_su": n_su_all, "n_drop": int(cnt["sum"].sum()), "drop_rate": round(cnt["sum"].sum() / n_su_all, 3)}, "rows": rows}
    (POC / "geology" / "geology_landslide_density.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"全區：有效坡面 {tot_eff:.1f} km²，崩塌 {tot_ls:.0f} ha，密度 {dens_all:.2%}，NDRE 下降 {out['meta']['n_drop']}/{n_su_all}")
    for r in rows:
        print(f"{r['abbrev']:5s} {r['name'][:16]:16s} {r['age'][:8]:8s} {r['eff_km2']:7.1f} km² {r['ls_ha']:7.1f} ha {r['density']:6.2%} FR {r['fr']:4.2f}  NDRE {r['n_drop']}/{r['n_su']}{'  (小樣本)' if r['small'] else ''}")


if __name__ == "__main__":
    main()
