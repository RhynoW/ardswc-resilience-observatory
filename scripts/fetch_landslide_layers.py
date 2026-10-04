"""下載歷年「全臺崩塌地圖層」（農業資料開放平臺）並裁成 POC 範圍，成功後刪除原始壓縮檔。

來源（政府資料開放授權條款第 1 版）：data.gov.tw 110–113 年度全臺崩塌地圖層，實際檔案在 data.moa.gov.tw。
伺服器回應慢、檔案大（約 300–500 MB／年），所以用 HTTP Range 斷點續傳，逾時自動重試。
裁切結果：data/biggis_interp/poc/landslide_<西元年>.gpkg（只留 POC 範圍＋緩衝內的多邊形）。

用法：python scripts/fetch_landslide_layers.py [年度 ...]   # 預設 113 112 111 110
"""
import re
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pyogrio
import requests
from shapely.geometry import box

REPO = Path(__file__).resolve().parent.parent
SHP = REPO / "data" / "biggis_interp" / "shp"
OUT = REPO / "data" / "biggis_interp" / "poc"
BASE = "https://data.moa.gov.tw/GetOpenDataFile.aspx?"
LAYERS = {  # 民國年 → (查詢參數, 西元年)
    113: ("id=J69&FileType=SHP&RID=65549", 2024),
    112: ("id=J05&FileType=SHP&RID=31194", 2023),
    111: ("id=I83&FileType=SHP&RID=26665", 2022),
    110: ("id=I67&FileType=SHP&RID=25026", 2021),
}
# POC：高雄（桃源、甲仙、六龜、那瑪夏）一帶 3×3 個 0.1° 格，外加 0.05° 緩衝
BBOX = (120.55, 22.95, 120.95, 23.35)  # lon0, lat0, lon1, lat1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def resolve(query):
    for _ in range(20):
        try:
            r = requests.get(BASE + query, allow_redirects=False, timeout=60)
            loc = r.headers.get("Location")
            if loc:
                return "https://data.moa.gov.tw" + loc if loc.startswith("/") else loc
        except requests.RequestException as e:
            log("resolve retry:", type(e).__name__)
        time.sleep(5)
    raise RuntimeError("無法取得下載網址")


def download(url, dest):
    total = None
    fails = 0
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have >= total:
            return
        try:
            with requests.get(url, headers={"Range": f"bytes={have}-"}, stream=True, timeout=(30, 60)) as r:
                if r.status_code == 416:
                    return
                if r.status_code == 200 and have:  # 伺服器不支援續傳：從頭來
                    have = 0
                    dest.unlink()
                cr = r.headers.get("Content-Range")
                if cr:
                    total = int(cr.split("/")[-1])
                elif r.status_code == 200:
                    total = int(r.headers.get("Content-Length", 0)) or None
                r.raise_for_status()
                with open(dest, "ab") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                fails = 0
        except requests.RequestException as e:
            fails += 1
            log(f"download retry {fails}: {type(e).__name__}; have={have/1e6:.0f} MB / {0 if not total else total/1e6:.0f} MB")
            if fails > 200:
                raise
            time.sleep(5)
        else:
            if total is None or dest.stat().st_size >= total:
                return


def crop(zpath, year):
    with zipfile.ZipFile(zpath) as z:
        shps = [n for n in z.namelist() if n.lower().endswith(".shp")]
    if not shps:
        raise RuntimeError("壓縮檔內沒有 .shp")
    src = f"zip://{zpath}!{shps[0]}"
    info = pyogrio.read_info(src)
    crs = info["crs"]
    log(f"{year}: {shps[0]} 筆數={info['features']} CRS={crs}")
    b = gpd.GeoSeries([box(*BBOX)], crs=4326).to_crs(crs).total_bounds
    g = pyogrio.read_dataframe(src, bbox=tuple(b))
    g["year"] = year
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"landslide_{year}.gpkg"
    g.to_file(out, driver="GPKG")
    log(f"{year}: 裁切 {len(g)} 個多邊形 → {out.name}；欄位 {[c for c in g.columns if c != 'geometry']}")
    return len(g)


def main():
    SHP.mkdir(parents=True, exist_ok=True)
    roc = [int(a) for a in sys.argv[1:]] or [113, 112, 111, 110]
    for y in roc:
        query, year = LAYERS[y]
        if (OUT / f"landslide_{year}.gpkg").exists():
            log(f"{y} 年度已有裁切結果，略過")
            continue
        zpath = SHP / f"landslide_{y}.zip"
        log(f"{y} 年度：取得下載網址")
        url = resolve(query)
        log(f"{y} 年度：下載 {url.split('/')[-1]}")
        download(url, zpath)
        log(f"{y} 年度：下載完成 {zpath.stat().st_size/1e6:.0f} MB")
        try:
            crop(zpath, year)
        except Exception as e:  # 裁切失敗就保留壓縮檔，方便除錯
            log(f"{y} 年度：裁切失敗，保留壓縮檔：{e!r}")
            continue
        zpath.unlink()
        log(f"{y} 年度：已刪除原始壓縮檔")
    log("全部完成")


if __name__ == "__main__":
    main()
