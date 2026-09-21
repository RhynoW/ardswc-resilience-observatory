# -*- coding: utf-8 -*-
"""
下載並整備「2025年版全臺灣20公尺網格數值地形模型 DTM」（內政部地政司，政府資料開放授權條款第1版；
https://data.gov.tw/dataset/176927 ）。

資料集本身只是一份 5.5 KB 的下載清單 CSV，指向 TGOS 上各縣市分幅 zip 與一份整島「不分幅」zip。
本腳本取整島版（269 MB，免登入），內含 float32 GeoTIFF（EPSG:3826 TWD97 TM2、20 m、
10035×18852、nodata=-32767、未壓縮 757 MB、逐列 strip 排列——視窗讀取時每次都要讀滿整列）。
為了讓小視窗讀取又快又省，轉成 512×512 分塊 + DEFLATE + 浮點預測器的 GeoTIFF（無損），
輸出 data/dtm/tw_dtm20.tif，供 scripts/dtm20.py 使用。

TGOS 伺服器連線中途會被截斷（實測 269 MB 只收到 60 MB 就結束、且回 200），所以下載一律
用 Range 續傳，直到大小與 Content-Length 相符並通過 zip 校驗。

用法：python scripts/prepare_dtm20.py [--keep-source]
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "dtm"
OUT_TIF = OUT_DIR / "tw_dtm20.tif"
ZIP_URL = ("https://www.tgos.tw:443/MDE/VirtualDir_TC/Product/528530be-0710-431e-954e-2f2f5e98b0c5/"
           "%E4%B8%8D%E5%88%86%E5%B9%85_%E5%85%A8%E5%8F%B020MDEM(2025).zip")
SRC_NAME = "DEM_tawiwan_V2025.tif"   # zip 內檔名（原檔拼字即如此）
UA = "ardswc-resilience-observatory/1.0 (dtm20-prepare)"


def _total_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download(url, dest, max_tries=15):
    total = _total_size(url)
    for attempt in range(1, max_tries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= total:
            break
        print(f"[dtm20] 下載 {have / 1e6:.0f}/{total / 1e6:.0f} MB（第 {attempt} 次）", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f:
                if have and r.status != 206:      # 伺服器不支援續傳 → 從頭來
                    f.seek(0)
                    f.truncate()
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as e:
            print(f"[dtm20] 連線中斷：{e}，續傳", flush=True)
    if dest.stat().st_size != total:
        raise RuntimeError(f"下載未完成：{dest.stat().st_size} / {total} bytes")
    with zipfile.ZipFile(dest) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"zip 校驗失敗：{bad}")


def convert(src_tif, dst_tif):
    import rasterio
    from pyproj import CRS
    # 用 pyproj 產生 WKT 再交給 rasterio：本機環境變數 PROJ_DATA 可能指向壞掉的 GDAL projlib，
    # 讓 rasterio 直接查 "EPSG:3826" 會失敗（Cannot find proj.db），pyproj 自帶資料庫不受影響。
    wkt = CRS.from_epsg(3826).to_wkt()
    with rasterio.open(src_tif) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                       predictor=3, zlevel=9, crs=rasterio.crs.CRS.from_wkt(wkt), BIGTIFF="IF_SAFER")
        tmp = dst_tif.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            band = 512
            for r0 in range(0, src.height, band):
                win = rasterio.windows.Window(0, r0, src.width, min(band, src.height - r0))
                dst.write(src.read(1, window=win), 1, window=win)
        tmp.replace(dst_tif)


def main():
    ap = argparse.ArgumentParser(description="下載並整備全臺灣 20 m DTM")
    ap.add_argument("--keep-source", action="store_true", help="保留下載的 zip 與解壓出的原始 tif")
    args = ap.parse_args()
    if OUT_TIF.exists():
        print(f"[dtm20] 已存在 {OUT_TIF}，略過（刪除該檔可重新整備）")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_fp = OUT_DIR / "tw20m.zip"
    download(ZIP_URL, zip_fp)
    src_tif = OUT_DIR / SRC_NAME
    if not src_tif.exists():
        print("[dtm20] 解壓…", flush=True)
        with zipfile.ZipFile(zip_fp) as z:
            z.extract(SRC_NAME, OUT_DIR)
    print("[dtm20] 轉成分塊壓縮 GeoTIFF…", flush=True)
    convert(src_tif, OUT_TIF)
    print(f"[dtm20] 完成：{OUT_TIF}（{OUT_TIF.stat().st_size / 1e6:.0f} MB）")
    if not args.keep_source:
        for p in (zip_fp, src_tif, OUT_DIR / "DEM_tawiwan_V2025.tfw", OUT_DIR / "manifest.csv"):
            p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
