# -*- coding: utf-8 -*-
"""
全臺灣 20 公尺網格數值地形模型（DTM）地形來源——取代 Cesium World Terrain。

資料：內政部地政司「2025年版全臺灣20公尺網格數值地形模型DTM資料」
（https://data.gov.tw/dataset/176927，政府資料開放授權條款第1版，可商用、需標示出處）。
EPSG:3826（TWD97 TM2 121 分帶）、20 m 網格、float32、nodata=-32767。由
scripts/prepare_dtm20.py 下載並轉成 data/dtm/tw_dtm20.tif（分塊壓縮 GeoTIFF）。

與 Cesium 版（cesium_terrain.IonTerrainSource）的差異：
- 本地檔案、免網路、免 token、可商用；不必逐 tile 呼叫外部 API，也沒有額度問題。
- 解析度固定 20 m（Cesium 免費層級實測 15–30 m 不等、部分地區更粗）。要求更細的
  網格（res_m < 20）只會是內插，沒有真實資訊，所以一律夾到 20 m，避免假裝比實際更精細。
- 海域/無資料處為 nodata，回傳 NaN（不用中位數硬填，避免沿海分析出現假平台/假窪地）。

介面：`Dtm20Source.fetch_grid(lat0, lon0, span_km, res_m)` 回傳與
`cesium_terrain.fetch_terrain` 相同的 (z_grid, lat_axis, lon_axis, info)——lat_axis 由南到北遞增；
`cesium_terrain.fetch_terrain(source=<Dtm20Source>)` 會自動轉呼叫本方法，等高線與流域分析
不必改動。
"""
import math
import os
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "data" / "dtm" / "tw_dtm20.tif"
NATIVE_RES_M = 20.0
ATTRIBUTION = "地形資料 © 內政部地政司 2025年版全臺灣20公尺網格數值地形模型（DTM，政府資料開放授權條款第1版）"

_transformer = None
_OV = None
_OV_FACTOR = 16


def _overview(path):
    """整島降採樣概覽（每格 16×16 原始格平均，約 320 m）；首次計算約 8 秒，之後存 .npy 直接載入。"""
    global _OV
    if _OV is not None:
        return _OV
    import rasterio
    from rasterio.enums import Resampling
    cache = Path(path).with_name(Path(path).stem + f"_ov{_OV_FACTOR}.npy")
    with rasterio.open(path) as ds:
        w, h = ds.width, ds.height
        if cache.exists():
            arr = np.load(cache)
        else:
            arr = ds.read(1, out_shape=(int(np.ceil(h / _OV_FACTOR)), int(np.ceil(w / _OV_FACTOR))),
                          resampling=Resampling.average).astype(np.float32)
            arr[~np.isfinite(arr) | (arr < -500)] = np.nan
            np.save(cache, arr)
    _OV = (arr, w / arr.shape[1], h / arr.shape[0])
    return _OV


def _to_twd97(lon, lat):
    global _transformer
    if _transformer is None:
        from pyproj import Transformer
        _transformer = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)
    return _transformer.transform(lon, lat)


def dtm_path():
    return Path(os.environ.get("TW_DTM20_PATH") or DEFAULT_PATH)


def is_available():
    return dtm_path().is_file()


class Dtm20Source:
    attributions = [ATTRIBUTION]

    def __init__(self, path=None):
        self.path = Path(path) if path else dtm_path()
        if not self.path.is_file():
            raise RuntimeError(
                f"找不到 DTM 檔案 {self.path}。請先執行 python scripts/prepare_dtm20.py 下載並整備。")

    def commercial_use_allowed(self):
        return True

    def endpoint(self):
        """比照 IonTerrainSource.endpoint()：/api/contours_status 用它做一次真實可用性檢查。"""
        import rasterio
        with rasterio.open(self.path) as ds:
            return {"path": str(self.path), "size": [ds.width, ds.height], "res_m": ds.res[0]}

    def fetch_grid(self, lat0, lon0, span_km=4.5, res_m=20.0):
        import rasterio
        from rasterio.windows import Window

        res_eff = max(float(res_m), NATIVE_RES_M)
        half = span_km * 1000.0 / 2.0
        dlat = half / 111320.0
        dlon = half / (111320.0 * math.cos(math.radians(lat0)))
        lat_min, lat_max = lat0 - dlat, lat0 + dlat
        lon_min, lon_max = lon0 - dlon, lon0 + dlon
        n = max(int(span_km * 1000 / res_eff), 8)
        gl = np.linspace(lat_min, lat_max, n)
        go = np.linspace(lon_min, lon_max, n)
        GL, GO = np.meshgrid(gl, go, indexing="ij")
        E, N = _to_twd97(GO, GL)

        with rasterio.open(self.path) as ds:
            left, top = ds.bounds.left, ds.bounds.top
            ds_w, ds_h = ds.width, ds.height
            nodata = ds.nodata if ds.nodata is not None else -32767.0
            # 像素中心座標系：整數 (row, col) 即像素中心
            col_f = (np.asarray(E) - left) / NATIVE_RES_M - 0.5
            row_f = (top - np.asarray(N)) / NATIVE_RES_M - 0.5
            c0, c1 = int(np.floor(col_f.min())) - 1, int(np.ceil(col_f.max())) + 2
            r0, r1 = int(np.floor(row_f.min())) - 1, int(np.ceil(row_f.max())) + 2
            c0c, c1c, r0c, r1c = max(c0, 0), min(c1, ds.width), max(r0, 0), min(r1, ds.height)
            if c1c <= c0c or r1c <= r0c:
                raise RuntimeError(
                    f"座標 ({lat0:.4f}, {lon0:.4f}) 不在全臺灣 20 m DTM 涵蓋範圍內（僅臺灣本島及部分離島）。")
            win = ds.read(1, window=Window(c0c, r0c, c1c - c0c, r1c - r0c)).astype(np.float32)

        valid = (win != nodata) & np.isfinite(win) & (win > -1000)
        if not valid.any():
            raise RuntimeError(f"座標 ({lat0:.4f}, {lon0:.4f}) 周邊無 DTM 有效資料（海域或無資料區）。")
        data = np.where(valid, win, 0.0).astype(np.float32)
        weight = valid.astype(np.float32)

        # 粗網格需求（如低 zoom 等高線）先做面積平均，避免逐點取樣造成混疊
        factor = res_eff / NATIVE_RES_M
        sx = sy = 1.0
        if factor >= 1.5:
            h, w = data.shape
            nw, nh = max(1, int(round(w / factor))), max(1, int(round(h / factor)))
            data = cv2.resize(data, (nw, nh), interpolation=cv2.INTER_AREA)
            weight = cv2.resize(weight, (nw, nh), interpolation=cv2.INTER_AREA)
            sx, sy = w / nw, h / nh
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = np.where(weight > 0.5, data / np.maximum(weight, 1e-6), np.nan).astype(np.float32)

        # 視窗內座標 → 縮小後陣列座標（像素中心對齊）
        cc = ((col_f - c0c) + 0.5) / sx - 0.5
        rr = ((row_f - r0c) + 0.5) / sy - 0.5
        from scipy.ndimage import map_coordinates
        # NaN 需要保留：先用最近有效值補洞做插值，再把原本無資料的格點遮回 NaN
        nan_mask = ~np.isfinite(grid)
        if nan_mask.any():
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
            filled = grid[tuple(idx)]
        else:
            filled = grid
        z = map_coordinates(filled, [rr, cc], order=1, mode="nearest").astype(np.float32)
        bad = map_coordinates(nan_mask.astype(np.float32), [rr, cc], order=1, mode="nearest") > 0.5
        z[bad] = np.nan
        # 完全超出資料範圍的點（視窗被邊界截斷）也視為無資料
        outside = (col_f < -0.5) | (row_f < -0.5) | (col_f > ds_w - 0.5) | (row_f > ds_h - 0.5)
        z[outside] = np.nan

        finite = np.isfinite(z)
        info = {"level": "dtm20", "tiles_ok": 1, "tiles_missing": 0, "vertices": int(valid.sum()),
                "grid": (n, n),
                "h_range": (float(np.nanmin(z)), float(np.nanmax(z))) if finite.any() else (float("nan"),) * 2,
                "res_m_effective": res_eff, "source": "tw-dtm20-2025",
                "attributions": self.attributions, "commercial_ok": True}
        return z, gl, go, info

    def sample_points(self, lon, lat, max_px=2048):
        """在任意 (lon, lat) 點取高程（雙線性）。供 3D Cesium 地形使用（/api/dtm_heights）。
        範圍過大時對視窗做面積平均降採樣（每邊最多 max_px），避免粗層級整張讀入。
        無資料（海域/範圍外）回 0.0——Cesium 地形不能有 NaN，海面即 0 m。"""
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window
        from scipy.ndimage import map_coordinates

        lon = np.asarray(lon, dtype=np.float64); lat = np.asarray(lat, dtype=np.float64)
        E, N = _to_twd97(lon, lat)
        out = np.zeros(lon.shape, np.float32)
        with rasterio.open(self.path) as ds:
            col_f = (np.asarray(E) - ds.bounds.left) / NATIVE_RES_M - 0.5
            row_f = (ds.bounds.top - np.asarray(N)) / NATIVE_RES_M - 0.5
            c0, c1 = max(int(np.floor(col_f.min())) - 1, 0), min(int(np.ceil(col_f.max())) + 2, ds.width)
            r0, r1 = max(int(np.floor(row_f.min())) - 1, 0), min(int(np.ceil(row_f.max())) + 2, ds.height)
            if c1 <= c0 or r1 <= r0:
                return out
            wpx, hpx = c1 - c0, r1 - r0
            sc = max(1.0, max(wpx, hpx) / float(max_px))
            if sc >= 2:
                # 粗層級（視窗每邊 > 4096 px（約 82 km））：用整島概覽（記憶體＋磁碟快取），避免每次解壓整張 224 MB
                ov, fx, fy = _overview(self.path)
                oc0, oc1 = int(c0 // fx), int(np.ceil(c1 / fx)) + 1
                or0, or1 = int(r0 // fy), int(np.ceil(r1 / fy)) + 1
                win = ov[or0:or1, oc0:oc1].astype(np.float32).copy()
                c0, r0, sx, sy = oc0 * fx, or0 * fy, fx, fy
            else:
                shape = (max(1, int(np.ceil(hpx / sc))), max(1, int(np.ceil(wpx / sc))))
                win = ds.read(1, window=Window(c0, r0, wpx, hpx), out_shape=shape,
                              resampling=Resampling.average if sc > 1 else Resampling.nearest).astype(np.float32)
                sx, sy = wpx / shape[1], hpx / shape[0]
            ds_w, ds_h = ds.width, ds.height
        bad = ~np.isfinite(win) | (win < -500)
        win[bad] = np.nan
        if bad.all():
            return out
        if bad.any():
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            filled = win[tuple(idx)]
        else:
            filled = win
        cc = ((col_f - c0) + 0.5) / sx - 0.5
        rr = ((row_f - r0) + 0.5) / sy - 0.5
        z = map_coordinates(filled, [rr, cc], order=1, mode="nearest").astype(np.float32)
        nanm = map_coordinates(bad.astype(np.float32), [rr, cc], order=1, mode="nearest") > 0.5
        z[nanm] = 0.0
        outside = (col_f < -0.5) | (row_f < -0.5) | (col_f > ds_w - 0.5) | (row_f > ds_h - 0.5)
        z[outside] = 0.0
        return z
