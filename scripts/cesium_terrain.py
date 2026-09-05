# -*- coding: utf-8 -*-
"""
Cesium Ion World Terrain 存取（供 change_detect_viewer 的可開關等高線圖使用）。

改編自 F:\\GitHub\\Infrared_Small_Target_Detection\\src\\stk_engine\\cesium_terrain.py
（該專案用途是把高程數值取回供幾何運算——視線遮蔽/陣地評估；本專案只需要拿高程網格
畫等高線疊圖，故拿掉對該專案 `stk_engine.terrain.Terrain` 類別的依賴，`fetch_terrain()`
改回傳原始 numpy 網格，其餘 tile 抓取/解碼/快取邏輯原樣保留，比照本專案一貫的
「自足複製、不依賴姊妹 repo」慣例（同 gmaps_tiles.py 的作法）。

資料流：
  1. Ion REST: /v1/assets/{id}/endpoint  -> 取得 tileset URL 與短期 token
  2. layer.json                          -> 取得 tiling scheme 與可用層級
  3. {z}/{x}/{y}.terrain                 -> quantized-mesh-1.0 二進位
  4. 解碼頂點 (zigzag delta) -> (lat, lon, height) 點雲
  5. 內插到規則網格

Token 處理（安全，同來源專案慣例）：依序嘗試 環境變數 CESIUM_ION_TOKEN ->
本專案根目錄 .cesium_ion_token 檔案 -> 呼叫端明確傳入。不接受寫死在程式碼中。

垂直基準：Cesium World Terrain 為橢球高 (HAE)，未在本模組內做 MSL 轉換——等高線圖僅供
視覺比對地形起伏，不用於精確高度換算，此差異（台灣一帶約 +17~20m）對等高線視覺效果
可忽略不計。

授權：Cesium World Terrain 免費層級的 attribution 含「Upgrade for commercial use」，
代表該層級不得商用，呼叫端有義務呈現 attributions（見 IonTerrainSource.attributions）。
"""
import gzip
import hashlib
import io
import json
import os
import struct
import urllib.error
import urllib.request

import numpy as np

ION_API = "https://api.cesium.com/v1/assets"
WORLD_TERRAIN_ASSET = 1
_TOKEN_FILE = ".cesium_ion_token"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _redact(msg, token):
    if not token:
        return str(msg)
    return str(msg).replace(token, "<TOKEN-REDACTED>")


def load_token(explicit=None, repo_root=None):
    if explicit:
        return explicit.strip()
    env = os.environ.get("CESIUM_ION_TOKEN")
    if env:
        return env.strip()
    root = repo_root or _REPO_ROOT
    fp = os.path.join(root, _TOKEN_FILE)
    if os.path.exists(fp):
        with io.open(fp, encoding="utf-8") as f:
            t = f.read().strip()
        if t:
            return t
    raise RuntimeError(
        "找不到 Cesium Ion token。請擇一設定：\n"
        "  1) 環境變數  CESIUM_ION_TOKEN=<你的 token>\n"
        f"  2) 專案根目錄建立 {_TOKEN_FILE} 檔\n"
        "  3) 呼叫時以 token= 參數傳入\n"
        "切勿把 token 寫進程式碼。")


class IonTerrainSource:
    """Cesium Ion 地形 tileset 的取用與快取。"""

    def __init__(self, token=None, asset_id=WORLD_TERRAIN_ASSET, cache_dir=None, timeout=45):
        self.token = load_token(token)
        self.asset_id = int(asset_id)
        self.timeout = timeout
        self.cache = cache_dir or os.path.join(_REPO_ROOT, "data", "cesium_cache")
        os.makedirs(self.cache, exist_ok=True)
        self._ep = None
        self._layer = None
        self.attributions = []

    def _get(self, url, headers=None):
        req = urllib.request.Request(url, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return r.status, dict(r.headers), raw
        except urllib.error.HTTPError as e:
            raise urllib.error.HTTPError(
                _redact(e.url, self.token), e.code,
                _redact(e.reason, self.token), e.headers, None) from None

    def endpoint(self):
        if self._ep is None:
            url = f"{ION_API}/{self.asset_id}/endpoint?access_token={self.token}"
            _, _, b = self._get(url)
            self._ep = json.loads(b)
            self.attributions = [a.get("html", "") for a in self._ep.get("attributions", [])]
        return self._ep

    def layer(self):
        if self._layer is None:
            ep = self.endpoint()
            _, _, b = self._get(ep["url"] + "layer.json",
                                 {"Authorization": f"Bearer {ep['accessToken']}"})
            self._layer = json.loads(b)
        return self._layer

    @property
    def max_level(self):
        return len(self.layer().get("available", [])) - 1

    def commercial_use_allowed(self):
        return not any("commercial use" in (a or "").lower() for a in self.attributions)

    @staticmethod
    def tile_xy(lat, lon, z):
        """Cesium 預設 TMS geographic：2^(z+1) x 2^z，原點 (-180,-90)。"""
        nx, ny = 2 ** (z + 1), 2 ** z
        x = int((lon + 180.0) / 360.0 * nx)
        y = int((lat + 90.0) / 180.0 * ny)
        return min(max(x, 0), nx - 1), min(max(y, 0), ny - 1)

    @staticmethod
    def tile_bounds(x, y, z):
        nx, ny = 2 ** (z + 1), 2 ** z
        w = -180.0 + 360.0 * x / nx
        e = -180.0 + 360.0 * (x + 1) / nx
        s = -90.0 + 180.0 * y / ny
        n = -90.0 + 180.0 * (y + 1) / ny
        return w, s, e, n

    def is_available(self, x, y, z):
        av = self.layer().get("available", [])
        if z < 0 or z >= len(av):
            return False
        for r in av[z]:
            if r["startX"] <= x <= r["endX"] and r["startY"] <= y <= r["endY"]:
                return True
        return False

    def best_level_for(self, lat, lon, max_level=None):
        top = self.max_level if max_level is None else min(max_level, self.max_level)
        for z in range(top, -1, -1):
            x, y = self.tile_xy(lat, lon, z)
            if self.is_available(x, y, z):
                return z
        return 0

    def fetch_tile(self, x, y, z):
        """取單一 tile 的原始位元組（含磁碟快取）。找不到回傳 None。"""
        key = hashlib.sha1(f"{self.asset_id}/{z}/{x}/{y}".encode()).hexdigest()[:16]
        fp = os.path.join(self.cache, f"{z}_{x}_{y}_{key}.terrain")
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                return f.read()
        miss = fp + ".404"
        if os.path.exists(miss):
            return None
        if not self.is_available(x, y, z):
            return None
        ep = self.endpoint()
        ver = self.layer().get("version", "1.2.0")
        url = f"{ep['url']}{z}/{x}/{y}.terrain?v={ver}"
        hdr = {"Authorization": f"Bearer {ep['accessToken']}",
               "Accept": "application/vnd.quantized-mesh,application/octet-stream;q=0.9"}
        try:
            st, _, raw = self._get(url, hdr)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(miss, "wb").close()
                return None
            raise
        if st != 200 or len(raw) < 92:
            open(miss, "wb").close()
            return None
        with open(fp, "wb") as f:
            f.write(raw)
        return raw


def _zigzag(u16):
    v = np.asarray(u16).astype(np.int32)
    return (v >> 1) ^ (-(v & 1))


def decode_quantized_mesh(raw, bounds):
    """解碼 quantized-mesh-1.0 頂點。bounds=(west,south,east,north) 度。回傳 (lat[],lon[],height[])。"""
    (_, _, _, min_h, max_h) = struct.unpack_from("<3d2f", raw, 0)
    off = 88
    vc, = struct.unpack_from("<I", raw, off)
    off += 4
    n = vc
    arr = np.frombuffer(raw, dtype="<u2", count=3 * n, offset=off)
    u = np.cumsum(_zigzag(arr[0:n].astype(np.uint16))) & 0x7FFF
    v = np.cumsum(_zigzag(arr[n:2 * n].astype(np.uint16))) & 0x7FFF
    h = np.cumsum(_zigzag(arr[2 * n:3 * n].astype(np.uint16))) & 0x7FFF

    w, s, e, nn = bounds
    lon = w + (u / 32767.0) * (e - w)
    lat = s + (v / 32767.0) * (nn - s)
    height = min_h + (h / 32767.0) * (max_h - min_h)
    return lat, lon, height


def fetch_terrain(lat0, lon0, span_km=4.5, res_m=30.0, token=None, level=None, source=None, verbose=False):
    """取得以 (lat0,lon0) 為中心、邊長 span_km 的真實地形，內插為 res_m 網格。

    回傳 (z_grid, lat_axis, lon_axis, info)——與原始 stk_engine 版本不同之處：不包 Terrain
    物件，直接給 numpy 網格＋座標軸（本模組唯一用途是畫等高線，不需要幾何查詢介面）。
    """
    src = source or IonTerrainSource(token=token)
    need = src.max_level
    for z in range(6, src.max_level + 1):
        if (180.0 / (2 ** z)) * 111320.0 / 48.0 <= res_m:
            need = z
            break
    have = src.best_level_for(lat0, lon0)
    level = int(min(level if level is not None else need, have))
    if verbose and have < need:
        print(f"[cesium] 該地點最高僅到 z={have}（需求 z={need}）；地形解析度將低於 {res_m:.0f}m")

    half = span_km * 1000.0 / 2.0
    dlat = half / 111320.0
    dlon = half / (111320.0 * np.cos(np.radians(lat0)))
    lat_min, lat_max = lat0 - dlat, lat0 + dlat
    lon_min, lon_max = lon0 - dlon, lon0 + dlon

    lats, lons, hs = [], [], []
    n_ok = n_miss = 0
    used_level = level
    for _attempt in range(4):
        x0, y0 = src.tile_xy(lat_min, lon_min, used_level)
        x1, y1 = src.tile_xy(lat_max, lon_max, used_level)
        lats, lons, hs = [], [], []
        n_ok = n_miss = 0
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                raw = src.fetch_tile(x, y, used_level)
                if raw is None:
                    n_miss += 1
                    continue
                b = src.tile_bounds(x, y, used_level)
                la, lo, hh = decode_quantized_mesh(raw, b)
                lats.append(la)
                lons.append(lo)
                hs.append(hh)
                n_ok += 1
        if n_ok:
            break
        used_level -= 1
        if used_level < 6:
            break

    if not n_ok:
        raise RuntimeError(
            f"Cesium 地形無可用 tile（中心 {lat0:.4f},{lon0:.4f}，層級 {level}..{used_level}）。")

    lat_pts = np.concatenate(lats)
    lon_pts = np.concatenate(lons)
    h_pts = np.concatenate(hs)

    n_lat = max(int(span_km * 1000 / res_m), 8)
    n_lon = n_lat
    gl = np.linspace(lat_min, lat_max, n_lat)
    go = np.linspace(lon_min, lon_max, n_lon)
    GL, GO = np.meshgrid(gl, go, indexing="ij")
    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        pts = np.column_stack([lat_pts, lon_pts])
        lin = LinearNDInterpolator(pts, h_pts)
        z = lin(GL, GO)
        bad = ~np.isfinite(z)
        if bad.any():
            nn_ = NearestNDInterpolator(pts, h_pts)
            z[bad] = nn_(GL[bad], GO[bad])
    except ImportError:
        z = np.empty_like(GL)
        for i in range(n_lat):
            for j in range(n_lon):
                k = np.argmin((lat_pts - gl[i]) ** 2 + (lon_pts - go[j]) ** 2)
                z[i, j] = h_pts[k]

    info = {"level": used_level, "tiles_ok": n_ok, "tiles_missing": n_miss,
            "vertices": int(len(h_pts)), "grid": (n_lat, n_lon),
            "h_range": (float(np.nanmin(z)), float(np.nanmax(z))),
            "attributions": src.attributions, "commercial_ok": src.commercial_use_allowed()}
    if verbose:
        print(f"[cesium] 層級 {used_level} | tile {n_ok} 成功/{n_miss} 缺 | 頂點 {len(h_pts)} -> "
              f"網格 {n_lat}x{n_lon} | 高程 {info['h_range'][0]:.0f}–{info['h_range'][1]:.0f} m")
    return z, gl, go, info
