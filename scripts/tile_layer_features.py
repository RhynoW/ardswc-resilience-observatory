# -*- coding: utf-8 -*-
"""
地調所「山崩地質資訊雲端服務平臺」圖磚 → 樣本點的圖層欄位（取點位像素的 alpha：有圖＝在圖層內）。
圖磚網址與可用圖層由 Rust_WMTS_Server 專案實測（static/layers.js）：
  https://landslide.geologycloud.tw/jlwmts/jetlink/{layer}/GoogleMapsCompatible/{z}/{x}/{y}.png（x 在前）
  swcb_Debris 土石流潛勢溪流（z≤13）、SensitiveArea 地質敏感區（全部類別，不只山崩地滑）、Dislope 順向坡判釋目錄（z≤13）、
  HistoryLS 歷史山崩目錄、Slp20 坡度圖。
限制：以 z13 圖磚取點（約 19 m/像素）；圖磚是「繪製後的點陣圖」，線狀圖層（潛勢溪流）在 z13 很細、易漏，不當主要特徵；
  HistoryLS 是過去災害的目錄，作預測特徵會有結果洩漏（災點本身可能就在目錄中），預設只輸出、解讀時要隔離；
  地質敏感區含非山崩類別（如斷層、地下水補注）。
用法：python scripts/tile_layer_features.py --samples data/warning_trainer/samples.csv --out data/warning_trainer/tiles.csv
"""
import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

import cv2
import numpy as np

BASE = "https://landslide.geologycloud.tw/jlwmts/jetlink/{layer}/GoogleMapsCompatible/{z}/{x}/{y}.png"
LAYERS = {"geosens_in": "SensitiveArea", "dislope_in": "Dislope", "histls_in": "HistoryLS"}
Z = 13
_cache = {}


def tile(layer, x, y):
    k = (layer, x, y)
    if k not in _cache:
        req = urllib.request.Request(BASE.format(layer=layer, z=Z, x=x, y=y), headers={"User-Agent": "Mozilla/5.0"})
        try:
            b = urllib.request.urlopen(req, timeout=30).read()
            _cache[k] = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:  # 404＝該磚無資料
            _cache[k] = None
    return _cache[k]


def sample(layer, lon, lat):
    n = 2 ** Z
    fx = (lon + 180) / 360 * n
    fy = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    x, y = int(fx), int(fy)
    t = tile(layer, x, y)
    if t is None:
        return 0
    px, py = min(int((fx - x) * t.shape[1]), t.shape[1] - 1), min(int((fy - y) * t.shape[0]), t.shape[0] - 1)
    return int(t[py, px, 3] > 0) if t.ndim == 3 and t.shape[2] == 4 else int(t[py, px].any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.samples, encoding="utf-8")))
    out = []
    with ThreadPoolExecutor(8) as ex:
        for r, vals in zip(rows, ex.map(lambda r: [sample(l, float(r["lon"]), float(r["lat"])) for l in LAYERS.values()], rows)):
            out.append({"id": r["id"], **dict(zip(LAYERS, vals))})
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", *LAYERS])
        w.writeheader()
        w.writerows(out)
    tot = len(out)
    print({k: f"{sum(o[k] for o in out)}/{tot}" for k in LAYERS}, "→", a.out)


if __name__ == "__main__":
    main()
