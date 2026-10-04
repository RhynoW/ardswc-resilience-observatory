"""第三批 POC（A 太魯閣地震、B 中部卡努、C 嘉義凱米）Sentinel-2 取像與事件前後變化偵測：沿用 s2_area.py（規則與參數不變，不重調）。
每個區域：選 3 個事前、3 個事後（poc_candidate_scan.py 實測雲量最低者）→ 抓取 → 9 個配對的偵測與評估（事件窗口＝最晚事前～最早事後之間、框內的事件目錄多邊形聯集）。
輸出 data/biggis_interp/areas/<name>/s2/eval.json。
用法：python scripts/area_run.py A|B|C [fetch|detect]
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import s2_area as A  # noqa: E402

CFG = {
    "A": {"name": "A_taroko_quake", "bbox": (121.45, 24.05, 121.75, 24.35), "pre": ["20240120", "20240130", "20240214"], "post": ["20240404", "20240414", "20240429"], "evt_range": ("2023-06-01", "2024-12-31")},
    "B": {"name": "B_central_khanun", "bbox": (120.98, 23.85, 121.28, 24.15), "pre": ["20230619", "20230709", "20230714"], "post": ["20230828", "20230912", "20231002"], "evt_range": ("2023-05-01", "2024-06-30")},
    "C": {"name": "C_chiayi_gaemi", "bbox": (120.55, 23.35, 120.85, 23.65), "pre": ["20240628", "20240703", "20240708"], "post": ["20240822", "20240901", "20240906"], "evt_range": ("2024-05-01", "2025-03-31")},
}


def setup(k):
    c = CFG[k]
    A.NAME = c["name"]
    A.BBOX = c["bbox"]
    A.AREA = REPO / "data" / "biggis_interp" / "areas" / c["name"]
    A.OUT = A.AREA / "s2"
    A.DATES = sorted(set(c["pre"] + c["post"]))
    A.PRE, A.POST = c["pre"], c["post"]
    A.EVT_RANGE = c["evt_range"]
    ymd = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"
    A.WINDOW = (ymd(max(c["pre"])), ymd(min(c["post"])))
    return c


if __name__ == "__main__":
    c = setup(sys.argv[1])
    step = sys.argv[2] if len(sys.argv) > 2 else "all"
    if step in ("fetch", "all"):
        A.fetch()
    if step in ("detect", "all"):
        A.detect()
