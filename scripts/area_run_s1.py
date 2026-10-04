"""第三批 POC（A 太魯閣地震、B 中部卡努、C 嘉義凱米）Sentinel-1 RTC 事件前後變化偵測（s1_area.py，規則不變，K=4 固定，另報 K=2,3 敏感度）。
事前窗口＝事件前 ~50–60 天，事後窗口＝事件後約 1 個月；軌道由 discover() 選同時有事前與事後景者（景數最多的前兩個）。
用法：python scripts/area_run_s1.py A|B|C [fetch|detect|all]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import area_run as AR  # noqa: E402
import s1_area as S1  # noqa: E402

WIN = {
    "A": (("2024-01-15", "2024-04-01"), ("2024-04-05", "2024-05-10")),
    "B": (("2023-06-10", "2023-07-25"), ("2023-08-04", "2023-08-30")),
    "C": (("2024-06-01", "2024-07-20"), ("2024-07-27", "2024-08-30")),
}

if __name__ == "__main__":
    k = sys.argv[1]
    AR.setup(k)
    S1.PRE_WIN, S1.POST_WIN = WIN[k]
    orb = S1.discover()
    S1.ORBITS = {n: v for n, v in list(orb.items())[:2]}
    print("軌道", S1.ORBITS, flush=True)
    step = sys.argv[2] if len(sys.argv) > 2 else "all"
    if step in ("fetch", "all"):
        S1.fetch()
    if step in ("detect", "all"):
        S1.detect()
