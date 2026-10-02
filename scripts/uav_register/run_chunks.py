"""UAV 對位分小批執行（每批 CHUNK 個候選，每批一個全新子行程），可續跑——
本機 GPU 曾在長時間連續運算後當機，分批＋批間休息能降低風險，且每批結束就寫進度檔，當機後從下一批接續。
用法：python run_chunks.py [目標總數=30] [CHUNK=20]
環境：UAV_DEVICE（預設 cuda）、UAV_HOLD_MAX（預設 10）、UAV_MIN_SEP_KM（預設 0.25）、UAV_REF
候選：batch_cands_2025.json（select_cands.py 產出）；進度：run_chunks_progress.json（下一個要試的候選索引）
"""
import json, os, subprocess, sys, time
from pathlib import Path
import thermal_guard as TG

HERE = Path(__file__).parent
DEST = Path("F:/GitHub/ardswc-resilience-observatory/webapp/change_detect_viewer/static/uav")
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 20
cands = json.loads((HERE / os.environ.get("UAV_CANDS_ALL", "batch_cands_2025.json")).read_text(encoding="utf-8"))
prog_f = HERE / "run_chunks_progress.json"
pos = json.loads(prog_f.read_text())["next"] if prog_f.exists() else int(os.environ.get("UAV_START", "0"))
env = {**os.environ, "UAV_DEVICE": os.environ.get("UAV_DEVICE", "cuda"), "UAV_HOLD_MAX": os.environ.get("UAV_HOLD_MAX", "10"),
       "UAV_MIN_SEP_KM": os.environ.get("UAV_MIN_SEP_KM", "0.25"), "PYTHONIOENCODING": "utf-8"}
while pos < len(cands):
    have = len(list(DEST.glob("*/meta.json")))
    if have >= TARGET:
        print(f"已達 {have} 個樣本，停止", flush=True)
        break
    TG.wait_cool("(chunk 開始前)", min_interval=0)
    chunk = cands[pos:pos + CHUNK]
    (HERE / "batch_cands.json").write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
    print(f"[chunk] 候選 {pos}–{pos + len(chunk) - 1}，目前 {have} 個，還需 {TARGET - have}", flush=True)
    r = subprocess.run([sys.executable, "batch.py", str(TARGET - have)], cwd=HERE, env=env)
    print(f"[chunk] 結束 rc={r.returncode}", flush=True)
    pos += len(chunk)
    prog_f.write_text(json.dumps({"next": pos}))
    time.sleep(20)                       # 批間休息，讓 GPU 降溫
print("完成", len(list(DEST.glob("*/meta.json"))), "個樣本，下一個候選索引", pos, flush=True)
