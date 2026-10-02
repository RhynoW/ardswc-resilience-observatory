"""對「LoFTR 粗對位失敗」的斜拍候選逐一跑 DTM 合成斜視圖法（scripts/uav_synth.py），通過者直接發布到 static/uav/。
每個候選一個全新子行程（避免 GPU 記憶體累積）；結果寫進 synth_progress.json，可續跑；達到目標總數就停。
用法：python run_synth.py [目標總數=30] [候選檔=batch_cands_2026.json]
環境：UAV_HOLD_MAX（預設 10）、UAV_MIN_SEP_KM（預設 0.25）、UAV_GPU_*（見 thermal_guard.py）
"""
import json, os, subprocess, sys, time
from pathlib import Path

import register as R

HERE = Path(__file__).parent
SCRIPT = HERE.parent / "uav_synth.py"
DEST = HERE.parent.parent / "webapp/change_detect_viewer/static/uav"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CANDS = HERE / (sys.argv[2] if len(sys.argv) > 2 else "batch_cands_2026.json")
SEP = float(os.environ.get("UAV_MIN_SEP_KM", "0.25"))
prog_f = HERE / f"synth_progress_{CANDS.stem}.json"
prog = json.loads(prog_f.read_text(encoding="utf-8")) if prog_f.exists() else {}
env = {**os.environ, "PYTHONIOENCODING": "utf-8", "UAV_HOLD_MAX": os.environ.get("UAV_HOLD_MAX", "10")}
cands = json.loads(CANDS.read_text(encoding="utf-8"))
by_id = {r["EventID"]: r for r in cands}
FAIL_SEP = float(os.environ.get("UAV_FAIL_SEP_KM", "0.3"))     # 同一地點連拍：附近已有失敗者就先跳過，優先試不同地點
for rec in cands:
    eid = rec["EventID"]
    metas = list(DEST.glob("*/meta.json"))
    have = len(metas)
    if have >= TARGET:
        print(f"已達 {have} 個樣本，停止", flush=True)
        break
    if eid in prog or (DEST / eid / "meta.json").exists():
        continue
    pts = [tuple(json.loads(m.read_text(encoding="utf-8"))["center_lonlat"]) for m in metas]
    if any(R.hav(rec["Lng"], rec["Lat"], *p) < SEP * 1000 for p in pts):
        prog[eid] = "太近"; prog_f.write_text(json.dumps(prog, ensure_ascii=False), encoding="utf-8"); continue
    failed = [(by_id[k]["Lng"], by_id[k]["Lat"]) for k, v in prog.items() if k in by_id and v != "太近" and v != "近失敗"]
    if any(R.hav(rec["Lng"], rec["Lat"], *q) < FAIL_SEP * 1000 for q in failed):
        prog[eid] = "近失敗"; prog_f.write_text(json.dumps(prog, ensure_ascii=False), encoding="utf-8"); continue
    print(f"[{have}/{TARGET}] {eid} {rec['County']}{rec['Town']} {rec['Lat']:.5f},{rec['Lng']:.5f}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(SCRIPT), eid, "--publish"], env=env, cwd=HERE.parent,
                       stdout=open(HERE / "work_synth" / f"{eid}.log", "w", encoding="utf-8") if (HERE / "work_synth").exists() else None,
                       stderr=subprocess.STDOUT)
    prog[eid] = {0: "通過", 1: "未通過", 2: "粗搜失敗", 3: "精修失敗"}.get(r.returncode, f"rc={r.returncode}")
    prog_f.write_text(json.dumps(prog, ensure_ascii=False), encoding="utf-8")
    print(f"  → {prog[eid]}（{time.time() - t0:.0f}s）", flush=True)
    time.sleep(15)
print("完成，目前", len(list(DEST.glob('*/meta.json'))), "個樣本", flush=True)
