"""GPU 過熱保護：溫度 >= PAUSE_C 就暫停，降到 RESUME_C 以下才繼續（本機 GPU 曾在長時間運算後當機）。
環境變數：UAV_GPU_PAUSE_C（預設 78）、UAV_GPU_RESUME_C（預設 65）、UAV_GPU_POLL_S（預設 5）。
讀不到 nvidia-smi（CPU 模式）時直接放行。"""
import os, subprocess, time

PAUSE_C = int(os.environ.get("UAV_GPU_PAUSE_C", "70"))
RESUME_C = int(os.environ.get("UAV_GPU_RESUME_C", "58"))
POLL_S = int(os.environ.get("UAV_GPU_POLL_S", "5"))
_last = 0.0


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(int(x) for x in out.split())
    except Exception:  # noqa: BLE001
        return None


DUTY = float(os.environ.get("UAV_GPU_DUTY", "1.5"))   # 每次推論後休息 = 推論時間 × DUTY（無管理員權限不能降功耗上限，改用軟體限流）


def duty_sleep(busy_s):
    time.sleep(min(busy_s * DUTY, 10))


def wait_cool(tag="", min_interval=2.0):
    """呼叫端在每次 GPU 重運算前呼叫；最多每 min_interval 秒才真的查一次溫度。"""
    global _last
    if time.time() - _last < min_interval:
        return
    t = gpu_temp()
    if t is not None and t >= PAUSE_C:
        t0 = time.time()
        print(f"[thermal] GPU {t}°C ≥ {PAUSE_C}°C，暫停降溫 {tag}", flush=True)
        while t is not None and t > RESUME_C:
            time.sleep(POLL_S)
            t = gpu_temp()
        print(f"[thermal] 降到 {t}°C，繼續（等了 {time.time() - t0:.0f} s）", flush=True)
    _last = time.time()
