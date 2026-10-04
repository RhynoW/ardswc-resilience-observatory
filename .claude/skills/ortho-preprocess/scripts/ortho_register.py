"""Wayback／航照影像「事後對位」前處理：把移動影像（mov）對到參考影像（ref）的像元格網上，消除山區殘餘錯位。

原則（來自本專案的查核與 UAV 對位經驗）：
  1. 兩期影像已在同一網格（Wayback 同 z、同範圍；或先重採樣到同一網格），只估計並修正「殘餘相對位移」。
  2. 位移在山區隨地形變化（不同拍攝角度的地形位移），單一單應矩陣不夠 → 以區塊相位相關得控制點，穩健擬合平滑位移場（多項式粗配 + 薄板樣條殘差），再 remap。
  3. 控制點要避開「真的有變化」的區域：用相位相關響應值、與鄰近一致性、MAD 離群剔除；雲與大面積變化會使區塊失敗而被剔除。
  4. 一定要留一檢核（hold-out）：20% 控制點不參與擬合，報告檢核殘差；另在修正後影像上獨立重估區塊位移。不通過 QC 就不輸出修正影像。
  5. 修正影像不是真值：位移場、檢核 RMSE 與通過與否都要隨影像保存，下游變遷偵測以此決定信心與最小可偵測面積。

用法（自 repo 根目錄）：
  python .claude/skills/ortho-preprocess/scripts/ortho_register.py diagnose --lat 23.6748 --lon 121.2115 --ref-date 20221113 --mov-date 20211029   # 先診斷

  python .claude/skills/ortho-preprocess/scripts/ortho_register.py wayback --lat 23.7741 --lon 121.1569 --half-m 900 \\
        --ref-date 20230227 --mov-date 20170218 --out E:/Temp/reg_test
  python .claude/skills/ortho-preprocess/scripts/ortho_register.py files --ref ref.png --mov mov.png --mpp 1.1 --out out_dir
輸出：mov_registered.png、field.npz（dx、dy，單位像素，mov 相對 ref）、qc.json。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))


def gradmag(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def block_shifts(ref, mov, blk=128, stride=64, min_resp=0.1):
    """回傳陣列 [(cy, cx, dx, dy, resp)]：mov 相對 ref 的位移（像素）。要對齊時以 mov 在 (x+dx, y+dy) 取樣。"""
    ga, gb = gradmag(ref), gradmag(mov)
    win = cv2.createHanningWindow((blk, blk), cv2.CV_32F)
    out = []
    H, W = ga.shape
    for y in range(0, H - blk + 1, stride):
        for x in range(0, W - blk + 1, stride):
            pa, pb = ga[y:y + blk, x:x + blk], gb[y:y + blk, x:x + blk]
            if pa.std() < 1e-3 or pb.std() < 1e-3:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(pa, pb, win)
            if resp >= min_resp and abs(dx) < blk / 3 and abs(dy) < blk / 3:
                out.append((y + blk // 2, x + blk // 2, dx, dy, resp))
    return np.array(out, np.float64).reshape(-1, 5)


def poly_terms(x, y, deg, W, H):
    u, v = (x - W / 2) / (W / 2), (y - H / 2) / (H / 2)
    cols = [np.ones_like(u)]
    for d in range(1, deg + 1):
        for i in range(d + 1):
            cols.append(u ** (d - i) * v ** i)
    return np.stack(cols, 1)


def fit_field(pts, shape, deg=2, smooth=1.0, iters=3, k=3.0):
    """控制點 → 穩健位移場。回傳 (fx, fy 完整解析度位移場, inlier 遮罩, 擬合模型)。"""
    H, W = shape
    y, x, dx, dy, r = pts.T
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        A = poly_terms(x[keep], y[keep], deg, W, H)
        w = np.sqrt(r[keep])[:, None]
        cx = np.linalg.lstsq(A * w, dx[keep] * w[:, 0], rcond=None)[0]
        cy = np.linalg.lstsq(A * w, dy[keep] * w[:, 0], rcond=None)[0]
        P = poly_terms(x, y, deg, W, H)
        res = np.hypot(dx - P @ cx, dy - P @ cy)
        mad = np.median(res[keep]) + 1e-6
        keep = res < max(k * 1.4826 * mad, 0.5)
        if keep.sum() < 12:
            break
    # 薄板樣條殘差（局部地形位移）
    P = poly_terms(x, y, deg, W, H)
    rx, ry = dx - P @ cx, dy - P @ cy
    rbf = RBFInterpolator(np.stack([x[keep], y[keep]], 1), np.stack([rx[keep], ry[keep]], 1), kernel="thin_plate_spline", smoothing=smooth * keep.sum() * 0.01)
    gx, gy = np.meshgrid(np.arange(0, W, 16), np.arange(0, H, 16))
    G = np.stack([gx.ravel() + 0.0, gy.ravel() + 0.0], 1)
    Pg = poly_terms(G[:, 0], G[:, 1], deg, W, H)
    lo = rbf(G)
    fx = (Pg @ cx + lo[:, 0]).reshape(gx.shape).astype(np.float32)
    fy = (Pg @ cy + lo[:, 1]).reshape(gx.shape).astype(np.float32)
    fx = cv2.resize(fx, (W, H), interpolation=cv2.INTER_CUBIC)
    fy = cv2.resize(fy, (W, H), interpolation=cv2.INTER_CUBIC)
    return fx, fy, keep


def warp(mov, fx, fy):
    H, W = mov.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(mov, gx + fx, gy + fy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def summarize(d, mpp):
    if len(d) == 0:
        return {"n": 0}
    m = np.hypot(d[:, 0], d[:, 1]) * mpp
    return {"n": int(len(m)), "median_m": round(float(np.median(m)), 2), "p90_m": round(float(np.percentile(m, 90)), 2), "rmse_m": round(float(np.sqrt((m ** 2).mean())), 2)}


def register(ref, mov, mpp, blk=128, stride=64, deg=2, rmse_gate_m=None, min_points=30):
    """主流程。rmse_gate_m：hold-out 殘差 RMSE 上限（預設 1.5 個像素的地面距離）。"""
    gate = rmse_gate_m if rmse_gate_m is not None else 1.5 * mpp
    H, W = ref.shape[:2]
    pts = block_shifts(ref, mov, blk, stride)
    qc = {"mpp": round(mpp, 3), "n_control_points": int(len(pts)), "gate_rmse_m": round(gate, 2),
          "before": summarize(pts[:, 2:4], mpp) if len(pts) else {"n": 0}}
    if len(pts) < min_points:
        qc.update(passed=False, reason=f"控制點 {len(pts)} < {min_points}（雲、植生變化或紋理不足）")
        return None, None, qc
    rng = np.random.default_rng(0)
    hold = rng.random(len(pts)) < 0.2
    fx, fy, inl = fit_field(pts[~hold], (H, W), deg)
    # hold-out：以位移場預測值對比觀測位移
    ys, xs = pts[hold, 0].astype(int), pts[hold, 1].astype(int)
    pred = np.stack([fx[ys, xs], fy[ys, xs]], 1)
    resid = pts[hold, 2:4] - pred
    # 只在「原本就一致」的控制點上算（用 MAD 剔除真變化造成的離群）
    rr = np.hypot(resid[:, 0], resid[:, 1])
    ok = rr < max(3 * 1.4826 * (np.median(rr) + 1e-6), 0.5)
    qc["holdout"] = {"n": int(hold.sum()), "n_inlier": int(ok.sum()), **summarize(resid[ok], mpp)}
    # 以全部控制點重新擬合並輸出
    fx, fy, inl = fit_field(pts, (H, W), deg)
    out = warp(mov, fx, fy)
    after = block_shifts(ref, out, blk, stride)
    qc["after"] = summarize(after[:, 2:4], mpp) if len(after) else {"n": 0}
    qc["field_median_m"] = round(float(np.median(np.hypot(fx, fy))) * mpp, 2)
    qc["field_p95_m"] = round(float(np.percentile(np.hypot(fx, fy), 95)) * mpp, 2)
    imp = qc["before"].get("median_m", 0) - qc["after"].get("median_m", 0)
    qc["passed"] = bool(qc["holdout"]["rmse_m"] <= gate and qc["after"].get("p90_m", 1e9) <= qc["before"].get("p90_m", 0) * 1.0 + 1e-6)
    qc["reason"] = "通過" if qc["passed"] else "hold-out RMSE 超過門檻或修正後 p90 未改善：不採用修正影像"
    qc["median_improvement_m"] = round(imp, 2)
    return out, (fx, fy), qc


def cmd_wayback(a):
    import wayback_assist as W
    frames = W.list_frames(a.lat, a.lon)
    by = {f["date"]: f for f in frames}
    for d in (a.ref_date, a.mov_date):
        if d not in by:
            sys.exit(f"此點沒有 {d} 的 Wayback 影像；可用日期：{sorted(by)}")
    rels = {r["num"]: r for r in W.releases()}
    ref, _ = W.fetch_image(a.lat, a.lon, a.half_m, by[a.ref_date]["release"])
    mov, _ = W.fetch_image(a.lat, a.lon, a.half_m, by[a.mov_date]["release"])
    mpp = 156543.03392 * math.cos(math.radians(a.lat)) / 2 ** W.DEFAULT_ZOOM
    run(ref, mov, mpp, a)


def diagnose(ref, mov, mpp):
    """只診斷不修正：兩期的相對錯位分布與建議用法（以像素為單位，與影像解析度無關）。"""
    pts = block_shifts(ref, mov, 128, 64)
    if len(pts) < 20:
        return {"n": int(len(pts)), "verdict": "無法診斷：控制點太少（雲、植生變化或紋理不足）", "mpp": round(mpp, 3)}
    d = np.hypot(pts[:, 2], pts[:, 3])
    med, p90 = float(np.median(d)), float(np.percentile(d, 90))
    out = {"mpp": round(mpp, 3), "n": int(len(d)), "median_px": round(med, 2), "p90_px": round(p90, 2), "median_m": round(med * mpp, 2), "p90_m": round(p90 * mpp, 2),
           "mean_dx_m": round(float(pts[:, 2].mean()) * mpp, 2), "mean_dy_m": round(float(pts[:, 3].mean()) * mpp, 2)}
    if p90 <= 2.0 and med <= 1.0:
        out["verdict"] = "A 可做像素級變遷偵測（相對錯位 ≤2 px）"
    elif p90 <= 5.0:
        out["verdict"] = f"B 只做物件級變遷（≥0.5 ha 或邊界外擴 {round(p90 * mpp, 1)} m 緩衝）；不要用像素差異"
    else:
        out["verdict"] = "C 不做像素級；改用已配準影像（如 Sentinel-2）或取得原始影像重新正射（RPC＋DEM）"
    return out


def cmd_diagnose(a):
    import wayback_assist as W
    frames = W.list_frames(a.lat, a.lon)
    by = {f["date"]: f for f in frames}
    for d in (a.ref_date, a.mov_date):
        if d not in by:
            sys.exit(f"此點沒有 {d} 的 Wayback 影像；可用日期：{sorted(by)}")
    ref, _ = W.fetch_image(a.lat, a.lon, a.half_m, by[a.ref_date]["release"])
    mov, _ = W.fetch_image(a.lat, a.lon, a.half_m, by[a.mov_date]["release"])
    mpp = 156543.03392 * math.cos(math.radians(a.lat)) / 2 ** W.DEFAULT_ZOOM
    print(json.dumps(diagnose(ref, mov, mpp), ensure_ascii=False, indent=1))


def cmd_files(a):
    ref, mov = cv2.imread(a.ref), cv2.imread(a.mov)
    if ref.shape != mov.shape:
        mov = cv2.resize(mov, (ref.shape[1], ref.shape[0]))
    run(ref, mov, a.mpp, a)


def run(ref, mov, mpp, a):
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res, field, qc = register(ref, mov, mpp, deg=a.deg)
    cv2.imwrite(str(out_dir / "ref.png"), ref)
    cv2.imwrite(str(out_dir / "mov_original.png"), mov)
    if res is not None and qc["passed"]:
        cv2.imwrite(str(out_dir / "mov_registered.png"), res)
        np.savez_compressed(out_dir / "field.npz", dx=field[0], dy=field[1], mpp=mpp)
    (out_dir / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(qc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("wayback")
    w.add_argument("--lat", type=float, required=True)
    w.add_argument("--lon", type=float, required=True)
    w.add_argument("--half-m", type=float, default=900)
    w.add_argument("--ref-date", required=True)
    w.add_argument("--mov-date", required=True)
    w.add_argument("--deg", type=int, default=2)
    w.add_argument("--out", required=True)
    dg = sp.add_parser("diagnose")
    dg.add_argument("--lat", type=float, required=True)
    dg.add_argument("--lon", type=float, required=True)
    dg.add_argument("--half-m", type=float, default=450)
    dg.add_argument("--ref-date", required=True)
    dg.add_argument("--mov-date", required=True)
    f = sp.add_parser("files")
    f.add_argument("--ref", required=True)
    f.add_argument("--mov", required=True)
    f.add_argument("--mpp", type=float, required=True)
    f.add_argument("--deg", type=int, default=2)
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    {"wayback": cmd_wayback, "files": cmd_files, "diagnose": cmd_diagnose}[a.cmd](a)
