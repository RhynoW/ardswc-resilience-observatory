"""合併多窗口 LoFTR 匹配 → 單一單應矩陣（UAV 全解析像素 → z17 底圖像素）→ 留一檢核 → 糾正輸出。"""
import json, math, sys
import cv2, numpy as np, torch, kornia.feature as KF
import register as R

torch.set_num_threads(4)
GSD, ROT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3, int(sys.argv[2]) if len(sys.argv) > 2 else 150
REF = "esri"; Z = 17
R.LAT, R.LON = 24.2092479, 121.6600388
cx, cy = R.tile_xy(R.LAT, R.LON, Z); x0, y0 = int(cx) - 4, int(cy) - 4
ref = cv2.imread(f"ref17_{REF}.jpg")
ts_merc = R.px_to_merc(0, 0, x0, y0, Z)[2]                       # z17 每像素 Web Mercator 公尺
ts = ts_merc * math.cos(math.radians(R.LAT))                     # 地面 m/px
matcher = KF.LoFTR(pretrained="outdoor").eval()
clahe = cv2.createCLAHE(2.5, (8, 8))
prep = lambda im: clahe.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
uav = cv2.imread("uav.jpg"); uav_m = uav.copy()
uav_m[int(uav.shape[0] * 0.93):, :int(uav.shape[1] * 0.46)] = 0
ref_g = prep(ref)
to_t = lambda g: torch.from_numpy(g)[None, None].float() / 255.0

s = GSD / ts
small = cv2.resize(prep(uav_m), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
h, w = small.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), ROT, 1.0)
c_, s_ = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s_ + w * c_), int(h * c_ + w * s_)
M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
rimg = cv2.warpAffine(small, M, (nw, nh), flags=cv2.INTER_AREA)
rimg = rimg[:(nh // 8) * 8, :(nw // 8) * 8]
Minv = cv2.invertAffineTransform(M)

P_uav, P_ref, WIN_ID = [], [], []
for wi, (wy, wx) in enumerate([(600, 1200), (600, 1472), (868, 1200), (868, 1472)]):
    sub = ref_g[wy:wy + 832, wx:wx + 832]; sub = sub[:(sub.shape[0] // 8) * 8, :(sub.shape[1] // 8) * 8]
    with torch.no_grad():
        out = matcher({"image0": to_t(rimg), "image1": to_t(sub)})
    keep = out["confidence"].cpu().numpy() > 0.5
    p0 = out["keypoints0"].cpu().numpy()[keep]; p1 = out["keypoints1"].cpu().numpy()[keep]
    # 還原到 UAV 全解析像素：反旋轉 → 反縮放
    q = (Minv @ np.c_[p0, np.ones(len(p0))].T).T / s
    P_uav.append(q); P_ref.append(p1 + np.array([wx, wy])); WIN_ID += [wi] * len(p0)
P_uav = np.vstack(P_uav).astype(np.float32); P_ref = np.vstack(P_ref).astype(np.float32); WIN_ID = np.array(WIN_ID)
print("tentative matches", len(P_uav))

# 去重（相鄰窗口重疊區會重複）
_, ui = np.unique(np.round(np.c_[P_uav, P_ref] / 2).astype(int), axis=0, return_index=True)
P_uav, P_ref, WIN_ID = P_uav[ui], P_ref[ui], WIN_ID[ui]
H, inl = cv2.findHomography(P_uav, P_ref, cv2.USAC_MAGSAC, 3.0, maxIters=20000, confidence=0.9999)
inl = inl.ravel().astype(bool); print("joint inliers", int(inl.sum()), "of", len(P_uav))
proj = cv2.perspectiveTransform(P_uav[inl][None], H)[0]; err = np.linalg.norm(proj - P_ref[inl], axis=1) * ts
print(f"fit RMSE {np.sqrt((err**2).mean()):.2f} m, median {np.median(err):.2f} m, 90% {np.percentile(err,90):.2f} m")
pts = P_uav[inl]; hh, ww = uav.shape[:2]
print("inlier coverage x[%.0f%%–%.0f%%] y[%.0f%%–%.0f%%]" % (pts[:,0].min()/ww*100, pts[:,0].max()/ww*100, pts[:,1].min()/hh*100, pts[:,1].max()/hh*100))
print("inliers per window", {int(k): int((WIN_ID[inl] == k).sum()) for k in np.unique(WIN_ID[inl])})

# 檢核：隨機留 30% 內點不參與擬合，量測其殘差（誠實的精度指標）
rng = np.random.default_rng(1); idx = np.where(inl)[0]; rng.shuffle(idx)
k = int(len(idx) * 0.7); tr, te = idx[:k], idx[k:]
H2, _ = cv2.findHomography(P_uav[tr], P_ref[tr], 0)
pe = cv2.perspectiveTransform(P_uav[te][None], H2)[0]; e2 = np.linalg.norm(pe - P_ref[te], axis=1) * ts
print(f"HOLD-OUT ({len(te)} pts) RMSE {np.sqrt((e2**2).mean()):.2f} m, median {np.median(e2):.2f} m, max {e2.max():.2f} m")

# 幾何合理性
corners = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
cr = cv2.perspectiveTransform(corners[None], H)[0]
area = cv2.contourArea(cr) * ts * ts
ang = math.degrees(math.atan2(H[1, 0], H[0, 0]))
print(f"footprint area {area/1e4:.1f} ha, corners convex {cv2.isContourConvex(cr.astype(np.float32))}, rot~{ang:.0f} deg")
print("UAV GSD at center ~", round(float(np.linalg.norm(cv2.perspectiveTransform(np.float32([[[ww/2+1,hh/2]]]),H)[0][0]-cv2.perspectiveTransform(np.float32([[[ww/2,hh/2]]]),H)[0][0])) * ts, 3), "m/px")
json.dump({"H": H.tolist(), "x0": x0, "y0": y0, "z": Z, "n_inl": int(inl.sum()), "holdout_rmse_m": float(np.sqrt((e2**2).mean())),
           "fit_rmse_m": float(np.sqrt((err**2).mean())), "corners_ref_px": cr.tolist(), "gsd": GSD, "rot": ROT}, open("final_reg.json", "w"))

# 糾正輸出：以 UAV 原生解析度（約 0.3 m）重採樣到北在上的 EPSG:3857 網格，另存疊圖
UP = 4                                                              # z17 像素 → 輸出像素放大倍數
x_min, y_min = np.floor(cr.min(0)).astype(int); x_max, y_max = np.ceil(cr.max(0)).astype(int)
S = np.array([[UP, 0, -x_min * UP], [0, UP, -y_min * UP], [0, 0, 1]], np.float64)
ow, oh = (x_max - x_min) * UP, (y_max - y_min) * UP
warped = cv2.warpPerspective(uav, S @ H, (ow, oh), flags=cv2.INTER_LANCZOS4)
valid = cv2.warpPerspective(np.full((hh, ww), 255, np.uint8), S @ H, (ow, oh), flags=cv2.INTER_NEAREST) > 0
cv2.imwrite("final_rect.png", warped)
mx, my, _ = R.px_to_merc(x_min, y_min, x0, y0, Z); pm = ts_merc / UP
open("final_rect.jgw", "w").write("\n".join(f"{v:.8f}" for v in [pm, 0, 0, -pm, mx + pm / 2, my - pm / 2]))
big = cv2.resize(ref[y_min:y_max, x_min:x_max], (ow, oh), interpolation=cv2.INTER_CUBIC)
ov = big.copy(); ov[valid] = (0.5 * warped[valid] + 0.5 * big[valid]).astype(np.uint8)
cv2.imwrite("final_overlay.jpg", cv2.resize(ov, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
# 並排：底圖 vs 糾正後 UAV（同範圍）
side = np.hstack([cv2.resize(big, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA), cv2.resize(warped, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)])
cv2.imwrite("final_side.jpg", side, [cv2.IMWRITE_JPEG_QUALITY, 88])
ll = [R.unmerc(*R.px_to_merc(px, py, x0, y0, Z)[:2]) for px, py in cr]
print("corners lon/lat:", [[round(a, 6), round(b, 6)] for a, b in ll])
c = cv2.perspectiveTransform(np.float32([[[ww / 2, hh / 2]]]), H)[0][0]; cl = R.unmerc(*R.px_to_merc(c[0], c[1], x0, y0, Z)[:2])
print("center lon/lat:", [round(cl[0], 6), round(cl[1], 6)], "output", (ow, oh))
