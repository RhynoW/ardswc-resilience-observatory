"""學習式匹配（LoFTR）UAV → 正射底圖：多旋轉 × 多尺度 × 分塊，MAGSAC 粗差剔除。
流程對應使用者提供的作業流程：(1) 降採樣到參考解析度 + 灰階 + CLAHE  (3) 學習式匹配 + 分塊  (4) MAGSAC。"""
import json, math, sys, time
import cv2, numpy as np, torch, kornia.feature as KF
import register as R

dev = R.pick_device()   # UAV_DEVICE=cpu 可強制 CPU；未設定時有 CUDA 就用 GPU
if dev == "cpu":
    torch.set_num_threads(6)
print("device", dev, flush=True)
matcher = KF.LoFTR(pretrained="outdoor").to(dev).eval()

REF_NAME = sys.argv[1] if len(sys.argv) > 1 else "esri"      # esri 沒有 NLSC 的重複浮水印
Z, HALF = 17, 4
R.LAT, R.LON = 24.2092479, 121.6600388                       # 石公溪
import os
_cx, _cy = R.tile_xy(R.LAT, R.LON, Z); x0, y0 = int(_cx) - HALF, int(_cy) - HALF
if os.path.exists(f"ref17_{REF_NAME}.jpg"):                  # 前次已抓好，不重抓
    ref = cv2.imread(f"ref17_{REF_NAME}.jpg")
else:
    ref, x0, y0 = R.mosaic(REF_NAME, Z, HALF)                # 2304×2304 ≈ 2.5 km
ts = R.px_to_merc(0, 0, x0, y0, Z)[2] * math.cos(math.radians(R.LAT))
print("ref", ref.shape, "gsd", round(ts, 3), flush=True)
cv2.imwrite(f"ref17_{REF_NAME}.jpg", ref, [cv2.IMWRITE_JPEG_QUALITY, 90])

clahe = cv2.createCLAHE(2.5, (8, 8))
prep = lambda im: clahe.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
uav = cv2.imread("uav.jpg")
uav[int(uav.shape[0] * 0.93):, :int(uav.shape[1] * 0.46)] = 0            # 遮授權文字
ref_g = prep(ref)

def to_t(g):
    return torch.from_numpy(g)[None, None].float().to(dev) / 255.0

def rot(im, deg):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(im, M, (nw, nh), flags=cv2.INTER_AREA), M

WIN, STRIDE = 832, 512
results = []
t0 = time.time()
for gsd in (0.3, 0.4, 0.55):
    s = gsd / ts
    small = cv2.resize(prep(uav), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for deg in range(0, 360, 30):
        rimg, M = rot(small, deg)
        # 統一成 8 的倍數、且不超過 WIN
        f = min(1.0, WIN / max(rimg.shape))
        if f < 1.0:
            rimg = cv2.resize(rimg, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        h8, w8 = (rimg.shape[0] // 8) * 8, (rimg.shape[1] // 8) * 8
        rimg = rimg[:h8, :w8]
        for wy in (600, 868):
            for wx in (1200, 1472):
                sub = ref_g[wy:wy + WIN, wx:wx + WIN]
                sh, sw = (sub.shape[0] // 8) * 8, (sub.shape[1] // 8) * 8
                sub = sub[:sh, :sw]
                with torch.no_grad():
                    out = matcher({"image0": to_t(rimg), "image1": to_t(sub)})
                conf = out["confidence"].cpu().numpy()
                keep = conf > 0.5
                if keep.sum() < 12:
                    continue
                p0 = out["keypoints0"].cpu().numpy()[keep]; p1 = out["keypoints1"].cpu().numpy()[keep]
                H, inl = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
                if H is None:
                    continue
                ni = int(inl.sum())
                results.append({"gsd": gsd, "rot": deg, "win": [wx, wy], "matches": int(keep.sum()), "inliers": ni, "f": f})
    print(f"gsd {gsd} done, best inliers {max((r['inliers'] for r in results), default=0)}, {time.time()-t0:.0f}s", flush=True)
    json.dump({"ref": REF_NAME, "z": Z, "x0": x0, "y0": y0, "results": sorted(results, key=lambda r: -r["inliers"])[:30]}, open(f"loftr_{REF_NAME}.json", "w"))

results.sort(key=lambda r: -r["inliers"])
print(json.dumps(results[:8], indent=1))
json.dump({"ref": REF_NAME, "z": Z, "x0": x0, "y0": y0, "results": results[:30]}, open(f"loftr_{REF_NAME}.json", "w"))
