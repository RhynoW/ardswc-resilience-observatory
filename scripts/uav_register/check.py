import json, sys, cv2, numpy as np
import register as R
from batch import mosaic_par
eid=sys.argv[1]; D=f'F:/GitHub/ardswc-resilience-observatory/webapp/change_detect_viewer/static/uav/{eid}/'
m=json.load(open(D+'meta.json',encoding='utf-8')); b=m['bounds_lonlat']; rgba=cv2.imread(D+'rect.png',cv2.IMREAD_UNCHANGED)
clon,clat=m['center_lonlat']; half=3 if m['quality']['footprint_ha']>120 else 2
ref,x0,y0=mosaic_par(clat,clon,17,half)
def px(lon,lat):
    mx,my=R.merc(lon,lat); ox,oy,ts=R.px_to_merc(0,0,x0,y0,17); return (mx-ox)/ts,(oy-my)/ts
xa,ya=px(b['west'],b['north']); xb,yb=px(b['east'],b['south'])
w,h=int(round(xb-xa)),int(round(yb-ya)); ov=cv2.resize(rgba,(w,h)); xa,ya=int(round(xa)),int(round(ya))
out=ref.copy(); H,W=out.shape[:2]
x1,y1,x2,y2=max(xa,0),max(ya,0),min(xa+w,W),min(ya+h,H)
reg=out[y1:y2,x1:x2]; o=ov[y1-ya:y2-ya,x1-xa:x2-xa]; a=(o[...,3:4]/255.0)*0.6
reg[:]=(reg*(1-a)+o[...,:3]*a).astype(np.uint8)
side=np.hstack([cv2.resize(ref,(700,700)),cv2.resize(out,(700,700))])
cv2.imwrite(f'check_{eid}.jpg',side,[cv2.IMWRITE_JPEG_QUALITY,85]); print(m['title'],m['quality']['holdout_rmse_m'])
