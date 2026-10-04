"""S1（RTC）與 S2 的座標對位查證（植生法）：S2 ExG（植生）對 S1 後向散射（VH／VV 的 dB）做區塊相位相關（128 px、步距 64），
報告各軌道、各極化、事前／事後的中位位移（公尺；mov＝S1 相對 ref＝S2，+x＝東、+y＝南）。結果見 ARCHITECTURE_v3.md 附八之三。
用法：python scripts/s1_s2_registration_veg.py
"""
import sys,warnings,numpy as np,cv2
warnings.filterwarnings("ignore")
sys.path.insert(0,'scripts')
import s2_area as A, s1_area as S1
tf,(H,W)=A.grid()
sl=A.slope(tf,H,W)
def exg(d):
    a=np.load(A.OUT/f'rgb_{d}.npy').astype(np.float32);r,g,b=a[...,0],a[...,1],a[...,2]
    return (2*g-r-b)/(r+g+b+1e-6)
def hp(a):
    a=np.where(np.isfinite(a),a,np.nanmedian(a)).astype(np.float32);return cv2.GaussianBlur(a,(0,0),1.5)-cv2.GaussianBlur(a,(0,0),15)
def bs(ref,mov,blk=128,stride=64,minresp=0.05):
    win=cv2.createHanningWindow((blk,blk),cv2.CV_32F);out=[]
    for y in range(0,H-blk+1,stride):
        for x in range(0,W-blk+1,stride):
            pa,pb=ref[y:y+blk,x:x+blk],mov[y:y+blk,x:x+blk]
            if pa.std()<1e-6 or pb.std()<1e-6: continue
            (dx,dy),rr=cv2.phaseCorrelate(pa,pb,win)
            if rr>=minresp and abs(dx)<15 and abs(dy)<15: out.append((y+blk//2,x+blk//2,dx*10,dy*10,rr))
    return np.array(out).reshape(-1,5)
S1.ORBITS={"desc105":None,"asc69":None}
db=lambda v:10*np.log10(np.maximum(v,1e-6))
rows={}
for tag,s2d in (('pre','20250615'),('post','20251011')):
    e=exg(s2d)
    for key in ('desc105','asc69'):
        for pol in ('vh','vv'):
            m,_=S1.stack(key,tag,pol)
            s=bs(hp(e),hp(db(m)))
            if len(s)==0: print(tag,key,pol,'n=0');continue
            sv=sl[s[:,0].astype(int),s[:,1].astype(int)]
            q=lambda a:round(float(np.median(a)),1)
            # 以 resp 加權前 50% 的穩健估計
            k=s[:,4]>=np.median(s[:,4]);
            print(f'S2 {s2d} vs S1 {key} {pol} {tag}: n={len(s)} resp={np.median(s[:,4]):.2f} median dx(東)={q(s[:,2])} m dy(南)={q(s[:,3])} m | 高響應半數 dx={q(s[k,2])} dy={q(s[k,3])} | 陡坡(>30°) dx={q(s[sv>30,2]) if (sv>30).sum()>5 else None} dy={q(s[sv>30,3]) if (sv>30).sum()>5 else None}')
