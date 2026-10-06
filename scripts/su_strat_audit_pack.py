"""把分層人工驗證的判讀流程打包成可回溯、可稽核的公開檔（放進 webapp/change_detect_viewer/static/poc/audit/，隨 Space 部署）：
  review.html           自足的判讀表（24 張圖縮成 JPEG 以 data URI 內嵌；與原判讀表相同的題目與選項；網址加 #audit 進入唯讀稽核檢視並載入 su_strat_human.json，不寫入瀏覽器儲存）
  su_strat_human.json   人工判讀結果（原樣）
  offset_check.json     輪廓偏移查證（各坡面 Wayback 對 Sentinel-2 的實測偏移與全區 Sentinel-2 對 DTM 的位移）
  manifest.json         稽核清單：每個檔案的 SHA-256、程式 commit、資料版本（SU 版本、參數）、選樣設計與種子、判讀者與流程說明、
                        以及「不公開」檔案（key.json 層別對照、selected.json、各層候選名單）的 SHA-256——承諾（commit）先公布雜湊，專業人員盲判完成後再揭露原檔供核對
不公開 key.json 的理由：之後 2–3 位專業人員要對同一批 24 個坡面盲判，公開「編號→層別」會破壞盲測；雜湊先公布即可證明事後未改動。
雜湊一律以 LF 位元組計算；本資料夾在 .gitattributes 設為 -text，避免 Git 換行轉換改變內容。
用法：python scripts/su_strat_audit_pack.py
"""
import base64
import datetime
import hashlib
import json
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "biggis_interp" / "poc" / "su_strat"
POC = REPO / "data" / "biggis_interp" / "poc"
DST = REPO / "webapp" / "change_detect_viewer" / "static" / "poc" / "audit"
IMG_W, IMG_Q = 1800, 76
CRLF, LF = b"\r\n", b"\n"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes().replace(CRLF, LF)).hexdigest()


def wlf(path, text):
    """以 UTF-8＋LF 位元組寫檔（不經 Windows 換行轉換），使雜湊在各平台一致。"""
    Path(path).write_bytes(text.encode("utf-8").replace(CRLF, LF))


def data_uri(png):
    im = cv2.imread(str(png))
    im = cv2.resize(im, (IMG_W, int(im.shape[0] * IMG_W / im.shape[1])), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, IMG_Q])[1].tobytes()).decode()


AUDIT_JS = """
<script>
(async function(){
  if(location.hash!=="#audit") return;
  const banner=document.createElement('div');
  banner.style.cssText='background:#fff3cd;border:1px solid #e0b100;padding:8px 14px;margin:0 0 12px;font-size:14px';
  banner.innerHTML='<b>稽核檢視（唯讀）</b>：下列為 su_strat_human.json 的判讀結果（1 位專案成員的人工判讀；判讀者看不到層別與系統資訊）。檔案雜湊見 <a href="manifest.json">manifest.json</a>。要重新判讀請移除網址中的 #audit。';
  document.querySelector('main').prepend(banner);
  IDS.forEach(i=>{document.querySelectorAll('#it_'+i+' input').forEach(e=>{e.checked=false;});document.getElementById('n_'+i).value='';});
  let H={};
  try{ H=await (await fetch('su_strat_human.json')).json(); }catch(e){ banner.innerHTML+='<br>（載入 su_strat_human.json 失敗）'; }
  window.sv=function(){}; window.exp=function(){};
  IDS.forEach(i=>{const s=H[i]; if(!s) return;
    const v=document.querySelector('input[name=v_'+i+'][value='+s.verdict+']'); if(v) v.checked=true;
    const c=document.querySelector('input[name=c_'+i+'][value='+s.conf+']'); if(c) c.checked=true;
    document.getElementById('s_'+i).checked=!!s.small; document.getElementById('n_'+i).value=s.note||'';
    document.getElementById('it_'+i).classList.add('done');
    document.querySelectorAll('#it_'+i+' input').forEach(e=>{e.disabled=true;});});
  document.querySelectorAll('header button').forEach(b=>b.style.display='none');
  document.getElementById('p').textContent='　稽核檢視：'+Object.keys(H).length+' / '+IDS.length+' 已判讀';
})();
</script>
"""


def main():
    DST.mkdir(parents=True, exist_ok=True)
    html = (SRC / "review.html").read_text(encoding="utf-8")
    ids = sorted(p.stem for p in SRC.glob("SV*.png"))
    for i in ids:
        uri = data_uri(SRC / f"{i}.png")
        html = html.replace(f'<a href="{i}.png" target="_blank"><img src="{i}.png" loading="lazy"></a>', f'<img src="{uri}" alt="{i} 四欄影像：Wayback 較早、最新，Sentinel-2 兩期，DTM 參考" loading="lazy">')
    html += AUDIT_JS
    wlf(DST / "review.html", html)
    for f in ("su_strat_human.json", "offset_check.json"):
        wlf(DST / f, (SRC / f).read_text(encoding="utf-8"))
    key_files = {f: sha(SRC / f) for f in ("key.json", "selected.json", "pool_summary.json")}
    data_files = {f: sha(POC / f) for f in ("su_table_eff.json", "su_meta_eff.json", "su_evidence.json", "su_lab10_eff.npy", "s2/det_20240404_A_fixed.npy", "s2/det_20250325_A_fixed.npy", "reverse/river.npy")}
    meta = json.load(open(POC / "su_meta_eff.json", encoding="utf-8"))
    manifest = {
        "title": "分層人工驗證（24 個坡面）判讀流程稽核清單",
        "built": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "code": {"repo": "RhynoW/ardswc-resilience-observatory", "commit_of_tooling": "a8fcd89ea25346d9770ab6ebd6fd22b8468c9575（選樣、判讀圖、偏移查證、輪廓匯出腳本）；本打包腳本見 scripts/su_strat_audit_pack.py",
                 "scripts": ["scripts/su_strat_select.py", "scripts/su_strat_panels.py", "scripts/su_outline_offset_check.py", "scripts/su_strat_export_outlines.py", "scripts/su_strat_audit_pack.py"]},
        "su_version": meta["params"]["su_version"], "su_params": {k: meta["params"].get(k) for k in ("stream_ha", "min_su_ha", "slope_min_deg", "dtm", "built")},
        "design": {"strata": 6, "per_stratum": 4, "total": len(ids), "selection_seed": 20261005, "panel_order_seed": 20261006, "min_separation_m": 1200, "min_effective_area_ha": 3.0,
                   "note": "層別定義見 scripts/su_strat_select.py 文件字串；各層候選名單與編號→層別對照不公開（見 withheld）。"},
        "rater": {"n": 1, "who": "專案成員（人工，單一判讀者）", "blind_to": ["層別", "事件目錄歸因", "照片", "系統推薦"], "saw": ["Wayback 較早與最新影像", "Sentinel-2 2024-04-04 與 2025-03-25", "有效坡面輪廓", "DTM 參考欄"],
                  "options": {"A": "近期有新增／擴大崩塌", "B": "有崩塌但近期無變化", "C": "無崩塌", "D": "有裸露但非崩塌", "E": "無法判讀"},
                  "provenance_note": "判讀結果由使用者以 review.html 匯出 su_strat_human.json 後貼入對話，由助理原樣存檔；原始匯出檔的時間戳記無法取得。"},
        "limits": ["每層 4 個，不可估計比率或準確率", "單一判讀者", "Wayback 影像對 DTM／Sentinel-2 有 10–25 m 偏移（offset_check.json）", "S5 為代理層（無治理資料）", "近期＝2024-04～2025-03"],
        "hash_note": "SHA-256 以 LF 位元組計算；.gitattributes 已將本資料夾設為 -text，避免換行轉換改變內容。",
        "files": {f: sha(DST / f) for f in ("review.html", "su_strat_human.json", "offset_check.json")},
        "inputs_sha256": data_files,
        "withheld_sha256_commitment": {**key_files, "reveal": "專業人員盲判完成後公開原檔，雜湊可供核對"},
    }
    wlf(DST / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    for p in sorted(DST.iterdir()):
        print(p.name, round(p.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
