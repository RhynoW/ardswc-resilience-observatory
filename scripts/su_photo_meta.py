"""取 POC 內歷史照片的描述／備註（水保署歷史影像平台 API）並分類，補進 su_evidence.json。
API：https://photo.ardswc.gov.tw/api/v1/rest/dataset/metadata/<PhotoType>?page=N（每頁 1000 筆；PhotoType 0＝災害事件、8＝媒體報導）。
只保留 su_photos.json 中的 EventID；原始頁面不存檔。
分類（規則式，依描述＋備註＋災害名稱關鍵字，先符合者優先；僅是初步分類，非判定）：
  落石／滾石 → 坡面崩塌／坍方／滑動／土石崩落 → 土石流／土砂淤積 → 溪流沖刷／河岸 → 道路邊坡／路基／擋土設施 → 淹水 → 其他／無描述
輸出：poc/su_photo_meta.json（逐照片：描述摘要、災害名稱、拍攝日、分類）；更新 poc/su_evidence.json 的 field_photos.classes。
限制：關鍵字分類粗糙（一句描述常含多種現象，只取第一個符合的類別）；描述由拍攝者填寫、品質不一；沒有尺度；照片座標精度未驗證。
用法：python scripts/su_photo_meta.py
"""
import json
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
API = "https://photo.ardswc.gov.tw/api/v1/rest/dataset/metadata"
RULES = [("落石／滾石", r"落石|滾石|墜石|石塊.{0,4}(滾|落)"),
         ("坡面崩塌／坍方", r"崩塌|坍方|崩落|滑動|滑坡|山崩|邊坡.{0,3}(滑|崩|坍)|土石崩|崩坍|岩屑崩|地滑|landslide"),
         ("土石流／土砂淤積", r"土石流|泥流|土砂|淤積|淤砂|砂石堆|淤泥|debris|DF\d{3}"),
         ("溪流沖刷／河岸", r"沖刷|沖毀|溪流|河岸|淘刷|河道|堤防|護岸|潰堤|RS\d{2,}"),
         ("道路邊坡／路基", r"路基|擋土|護欄|路面|道路|邊坡|橋|便道|坡面保護|護坡|駁坎|殘坡"),
         ("淹水", r"淹水|積水|水患|溢流|flood")]


def get(url):
    for _ in range(3):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return None


def classify(text):
    for name, pat in RULES:
        if re.search(pat, text, re.I):
            return name
    return "其他／無描述"


def main():
    ph = json.load(open(POC / "su_photos.json", encoding="utf-8"))
    want = {p["id"]: p for p in ph if p["type"] in ("災害事件", "媒體報導")}
    print("要取描述的照片", len(want))
    meta = {}
    cache = Path(os.environ.get("PHOTO_PAGE_CACHE", str(REPO / "data" / "ardswc_meta_cache" / "pages")))
    cache.mkdir(parents=True, exist_ok=True)

    def page(t, pg):
        f = cache / f"{t}_{pg}.json"
        if f.exists():
            return json.load(open(f, encoding="utf-8"))
        for _ in range(4):
            try:
                body = urllib.request.urlopen(urllib.request.Request(f"{API}/{t}?page={pg}", headers={"User-Agent": "Mozilla/5.0"}), timeout=90).read()
                f.write_bytes(body)
                return json.loads(body.decode("utf-8"))
            except Exception:  # noqa: BLE001
                time.sleep(3)
        raise RuntimeError(f"取頁失敗 type {t} page {pg}")
    for t in ("0", "8"):
        pg, n = 1, 0
        while pg < 120:
            r = page(t, pg)
            n += len(r)
            for x in r:
                if x.get("EventID") in want:
                    txt = " ".join(str(x.get(k) or "") for k in ("Description", "Note", "DisasterName"))
                    meta[x["EventID"]] = {"desc": (x.get("Description") or "")[:120], "name": x.get("DisasterName"), "date": (x.get("PhotoDate") or "")[:10], "year": x.get("DisasterYear"), "class": classify(txt)}
            if len(r) < 1000:
                break
            pg += 1
        print("PhotoType", t, "頁數", pg, "列數", n, "累計命中", len(meta), flush=True)
    (POC / "su_photo_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    cls = Counter(m["class"] for m in meta.values())
    print("取得描述", len(meta), "/", len(want), dict(cls))
    ev = json.load(open(POC / "su_evidence.json", encoding="utf-8"))
    by = defaultdict(Counter)
    for p in ph:
        if p["su_id"] and p["id"] in meta:
            by[p["su_id"]][meta[p["id"]]["class"]] += 1
    for e in ev:
        if e["su_id"] in by:
            e["field_photos"]["classes"] = dict(by[e["su_id"]])
    (POC / "su_evidence.json").write_text(json.dumps(ev, ensure_ascii=False), encoding="utf-8")
    # 只有照片、衛星無裸露的 SU：類別分布
    only = Counter()
    for e in ev:
        if e["satellite"]["category"] == "無崩塌" and "classes" in e["field_photos"]:
            for k, v in e["field_photos"]["classes"].items():
                only[k] += v
    print("只有照片（衛星無裸露）的 SU 的照片類別（筆數）：", dict(only))
    # 有衛星裸露且有照片
    both = Counter()
    for e in ev:
        if e["satellite"]["category"] != "無崩塌" and "classes" in e["field_photos"]:
            for k, v in e["field_photos"]["classes"].items():
                both[k] += v
    print("衛星有裸露且有照片的 SU 的照片類別（筆數）：", dict(both))
    
    print("2021 年後各類：", dict(Counter(m["class"] for m in meta.values() if m["year"] and str(m["year"]) >= "2021")))


if __name__ == "__main__":
    main()
