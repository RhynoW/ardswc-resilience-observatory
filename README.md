---
title: 坡地韌性哨兵
emoji: 🏔️
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 地景複發熱點觀測站｜坡地韌性哨兵
### TASTI_SatDashboard・開源空間資訊於國家韌性應用黑客松 OSSInt 2026 決賽

把水保署既有的事件型崩塌目錄、歷史現地照片與衛星影像，整理成**每個坡面的證據鏈**，幫第一線巡查決定先看哪裡：

> **機器幫忙決定先看誰・證據鏈告訴你為什麼・最後由人決定要不要去**

線上系統：<https://rhynowu-ardswc-resilience-observatory.hf.space/>

## 五大主題

| 主題 | 一句話 | 代表數字 |
|---|---|---|
| 技術亮點 | 免費衛星＋開放資料，每個數字都量過 | 13,175 個坡面單元；大事件後紅邊偵測率 75%（花蓮）・43%（太魯閣），誤報約 2%；35 組無 GPS 的 UAV 照片自動對位 |
| 架構創新 | 證據鏈分開呈現，不給總分，最後由人決定 | 五類證據、三種候選層級（優先 411・一般 12,105・資料不足 214） |
| 韌性呈現 | 每個坡面一份履歷：壞過、在變、會碰到誰 | 太魯閣 330 個紅邊下降坡面中 229 個在地震當月；411 個優先坡面中 200 個附近有道路、67 個在土石流潛勢溪流集水區 |
| 對公部門的啟示 | 自動訊號只排「先看誰」，先量再用 | 平時零星紅邊下降只有約 25% 屬實（95% CI 5–45%） |
| 對全民的啟示 | 看得懂、查得到、一起補 | 不給風險分數；坡面頁列出附近道路、建物與潛勢溪流；OpenStreetMap 可一起補 |

## 網站導覽

| 頁面 | 內容 |
|---|---|
| [`/`](https://rhynowu-ardswc-resilience-observatory.hf.space/) | 首頁：五大主題 |
| [`/su`](https://rhynowu-ardswc-resilience-observatory.hf.space/su) | **坡面證據鏈**（高雄 POC）：每個坡面的①事件目錄與年度圖層 ②現地照片 ③Sentinel-2 紅邊時間序列 ④影像品質 ⑤崩了會碰到什麼（道路、聚落、土石流潛勢溪流），與三種候選層級。地圖可切 **3D 地形**（CesiumJS＋20 m DTM）與**全頁顯示**；底圖含 Esri Wayback、國土測繪中心 2014–2025 逐年航照；參考疊圖含地調所順向坡、歷史山崩、地質敏感區與國土測繪中心光達地形變異 |
| [`/poc`](https://rhynowu-ardswc-resilience-observatory.hf.space/poc) | 五個 POC 區域：高雄、花蓮（馬太鞍溪堰塞湖）、太魯閣（0403 地震）、中部、嘉義的衛星與事件目錄比對、坡面紅邊檢核 |
| [`/roadmap`](https://rhynowu-ardswc-resilience-observatory.hf.space/roadmap) | 系統藍圖對照：專家構想的六個模組 vs 現況、道路聚落暴露、各地層崩塌密度 |
| [`/static/uav_cesium.html`](https://rhynowu-ardswc-resilience-observatory.hf.space/static/uav_cesium.html) | UAV 自動套合 3D 檢視（已完成成果，依輔導建議停止擴充） |
| [`/system`](https://rhynowu-ardswc-resilience-observatory.hf.space/system) | 完整技術說明（原首頁：方法演進、驗證、參考資訊） |

## 方法摘要

- **坡面單元（SUv2）**：內政部 20 m DEM，集水區重疊法（正、反 DEM 子集水區疊合，河網門檻 20 ha，<2 ha 併入鄰居）；13,175 個單元，其中 12,730 個有坡度 ≥15° 的有效坡面。ID 帶版本（`SUv2-D20m-WO-S20-M2-E15s3-F0.2`）。
- **近期衛星變化**：Sentinel-2 L2A NDRE＝(B8A−B5)/(B8A+B5)，SCL 去雲、有效坡面中位數、扣同日全區中位數；「持續下降」＝低於基準 0.08 以上且 60 天後仍偏低。
- **候選層級（規則 v1）**：優先人工判讀＝近期照片描述崩塌（P1）、事件目錄新增或擴大（P2）、NDRE 持續下降（P3）任一；資料不足＝有效坡面太小或衛星觀測不足；其餘為一般判讀。**不給總分**，第⑤類暴露資訊不影響層級。
- **UAV 自動對位**：歷史影像平台的 UAV 照片沒有 EXIF／GPS；以 LoFTR 特徵匹配與 20 m DTM 合成斜視圖解算相機姿態，留出點 RMSE ≤10 m 才發布。

## 驗證（95% 信賴區間）

| 項目 | 結果 | 95% CI |
|---|---|---|
| 大事件後崩塌坡面被抓到（花蓮・太魯閣） | 73/97＝75%・169/395＝43% | 66–83%・38–48% |
| 沒有事件的坡面被判下降（誤報） | 1.9%・2.0% | 1.6–2.3%・1.6–2.4% |
| 高雄平時 NDRE 下降經分層盲判屬實 | 約 25%（分層加權，5/26） | 5–45% |
| 影像目視與官方年度圖層「有無裸露」一致 | 20/29＝69% | 51–83% |

方法參考 Olofsson et al. (2014)、Stehman (2014)；計算見 `scripts/accuracy_ci.py`。官方判釋是衛星判釋，不是現地確認。

## 資料來源

農業部農村發展及水土保持署 BigGIS（事件型崩塌目錄、全臺崩塌地圖層、土石流潛勢溪流）與歷史影像平台；內政部地政司 2025 年版 20 m DTM；國土測繪中心正射影像與空載光達地形變異分析圖；經濟部地質調查及礦業管理中心山崩地質資訊雲端服務平臺（參考疊圖）；Copernicus Sentinel-1／2（Microsoft Planetary Computer、Copernicus Data Space）；Esri World Imagery／Wayback；© OpenStreetMap contributors（ODbL）。各服務使用條款依其公告，網站頁尾標示出處。

## 主要程式

| 步驟 | 腳本 |
|---|---|
| 坡面單元 | `scripts/slope_units_v2.py`、`scripts/su_export_gpkg_v2.py` |
| Sentinel-2 紅邊時間序列 | `scripts/s2_ndre_series.py`（高雄）、`scripts/region_su_ndre.py`（花蓮、太魯閣） |
| 證據鏈與候選層級 | `scripts/su_pages_build.py` |
| 道路聚落、土石流潛勢溪流 | `scripts/su_osm_exposure.py`、`scripts/su_debris_link.py` |
| 地層崩塌密度 | `scripts/geology_landslide_density.py` |
| 驗證與信賴區間 | `scripts/ndre_validation.py`、`scripts/accuracy_ci.py` |
| UAV 自動對位 | `scripts/uav_synth.py`、`scripts/uav_register/` |
| 網站 | `webapp/change_detect_viewer/`（Flask＋Leaflet＋CesiumJS） |

架構、決策紀錄與各版本驗證細節見 [ARCHITECTURE_v3.md](ARCHITECTURE_v3.md)（附錄十五～二十一為決賽版內容）。

## 治理原則與限制

- 所有自動訊號只用來排出「先看誰」，不代表災害已確定或其嚴重程度；最後由人決定是否現勘。
- 平時零星的紅邊變化不可靠（約 25% 屬實）；坡面中位數約只看得到占坡面兩成以上的崩塌，小崩塌要靠事件目錄與現地照片。
- OpenStreetMap 在山區常缺漏，道路與建物數字是下限。
- Google Earth 內容只用於個案人工複核，不進批次偵測或訓練。
- 依輔導委員建議，3D 建模、UAV 擴充、自動套合與 250 m 網格熱點不再是主線；早期的熱點排序版本保留在 `/system` 作為方法演進紀錄。
