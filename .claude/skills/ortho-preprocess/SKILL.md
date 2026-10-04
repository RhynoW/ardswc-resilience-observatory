---
name: ortho-preprocess
description: Wayback／航照／UAV 影像做前後期變遷分析前的「正射與對位前處理」：先診斷兩期相對錯位（依坡度），再依結果決定可否做像素級變遷、是否對位、或改用其他影像；對位須通過 hold-out 檢核。山區（如花蓮）使用 Wayback 或航拍做變遷偵測、或使用者提到正射、對位、配準、registration 時使用。
---

# Wayback／航照影像正射與對位前處理

## 為什麼需要（本專案的查核結果，2026-10-04）
- Wayback（Esri Vivid／Maxar 等）是**已正射但精度有限**的鑲嵌影像：metadata 的 `SRC_ACC` 標稱水平精度 **5 m（2018 年以前常為 8.47 m）**；約 1.2 m/px（z17）時等於 4–7 像素，且**沒有拍攝角度與 RPC**，無法自行重新正射。
- 實測兩期相對錯位（區塊相位相關）：高雄 2022-08-26 vs 2023-11-02，**中位 1.7 m（坡度 <10°）→ 4.2 m（>40°）、p90 5.5→10.7 m**，62% 的區塊 >2 像素；花蓮陡坡（38–55°，5 個點、10 個影像對）：**只有 1 對**（WV03 2021-10-29 vs WV02 2022-11-13）達 A（中位 0.8 m、p90 2.1 m）；其餘 9 對中位 0.3–5.6 m、**p90 7–15 m（≥6 像素，屬 C）**，連 2023-01-14 vs 2023-02-27 這種間隔 6 週的同感測器影像對也是中位 4.3 m、p90 9.1 m；2017/2018 年（SRC_ACC 8.47 m）對近年影像的整體偏移另有 5–6.7 m。
- Sentinel-2（10 m）高雄兩期中位 2.3 m（0.23 px）、p90 5.7 m；花蓮乾淨配對（10-01 vs 10-16）中位 2.2 m；含雲日期的估計不可靠。
- **全局或平滑位移場事後對位在山區沒有預測力**：hold-out 區塊的殘差 RMSE（4.6–7.7 m）與修正前相當；修正後重估的位移下降是用同一批估計擬合的循環結果。（TPS、多項式次數、區塊大小與「錯位正比於高程」的地形位移模型都試過，hold-out 皆無改善；DTM 山陰影跨模態匹配也試過且失敗（一致性殘差 12–23 m，遠大於相對位移 5.5–6.3 m）；詳見 ARCHITECTURE_v3.md 附六補。）所以本 skill 的 `wayback` 對位指令有 hold-out 門檻，**未通過就不輸出修正影像**。

## 流程（先診斷，再決定）
1. **診斷**（約 30 秒／點）：
   `python .claude/skills/ortho-preprocess/scripts/ortho_register.py diagnose --lat <緯度> --lon <經度> --ref-date YYYYMMDD --mov-date YYYYMMDD`
   在**陡坡**與**平地**各取幾個點（至少 5 個陡坡點）；日期用 `wayback_assist.list_frames` 看該點有哪些期（腳本找不到日期會列出可用日期）。
2. **依結論決定用法**：
   - **A（p90 ≤ 2 px 且中位 ≤ 1 px）**：可做像素級變遷偵測。
   - **B（p90 ≤ 5 px）**：只做**物件級**變遷（崩塌塊 ≥0.5 ha），並把 p90 當邊界緩衝；不要用像素差異、不要評估 <0.5 ha 的小崩塌。
   - **C（p90 > 5 px）**：不做像素級。改用已配準影像（Sentinel-2）做變遷；Wayback／航照只做人工目視佐證；若要像素級，須取得原始影像（含 RPC）＋DEM 重新正射（`gdalwarp -rpc -to RPC_DEM=…`），本專案目前沒有這類資料。
3. **選期別**：不要預設同年度或同感測器就會對得上（花蓮 10 對中只有 1 對為 A）；逐對診斷後再選。避免 2018 年以前（`SRC_ACC` 8.47 m）的影像與近年影像配對。
4. **對位（僅 B 且需要時）**：`ortho_register.py wayback --lat … --lon … --ref-date … --mov-date … --out <dir>`（或 `files` 子命令吃兩張同尺寸影像）。看 `qc.json`：`holdout.rmse_m ≤ 1.5 個像素`、`passed: true` 才可用 `mov_registered.png`；`field.npz` 是位移場（像素）。**不通過就不要用**，保持原影像並在成果中標註錯位緩衝。
5. **記錄**：把診斷結果（median/p90、`SRC_ACC`、感測器）與是否對位寫進成果，供下游決定最小可偵測面積。成果敘述不可寫「已正射對位」除非 QC 通過。

## UAV／航拍（BigGIS 正射圖磚、舊航照）
本專案已有的做法在 `scripts/uav_register/`：
- `register.py`：SIFT + RANSAC 單應矩陣，對**兩個獨立底圖**（NLSC PHOTO、Esri）各對位一次互相驗證。
- `loftr_match.py`／`finalize.py`：學習式匹配（kornia LoFTR）：降到底圖解析度 + 灰階 + CLAHE → 多旋轉 × 多尺度 × 分塊 → MAGSAC 剔除粗差 → 合併單應矩陣 → **留一檢核 RMSE** → 糾正輸出。GPU 保護見 `thermal_guard.py`。
- `scripts/uav_synth.py`＋`run_synth.py`：斜拍影像用 **DTM 合成斜視圖**再匹配（粗對位失敗的候選）。
- `check.py`：糾正結果疊回底圖目視檢查。
原則同上：以 hold-out RMSE 為通過條件；BigGIS UAV 圖磚雖稱正射，仍須用 `diagnose` 風格的方法（與 Wayback 或 Sentinel-2 比對）量測，山區未驗證前不當作與其他期可像素級對位。

## 限制與誠實原則
- 相位相關需要紋理；雲、水體、植生季節變化、大面積真變化會使區塊失敗或偏向 0，診斷只是**下限估計**。
- 診斷用 128 px 區塊（約 140 m）；更小尺度的錯位（如 <50 m 的局部地形位移）看不到。
- 不要把「診斷為 A」當成現地精度；它只表示兩期相對一致。絕對位置精度仍以 `SRC_ACC`（5 m 以上）為上限。
- 結果引用：`data/biggis_interp/areas/registration_check.json`（查核數據）、`ARCHITECTURE_v3.md` 附六。
