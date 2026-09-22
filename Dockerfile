# 3.11-slim（不帶明確 codename）會隨 Debian 版本推移滾動——2026-09-05 實測已滾到 trixie，
# 導致 Playwright 的 `--with-deps` 字型套件安裝失敗（trixie 沒有 ttf-unifont/ttf-ubuntu-
# font-family 這兩個套件名）。明確釘住 bookworm（Debian 12，Playwright 官方支援的版本）。
FROM python:3.11-slim-bookworm

# OpenCV runtime libs (headless build still needs libgl/libglib at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# UAV 影像自動對位（scripts/uav_register，LoFTR + MAGSAC）需要 torch + kornia。
# TORCH_VARIANT 建置參數決定 torch wheel 來源（預設 cpu，不開 GPU 時行為與之前完全相同）：
#   cpu    → https://download.pytorch.org/whl/cpu   （HF cpu-basic；預設 PyPI 版會連帶下載數 GB 的 CUDA 套件）
#   cu128  → https://download.pytorch.org/whl/cu128 （HF GPU 硬體；wheel 自帶 CUDA 執行庫，驅動由主機提供，映像多數 GB）
# HF Docker Space 會把 Space 的 Variables 當建置參數傳入：Settings → Variables 新增 TORCH_VARIANT=cu128 後重建。
# 版本與本機驗證版一致。並預先下載 LoFTR 權重到 TORCH_HOME（切到 appuser 後才讀得到；執行期不必再連外）。
# 非致命：失敗只影響對位腳本、不影響網站；輸出存 /app/uav_deps.log，/api/health 在未安裝時附上尾段。
ARG TORCH_VARIANT=cpu
ENV TORCH_HOME=/app/.torch \
    TORCH_VARIANT=${TORCH_VARIANT}
RUN (pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
     && pip install --no-cache-dir kornia==0.8.3 \
     && python -c "import kornia.feature as KF; KF.LoFTR(pretrained='outdoor'); print('LoFTR weights OK, torch variant: ${TORCH_VARIANT}')") \
    > /app/uav_deps.log 2>&1 \
    || echo "WARN: UAV deps (torch/kornia, ${TORCH_VARIANT}) install failed; registration scripts unavailable"; \
    tail -n 3 /app/uav_deps.log || true

# 「線上分析（自訂座標）」用 Playwright 內建 Chromium（headless=True，見
# scripts/ge_web_capture_v2.py 的 --headless-container 分支）驅動 GE Web——2026-09-05 已在
# HF cpu-basic（2 vCPU、無 GPU、software WebGL）實測驗證可行。PLAYWRIGHT_BROWSERS_PATH 設在
# root 之外的共用位置，之後切到非 root 的 appuser 才讀得到（Playwright 預設裝到目前使用者的
# home cache，若在切換使用者前裝在 root home 下，appuser 執行期會找不到瀏覽器）。
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install --with-deps chromium

COPY . .

# 全臺灣 20 m DTM（內政部地政司，政府資料開放授權條款第1版）：建置時從 TGOS 下載（269 MB，
# 續傳）並轉成 224 MB 的分塊壓縮 GeoTIFF，不放進 git。失敗不擋建置——執行期找不到 DTM 檔時
# 等高線/流域分析自動退回 Cesium World Terrain（需 CESIUM_ION_TOKEN），兩者皆無則該功能隱藏/回明確錯誤。
# 輸出存成 /app/data/dtm_prepare.log：建置日誌需登入才看得到，/api/health 在 DTM 不可用時會附上此檔尾段。
RUN mkdir -p /app/data && (python scripts/prepare_dtm20.py > /app/data/dtm_prepare.log 2>&1 \
    || echo "WARN: DTM prepare failed; falling back to Cesium terrain"); \
    tail -n 5 /app/data/dtm_prepare.log || true

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data/cesium_cache /app/data/contour_cache \
    && chown -R appuser:appuser /app /ms-playwright
USER appuser

# CESIUM_ION_TOKEN 為選配 Space secret：沒設定時等高線疊圖開關自動不顯示（fail-closed），
# 其餘功能（100 熱點清單/地圖/巡查優先級/11 個熱點的完整比對面板）不受影響。
#
# GMAPS_DEMO_APPLE_AUTO=0：gmaps_tiles.py 內建的 Apple 底圖 token 自動換發（重放 DuckDuckGo
# 載入 MapKit JS 的兩段請求）預設開啟，該檔案自己的註解也寫明這個預設值是假設「只在使用者
# 本機執行」；部署到公開 HF Space 後這個假設不成立——實測從這個容器連 duckduckgo.com /
# cdn.apple-mapkit.com 每次都逾時（TIMEOUT=15s），導致快取永遠沒建立成功，
# 使得每一次首頁請求／`/api/apple_status` 都要在 `apple_is_available()` 卡住整整 15–16 秒
# 才有回應——這正是使用者回報「歷史影像無法顯示」的根因（頁面卡在載入、看起來像壞掉）。
# 關掉後 `apple_is_available()` 直接短路回 False，首頁與各 API 恢復到 <1 秒；前端本來就會
# 在 `/api/apple_status` 回報未設定時自動隱藏 Apple 底圖選項（fail-closed，不影響其他 6 種
# 底圖／等高線／100 熱點清單／11 熱點完整比對面板）。同樣的取捨 gmaps_bbox_demo 早已採用，
# 這裡只是把這份複製過來卻忘記同步調整預設值的地方補上。
# ENABLE_LIVE_CAPTURE / GE_CAPTURE_CONTAINER_MODE（2026-09-05，取代原先的「公開版做不到」
# 結論）：起初認為即時 GE Web 擷取需要真實瀏覽器自動化、這個容器沒有瀏覽器裝不了，故先前
# 直接關閉。實測推翻此結論——見 ARCHITECTURE.md「線上分析」章節與
# scripts/ge_web_capture_v2.py 的 --headless-container 說明。GE_CAPTURE_CONTAINER_MODE=1
# 讓 `/api/capture_custom` 改呼叫 headless Chromium 路徑（1920×1080 viewport，實測在此硬體
# 最穩定）而非本機用的 8K/headed/真 Chrome 路徑。ENABLE_LIVE_CAPTURE 保留當緊急停用開關
# （例如濫用或資源異常時手動設回 0），非長期預設關閉。
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    GMAPS_DEMO_APPLE_AUTO=0 \
    ENABLE_LIVE_CAPTURE=1 \
    GE_CAPTURE_CONTAINER_MODE=1

EXPOSE 7860
CMD ["python", "webapp/change_detect_viewer/app.py"]
