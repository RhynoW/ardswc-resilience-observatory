FROM python:3.11-slim

# OpenCV runtime libs (headless build still needs libgl/libglib at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data/cesium_cache /app/data/contour_cache \
    && chown -R appuser:appuser /app
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
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    GMAPS_DEMO_APPLE_AUTO=0

EXPOSE 7860
CMD ["python", "webapp/change_detect_viewer/app.py"]
