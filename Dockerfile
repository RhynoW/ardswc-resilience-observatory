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
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser

EXPOSE 7860
CMD ["python", "webapp/change_detect_viewer/app.py"]
