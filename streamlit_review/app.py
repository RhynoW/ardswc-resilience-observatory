# -*- coding: utf-8 -*-
"""
人工覆核工作單（Streamlit 版）：把 review/review_queue.html 嵌進 Streamlit 頁面。

  * HTML 由 scripts/review_queue.py 產生，內含 A 級候選熱點、前後期 Sentinel-2 影像面板、
    NLSC 1/5000 相片基本圖＋Esri Wayback 年度滑桿、通用版正射影像（混合）與變遷候選綠框。
  * 面板影像在原 HTML 以相對路徑 ../data/ge_captures/ 引用；此處改寫成 GitHub raw 位址
    （需先把 data/ge_captures 推上 GitHub 公開 repo）。
  * 判讀只存在使用者瀏覽器的 localStorage，需自行「匯出 ledger.json」；本頁不會寫回伺服器。
部署：Streamlit Community Cloud → New app → 此 repo / 分支 master / 主檔 streamlit_review/app.py。
本機測試：streamlit run streamlit_review/app.py
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

REPO = Path(__file__).resolve().parent.parent
HTML = REPO / "review" / "review_queue.html"
RAW = "https://raw.githubusercontent.com/RhynoW/ardswc-resilience-observatory/master/"

st.set_page_config(page_title="人工覆核工作單", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:0}</style>", unsafe_allow_html=True)

if not HTML.exists():
    st.error("找不到 review/review_queue.html，請先執行 python scripts/review_queue.py。")
    st.stop()

page = HTML.read_text(encoding="utf-8").replace("../data/ge_captures/", RAW + "data/ge_captures/")
components.html(page, height=1800, scrolling=True)
st.caption("判讀只保存在你的瀏覽器；請定期「匯出 ledger.json」。自動訊號（SSIM、綠框）僅為影像變化候選訊號，需人工確認。")
