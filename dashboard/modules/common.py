# dashboard/modules/common.py
from contextlib import contextmanager
import streamlit as st

# ===== 색상 토큰 (메인/사이드바 원래 톤으로 복원) =====
COLORS = {
    "bg":   "#F7FAFC",   # 메인 배경
    "bg2":  "#FFFFFF",   # 사이드바 배경
    "text": "#1A202C",
    "primary": "#6366F1",
    "border":  "#E2E8F0",
}

def inject_css():
    st.markdown(f"""
    <style>
    :root {{
      --bg: {COLORS['bg']}; --bg2: {COLORS['bg2']}; --text: {COLORS['text']};
      --primary: {COLORS['primary']}; --border: {COLORS['border']};
    }}
    html, body, [data-testid="stAppViewContainer"] {{
      background: var(--bg); color: var(--text);
      font-family: 'Space Grotesk', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto;
      font-weight: 300; font-size: 14px;
    }}
    /* 상단 툴바 색 통일 */
    [data-testid="stHeader"], [data-testid="stHeader"] > div {{ background: var(--bg); }}

    /* ===== 사이드바 ===== */
    [data-testid="stSidebar"] {{
      background: var(--bg2);
      border-right: 1px solid var(--border);
    }}
    /* 사이드바의 텍스트 */
    [data-testid="stSidebar"] * {{ color: #1A202C !important; }}
    /* 라디오의 기본 원 숨김 + 링크형 */
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
      display: none !important;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
      display:block; padding:8px 10px; border-radius:10px;
      opacity:.70; font-weight:300; cursor:pointer;
      border:1px solid transparent; background: transparent;
      text-decoration:none !important;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
      background: rgba(99,102,241,.18);
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label.sb-active {{
      opacity:1; font-weight:700;
      background: rgba(99,102,241,.18); border: 1px solid var(--primary);
    }}

    /* ===== 카드 공통 ===== */
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,1), rgba(255,255,255,0.92));
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 4px 12px rgba(26,32,44,0.08);
      padding: 14px 16px;
      margin-bottom: 18px;
    }}
    .card-title {{
      font-weight: 600; opacity:.85; margin-bottom: 10px; color: var(--text);
    }}
    /* 점선 → 실선 */
    .placeholder {{
      border:1px solid rgba(148,163,184,.45);
      border-radius:12px; width:100%;
      background: rgba(255,255,255,0.8);
    }}
    </style>
    """, unsafe_allow_html=True)

@contextmanager
def card(title: str):
    st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown('</div>', unsafe_allow_html=True)

def placeholder(height: int = 160):
    st.markdown(f'<div class="placeholder" style="height:{height}px"></div>', unsafe_allow_html=True)

def section_header(title: str, subtitle: str = "스타일 전용 빈 카드 레이아웃"):
    st.markdown(f"### {title}")
    st.caption(subtitle)
