# dashboard/app.py
import streamlit as st
from modules import tab1, tab2, tab3
from modules.common import inject_css   # ← modules 경로에서 import

st.set_page_config(page_title="대시보드", page_icon="📊", layout="wide")

# 스타일 먼저 주입
inject_css()

PAGES = {
    "실시간 데이터 확인": tab1.render,
    "과거 데이터 분석":   tab2.render,
    "부록":             tab3.render,
}
TAB_NAMES = list(PAGES.keys())

if "tab" not in st.session_state:
    st.session_state.tab = TAB_NAMES[0]

# URL query ↔ 상태 동기화 (리로드 없이)
qp = st.query_params.get("tab", None)
if qp in TAB_NAMES and st.session_state.tab != qp:
    st.session_state.tab = qp

with st.sidebar:
    st.markdown("#### 대시보드")
    st.radio(
        "탭 선택",
        options=TAB_NAMES,
        index=TAB_NAMES.index(st.session_state.tab),
        key="tab",
        label_visibility="collapsed",
    )
    # 활성 라벨 강조
    st.markdown(f"""
    <script>
      (function(){{
        const want = {repr(st.session_state.tab)};
        const group = document.querySelector('[data-testid="stSidebar"] [role="radiogroup"]');
        if(!group) return;
        group.querySelectorAll('label').forEach(lbl => {{
          lbl.classList.remove('sb-active');
          const txt = (lbl.innerText || '').trim();
          if(txt === want) lbl.classList.add('sb-active');
        }});
      }})();
    </script>
    """, unsafe_allow_html=True)

# 본문 렌더
PAGES[st.session_state.tab](st.session_state.tab)
