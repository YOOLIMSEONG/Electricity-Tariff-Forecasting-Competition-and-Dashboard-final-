# dashboard/modules/tab1.py
import streamlit as st
from .common import card, placeholder, section_header   # ← 같은 폴더이므로 단일 점

def render(title: str = "실시간 데이터 확인"):
    section_header(title)

    t1, t2, t3 = st.columns(3, gap="large")
    with t1: 
        with card("Top - Card 1"): placeholder(150)
    with t2: 
        with card("Top - Card 2"): placeholder(150)
    with t3: 
        with card("Top - Card 3"): placeholder(150)

    m1, m2 = st.columns([2,1], gap="large")
    with m1:
        with card("Middle - Card 1"): placeholder(300)
    with m2:
        with card("Middle - Card 2"): placeholder(300)

    with card("Bottom - Large Card"):
        placeholder(260)
