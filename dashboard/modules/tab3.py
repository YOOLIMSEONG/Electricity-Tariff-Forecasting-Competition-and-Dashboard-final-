# dashboard/modules/tab4.py
import streamlit as st
from .common import card, placeholder, section_header

def render(title: str = "부록"):
    section_header(title)

    a,b,c = st.columns(3, gap="large")
    with a: 
        with card("Top - Card 1"): placeholder(150)
    with b: 
        with card("Top - Card 2"): placeholder(150)
    with c: 
        with card("Top - Card 3"): placeholder(150)

    x,y = st.columns([2,1], gap="large")
    with x: 
        with card("Middle - Card 1"): placeholder(300)
    with y: 
        with card("Middle - Card 2"): placeholder(300)

    with card("Bottom - Large Card"): 
        placeholder(260)
