# dashboard/modules/tab3.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def render(title: str = "부록"):
    """부록:전처리, 모델링 요약"""
    
    st.markdown(f"### {title}")
    st.markdown("전처리, 모델링 프로세스 요약")
    
    # 탭 구분
    prep_tab, model_tab = st.tabs(["전처리", "모델링"])
    
    # ===== 전처리 탭 =====
    with prep_tab:
        st.markdown("#### 1. 파생변수 생성")
        
        prep_col1, prep_col2 = st.columns(2, gap="large")
        
        with prep_col1:
            st.markdown("**시간 관련 파생변수**")
            time_features = [
                "• `month`: 월 (1-12)",
                "• `day`: 일 (1-31)",
                "• `hour`: 시간 (0-23)",
                "• `weekday`: 요일 (0=월, 6=일)",
                "• `is_weekend`: 주말 여부 (0/1)",
                "• `is_holiday`: 공휴일 여부 (0/1)",
                "• `is_special_day`: 특수한 날(공휴일,명절,대체휴일 포함) (0/1)",
                "• `season`: 계절 (봄/여름/가을/겨울)"
            ]
            for feat in time_features:
                st.caption(feat)
        
        
        st.markdown("#### 2. 데이터 전처리 프로세스")
        
        prep_steps = [
            ("1️⃣ 연도 변경", "2024년 > 2018년"),
            ("2️⃣ train/valid 분리", "1월~10월 train / 11월 valid"),
            ("3️⃣ 전력사용량이 0에 가까울 때 단가가 비정상적으로 커짐", "eps로 분모 무한0 방지 + 단가 이상치 클리핑(극단값 제거) 적용"),
            ("4️⃣ Teacher vs Student 피처분리", "데이터 누수 방지")
        ]
        
        for step, desc in prep_steps:
            with st.container():
                st.markdown(f"**{step}**")
                st.caption(desc)
        
        st.markdown("#### 3. Teacher vs Student 피처 분리")
        
        feature_col1, feature_col2 = st.columns(2, gap="large")
        
        with feature_col1:
            st.markdown("**Teacher 모델 (학습용)**")
            st.caption("전체 가능한 피처 사용")
            teacher_features = [
                "• 전력사용량(kWh)",
                "• 무효전력량 (지상/진상)",
                "• 역률 (지상/진상)",
                "• 탄소배출량(tCO2)",
                "• 모든 시간 변수",
            ]
            for feat in teacher_features:
                st.caption(feat)
        
        with feature_col2:
            st.markdown("**Student 모델 (예측용)**")
            st.caption("실시간으로 얻을 수 있는 피처만 사용")
            student_features = [
                "• month, day, hour, minute",
                "• weekday, is_weekend",
                "• 작업유형",
                "※ 전력/역률 변수 제외",
            ]
            for feat in student_features:
                st.caption(feat)
    
    # ===== 모델링 탭 =====
    with model_tab:
        st.markdown("#### 1. 모델 구조")
        
        model_col1, model_col2 = st.columns(2, gap="large")
        
        with model_col1:
            st.markdown("**Teacher 모델**")
            teacher_info = [
                ("알고리즘", "HistGradientBoostingRegressor"),
                ("최대 반복", "800회"),
                ("조기 종료", "검증 손실 기준"),
                ("목적", "최고 정확도 추구"),
            ]
            for key, val in teacher_info:
                st.caption(f"• {key}: {val}")
        
        with model_col2:
            st.markdown("**Student 모델**")
            student_info = [
                ("알고리즘", "HistGradientBoostingRegressor"),
                ("하이퍼파라미터", "RandomizedSearchCV"),
                ("튜닝 반복", "60회"),
                ("목적", "빠른 학습,예측 속도"),
            ]
            for key, val in student_info:
                st.caption(f"• {key}: {val}")
        
        st.markdown("#### 2. 지식 증류 (Teacher > Student 학습 이전)")
        
        # 1️⃣ Teacher OOF 예측
        col_step1_label, col_step1_help = st.columns([0.1,0.9], gap="small")
        with col_step1_label:
            st.markdown("**1️⃣ Teacher OOF 예측**")
        with col_step1_help:
            with st.popover("❓", use_container_width=False):
                st.markdown("**Teacher OOF 예측**")
                st.write("")
                st.markdown("• Train set을 시간 순서대로 여러 구간으로 나눔")
                st.markdown("• 각 fold에서 미래 데이터(검증 구간)에 대한 Teacher 예측값을 저장")
                st.markdown("• 이렇게 하면 데이터 누수 없이 전체 기간에 대해 Teacher의 예측값(oof_pred)을 확보할 수 있음")
        st.caption("TimeSeriesSplit으로 시계열 데이터 학습 → 각 fold의 검증 세트에서 예측값 생성")
        
        st.markdown("")  # 간격
        
        # 2️⃣ Alpha 최적화
        col_step2_label, col_step2_help = st.columns([0.1,0.9], gap="small")
        with col_step2_label:
            st.markdown("**2️⃣ Alpha 최적화**")
        with col_step2_help:
            with st.popover("❓", use_container_width=False):
                st.markdown("**Alpha 최적화**")
                st.write("")
                st.markdown("• **α(알파)**는 Teacher 예측을 얼마나 신뢰할지 결정하는 비율")
                st.markdown("• α가 크면 Teacher 예측을 많이 반영 → Teacher 지식을 더 많이 계승")
                st.markdown("• α가 작으면 원래 타깃 데이터에 더 충실")
        st.caption("y_blend = (1-α)×y_true + α×y_teacher 형태로 증류 타깃 생성")
        st.caption("기준: 마지막 fold의 홀드아웃 세트에서 MSE 최소화")
        
        st.markdown("")  # 간격
        
        # 3️⃣ Student 학습
        col_step3_label, col_step3_help = st.columns([0.1,0.9], gap="small")
        with col_step3_label:
            st.markdown("**3️⃣ Student 학습**")
        with col_step3_help:
            with st.popover("❓", use_container_width=False):
                st.markdown("**Student 학습**")
                st.write("")
                st.markdown("• Teacher가 파악한 시간대별 / 공휴일 / 유형별 패턴을 따라 배우고")
                st.markdown("• 피처는 단순함 (즉, 계산량 ↓)")
                st.markdown("• 하지만 정확도는 단순 모델 대비 훨씬 ↑")
        st.caption("Student는 단순 피처로 y_blend를 학습 → 배포 환경에서도 높은 정확도 유지")
        
        st.markdown("#### 3. 튜닝 & 평가")
        
        valid_col1, valid_col2 = st.columns(2, gap="large")
        
        with valid_col1:
            st.markdown("**최적파라미터**")
            st.caption("""
            • RandomizedSearchCV 튜닝반복 60회
            """)
        
        with valid_col2:
            st.markdown("**성능 평가**")
            eval_metrics = [
                "• Teacher 모델 성능 (기준점) = [Teacher] OOF 오차: 8,468,927",
                "• Student 모델 학습 = [Student] Best CV 오차 (distilled target): 4,509,263",
                "• Student 모델 최종 성적 = [Student] Holdout 오차 : 5,662,961",
                "더 적은 변수를 사용한 Student 모델이 모든 변수를 사용한 Teacher 모델보다 더 우수한 성능(낮은 오차)을 달성"
            ]
            for metric in eval_metrics:
                st.caption(metric)
        
        st.markdown("#### 4. 모델예측결과")
        
        left_col, right_col = st.columns([0.7, 0.3], gap="large")
        
        with left_col:
            st.markdown("12월의 전기요금 예측 결과파일")
        
        with right_col:
            # submission.csv 파일 읽기
            submission_path = Path(__file__).resolve().parent.parent / "data" / "submission.csv"
            submission_df = pd.read_csv(submission_path)
            
            # CSV를 바이너리로 변환
            csv_buffer = submission_df.to_csv(index=False, encoding='utf-8-sig')
            
            # 다운로드 버튼
            st.download_button(
                label="예측 결과 다운로드",
                data=csv_buffer,
                file_name="submission_예측결과.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
