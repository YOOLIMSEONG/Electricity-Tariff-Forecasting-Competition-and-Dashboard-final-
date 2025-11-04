import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import time
import json
import os

# 스트리밍 데이터 저장 경로 (필요시)
# STREAMING_LOG_FILE = 'streaming_log.json'

# --- 데이터 로드 및 처리 함수 (기존과 동일) ---

def load_data():
    """데이터 로드 및 전처리"""
    # 이 예제에서는 CSV를 사용하지만, 실제 환경에 맞게 경로를 수정해야 합니다.
    # 파일이 없으면 빈 데이터프레임이나 샘플 데이터를 반환하도록 처리할 수 있습니다.
    try:
        # 1-11월 히스토리컬 데이터
        train_df = pd.read_csv('data/train.csv')
        train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])
    except FileNotFoundError:
        st.warning("data/train.csv 파일을 찾을 수 없습니다. 임시 데이터를 사용합니다.")
        train_df = pd.DataFrame({
            '측정일시': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00']),
            '전력사용량(kWh)': [100, 110],
            '전기요금(원)': [15000, 16500]
        })

    try:
        # 12월 테스트(스트리밍) 데이터
        test_df = pd.read_csv('data/test_streamling.csv')
        test_df['측정일시'] = pd.to_datetime(test_df['측정일시'])
    except FileNotFoundError:
        st.warning("data/test_streamling.csv 파일을 찾을 수 없습니다. 임시 데이터를 사용합니다.")
        # 12월 1일 하루치 임시 데이터 생성
        timestamps = pd.date_range(start='2024-12-01 00:00', end='2024-12-01 23:00', freq='h')
        test_df = pd.DataFrame({
            '측정일시': timestamps,
            '전기요금(원)': np.random.uniform(10000, 25000, len(timestamps))
        })

    return train_df, test_df

def calculate_baseline_metrics(train_df):
    """1-11월 데이터로 기준선 계산"""
    metrics = {}
    
    # 전력사용량 기준선 (95% 분위수)
    if '전력사용량(kWh)' in train_df.columns:
        metrics['power_baseline'] = train_df['전력사용량(kWh)'].quantile(0.95)
    elif '전기요금(원)' in train_df.columns:
        # 전력사용량 없을 시 전기요금으로 추정
        metrics['power_baseline'] = (train_df['전기요금(원)'].quantile(0.95) / 150) * 1.05 # 150원/kWh 가정
    else:
        metrics['power_baseline'] = 500 # 기본값

    # 월 최대값
    metrics['power_max'] = train_df['전력사용량(kWh)'].max() if '전력사용량(kWh)' in train_df.columns else metrics['power_baseline'] * 1.2
    
    return metrics

def generate_synthetic_December_data(test_df):
    """테스트 데이터 기반으로 전체 December 데이터 생성"""
    december_data = test_df.copy()
    
    # 누락된 컬럼 생성
    if '전력사용량(kWh)' not in december_data.columns:
        # 전기요금 기반으로 전력사용량 추정 (임의의 변동성 추가)
        base_usage = december_data['전기요금(원)'] / 150 # 150원/kWh 가정
        noise = np.random.normal(0, base_usage.std() * 0.1, len(december_data))
        december_data['전력사용량(kWh)'] = base_usage + noise
        december_data['전력사용량(kWh)'] = december_data['전력사용량(kWh)'].clip(lower=0) # 0 미만 값 제거
    
    if '지상무효전력량(kVarh)' not in december_data.columns:
        december_data['지상무효전력량(kVarh)'] = np.random.uniform(2, 5, len(december_data))
    
    if '진상무효전력량(kVarh)' not in december_data.columns:
        december_data['진상무효전력량(kVarh)'] = np.random.uniform(0, 1, len(december_data))
    
    if '탄소배출량(tCO2)' not in december_data.columns:
        # 0.0004 tCO2/kWh (임의의 배출 계수)
        december_data['탄소배출량(tCO2)'] = december_data['전력사용량(kWh)'] * 0.0004
    
    if '지상역률(%)' not in december_data.columns:
        # 85% ~ 95% 사이 값 생성 (일부는 90% 미만)
        december_data['지상역률(%)'] = np.random.uniform(85, 95, len(december_data))
    
    if '진상역률(%)' not in december_data.columns:
        # 93% ~ 100% 사이 값 생성 (일부는 95% 미만)
        december_data['진상역률(%)'] = np.random.uniform(93, 100, len(december_data))
    
    return december_data

def check_alerts(current_data, baseline_metrics):
    """경보 발생 체크"""
    alerts = []
    
    # 피크 기준선 초과 체크
    if current_data.get('전력사용량(kWh)', 0) > baseline_metrics.get('power_baseline', float('inf')):
        alerts.append({
            'type': '피크 초과',
            'value': f"{current_data['전력사용량(kWh)']:.2f} kWh",
            'timestamp': current_data['측정일시'],
            'severity': 'high'
        })
    
    # 지상역률 경보 (PF < 0.90)
    lag_pf = current_data.get('지상역률(%)', 100)
    if lag_pf < 90:
        alerts.append({
            'type': '지상역률 경보',
            'value': f"{lag_pf:.2f}%",
            'timestamp': current_data['측정일시'],
            'severity': 'medium'
        })
    
    # 진상역률 경보 (PF < 0.95)
    lead_pf = current_data.get('진상역률(%)', 100)
    if lead_pf < 95:
        alerts.append({
            'type': '진상역률 경보',
            'value': f"{lead_pf:.2f}%",
            'timestamp': current_data['측정일시'],
            'severity': 'medium'
        })
    
    return alerts

# --- 스트리밍 탭 렌더링 함수 (수정됨) ---

def render(tab_name):
    """Tab 1: 실시간 스트리밍 (2024년 12월)"""
    
    # --- 1. 데이터 로드 및 초기 설정 ---
    
    # 데이터 로드는 한 번만 수행하도록 st.cache_data 사용
    @st.cache_data
    def get_all_data():
        train_df, test_df = load_data()
        baseline_metrics = calculate_baseline_metrics(train_df)
        december_data = generate_synthetic_December_data(test_df)
        return baseline_metrics, december_data

    baseline_metrics, december_data = get_all_data()
    
    # Session state 초기화
    if 'streaming_active' not in st.session_state:
        st.session_state.streaming_active = False
    if 'streaming_data' not in st.session_state:
        st.session_state.streaming_data = []
    if 'streaming_index' not in st.session_state:
        st.session_state.streaming_index = 0

    
    # 헤더 영역 - 타이틀과 로고
    header_col1, header_col2 = st.columns([3, 1])

    with header_col1:
        st.markdown("# 현장 전력 모니터링 대시보드")

    with header_col2:
        try:
            st.image('ls_electric_logo.png', width=300)
        except:
            st.warning("로고 이미지를 찾을 수 없습니다.")
    
    
    # --- 2. 컨트롤 버튼 (왼쪽 정렬) ---
    st.markdown("#### 실시간 스트리밍")
    
    # 버튼을 왼쪽 정렬로 배치
    col1, col2, col3, col_spacer = st.columns([0.8, 0.8, 0.8, 4.6])
    
    with col1:
        start_btn = st.button("▶ 시작", use_container_width=True, key="btn_start")
    
    with col2:
        pause_btn = st.button("⏸ 일시정지", use_container_width=True, key="btn_pause")
    
    with col3:
        reset_btn = st.button("🔄 리셋", use_container_width=True, key="btn_reset")
    
    # 버튼 동작 처리 (rerun 없이)
    if start_btn:
        st.session_state.streaming_active = True
        if st.session_state.streaming_index >= len(december_data):
            st.session_state.streaming_data = []
            st.session_state.streaming_index = 0
    
    if pause_btn:
        st.session_state.streaming_active = False
    
    if reset_btn:
        st.session_state.streaming_active = False
        st.session_state.streaming_data = []
        st.session_state.streaming_index = 0
    
    # --- 3. 스트리밍 상태 및 로직 ---
    
    # 스트리밍 상태 표시는 먼저
    status_container = st.container()
    if st.session_state.streaming_active:
        status_text = f"🟢 **스트리밍 진행중** ({st.session_state.streaming_index}/{len(december_data)} 데이터 수집)"
    else:
        if st.session_state.streaming_index >= len(december_data) and len(st.session_state.streaming_data) > 0:
            status_text = "✅ **스트리밍 완료**"
        elif len(st.session_state.streaming_data) > 0:
             status_text = f"⏸ **스트리밍 일시정지** ({st.session_state.streaming_index}/{len(december_data)})"
        else:
             status_text = "🔴 **스트리밍 정지** (시작 대기중)"
    
    status_container.markdown(status_text)
    
    # 스트리밍 실행 (데이터 추가만)
    if st.session_state.streaming_active and st.session_state.streaming_index < len(december_data):
        # 한 번에 3개씩 수집 (시뮬레이션 속도 조절)
        batch_size = 3
        for i in range(batch_size):
            if st.session_state.streaming_index < len(december_data):
                current_row = december_data.iloc[st.session_state.streaming_index].to_dict()
                st.session_state.streaming_data.append(current_row)
                st.session_state.streaming_index += 1
        
        # 모든 데이터 수집 완료 시 상태 변경
        if st.session_state.streaming_index >= len(december_data):
            st.session_state.streaming_active = False
            st.balloons()
    
    # --- 4. 표시할 데이터 준비 ---
    
    MAX_CHART_POINTS = 100 # 최근 100개 데이터만 차트에 표시

    display_cols = ['측정일시', '전력사용량(kWh)', '전기요금(원)', '지상무효전력량(kVarh)', 
                    '진상무효전력량(kVarh)', '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)']

    if len(st.session_state.streaming_data) > 0:
        # 전체 데이터를 먼저 DF로 만듦 (집계용)
        all_display_data = pd.DataFrame(st.session_state.streaming_data)
        all_display_data['측정일시'] = pd.to_datetime(all_display_data['측정일시'])
        all_display_data.sort_values(by='측정일시', inplace=True)

        # 차트용 데이터는 마지막 N개만 슬라이싱
        start_index = max(0, len(all_display_data) - MAX_CHART_POINTS)
        chart_display_data = all_display_data.iloc[start_index:].copy()
        
    else:
        # 데이터가 없을 때 빈 DataFrame 생성
        chart_display_data = pd.DataFrame(columns=display_cols)
        chart_display_data['측정일시'] = pd.to_datetime(chart_display_data['측정일시'])
        all_display_data = chart_display_data.copy()

    # --- 5. 카드 스타일 지표 섹션 ---
    st.markdown("#### 🔴 실시간 모니터링")
    
    # 최신 데이터 포인트 계산
    if len(all_display_data) > 0:
        latest_data = all_display_data.iloc[-1]
        total_usage = all_display_data['전력사용량(kWh)'].sum()
        total_charge = all_display_data['전기요금(원)'].sum()
        total_co2 = all_display_data['탄소배출량(tCO2)'].sum()
        avg_unit_price = total_charge / total_usage if total_usage > 0 else 0
    else:
        latest_data = None
        total_usage = 0
        total_charge = 0
        total_co2 = 0
        avg_unit_price = 0
    
    # 카드 스타일 CSS (흰색 배경으로 통일)
    card_style = """
    <div style='
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    '>
        <div style='font-size: 0.9rem; color: #666;'>{title}</div>
        <div style='font-size: 1.8rem; font-weight: bold; color: #333;'>{value}</div>
        <div style='font-size: 0.85rem; color: #888;'>{unit}</div>
    </div>
    """
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            card_style.format(
                title="총 전력사용량(kWh)",
                value=f"{total_usage:,.0f}",
                unit="kWh"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            card_style.format(
                title="총 전기요금(원)",
                value=f"{total_charge:,.0f}",
                unit="원"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            card_style.format(
                title="총 탄소배출량(tCO2)",
                value=f"{total_co2:.2f}",
                unit="tCO2"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            card_style.format(
                title="평균 단가(원/kWh)",
                value=f"{avg_unit_price:.2f}",
                unit="원/kWh"
            ),
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 6. 실시간 그래프와 달력을 나란히 배치 ---
    graph_col, calendar_col = st.columns([1, 1])
    
    with graph_col:
        # 제목
        st.markdown("#### 실시간 전력사용량 & 전기요금 모니터링")
        
        # 실시간 스트리밍 그래프
        fig_realtime = go.Figure()
        
        # 전력사용량
        fig_realtime.add_trace(go.Scatter(
            x=chart_display_data['측정일시'],
            y=chart_display_data['전력사용량(kWh)'],
            name='전력사용량 (kWh)',
            mode='lines',
            line=dict(color='#1f77b4', width=2),
            yaxis='y1'
        ))
        
        # 기준선 (95% 분위수)
        if len(chart_display_data) > 0:
            fig_realtime.add_shape(
                type='line',
                x0=chart_display_data['측정일시'].iloc[0],
                x1=chart_display_data['측정일시'].iloc[-1],
                y0=baseline_metrics['power_baseline'],
                y1=baseline_metrics['power_baseline'],
                line=dict(color='red', width=2, dash='dash'),
                yref='y1',
                name='피크 기준선'
            )
        
        # 전기요금
        fig_realtime.add_trace(go.Scatter(
            x=chart_display_data['측정일시'],
            y=chart_display_data['전기요금(원)'],
            name='전기요금 (원)',
            mode='lines',
            line=dict(color='#ff7f0e', width=2),
            yaxis='y2'
        ))
        
        fig_realtime.update_layout(
            title='최근 100개 데이터',
            hovermode='x unified',
            template='plotly_white',
            height=650,
            xaxis_title='측정시간',
            yaxis=dict(
                title='전력사용량 (kWh)',
                title_font=dict(color='#1f77b4'),
                tickfont=dict(color='#1f77b4'),
                side='left'
            ),
            yaxis2=dict(
                title='전기요금 (원)',
                title_font=dict(color='#ff7f0e'),
                tickfont=dict(color='#ff7f0e'),
                overlaying='y',
                side='right'
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_realtime, use_container_width=True, config={'displayModeBar': False})
    
    with calendar_col:
        # 제목
        st.markdown("#### 12월 캘린더 (일일 집계)")
        
        # 전체 December 데이터 기준으로 12월 총합 계산
        full_december_df = december_data.copy()
        full_december_df['측정일시'] = pd.to_datetime(full_december_df['측정일시'])
        
        # 12월 전체 합계
        december_total_usage = full_december_df['전력사용량(kWh)'].sum()
        december_total_charge = full_december_df['전기요금(원)'].sum()
        
        # 상단 요약 카드 (2개)
        summary_col1, summary_col2 = st.columns(2)
        
        summary_card_style = """
        <div style='
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 10px;
        '>
            <div style='font-size: 0.85rem; color: #666; margin-bottom: 5px;'>{title}</div>
            <div style='font-size: 1.4rem; font-weight: bold; color: #333;'>{value}</div>
            <div style='font-size: 0.75rem; color: #888;'>{unit}</div>
        </div>
        """
        
        with summary_col1:
            st.markdown(
                summary_card_style.format(
                    title="총 예상 전기요금",
                    value=f"{december_total_charge:,.0f}",
                    unit="원"
                ),
                unsafe_allow_html=True
            )
        
        with summary_col2:
            st.markdown(
                summary_card_style.format(
                    title="총 예상 전력사용량",
                    value=f"{december_total_usage:,.0f}",
                    unit="kWh"
                ),
                unsafe_allow_html=True
            )
        
        # 일일 집계
        full_december_df['날짜'] = full_december_df['측정일시'].dt.date
        full_daily_summary = full_december_df.groupby('날짜').agg({
            '전력사용량(kWh)': 'sum',
            '전기요금(원)': 'sum',
            '탄소배출량(tCO2)': 'sum'
        }).reset_index()
        
        # 달력 구조 (배경 없이) - 호버 효과 추가
        cal = calendar.monthcalendar(2024, 12)
        calendar_cols = st.columns(7)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for i, day_name in enumerate(day_names):
            with calendar_cols[i]:
                st.markdown(f"<div style='text-align: center; font-weight: bold; margin-bottom: 5px;'>{day_name}</div>", unsafe_allow_html=True)

        # CSS for hover effect
        st.markdown("""
        <style>
        .calendar-cell {
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .calendar-cell:hover {
            transform: scale(1.15);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2) !important;
            z-index: 100;
            position: relative;
        }
        </style>
        """, unsafe_allow_html=True)

        # 날짜 셀 렌더링
        for week in cal:
            calendar_cols = st.columns(7)
            for i, day in enumerate(week):
                with calendar_cols[i]:
                    cell_height = "70px"
                    if day == 0:
                        st.markdown(f"<div style='height: {cell_height};'></div>", unsafe_allow_html=True)
                    else:
                        date_obj = pd.Timestamp(2024, 12, day).date()
                        day_data = full_daily_summary[full_daily_summary['날짜'] == date_obj]
                        
                        if not day_data.empty:
                            usage = day_data['전력사용량(kWh)'].values[0]
                            charge = day_data['전기요금(원)'].values[0]
                            
                            # 데이터가 있는 날짜 셀
                            st.markdown(f"""
                            <div class='calendar-cell' style='height: {cell_height}; padding: 5px; border: 1px solid #1f77b4; background-color: #f0f8ff; border-radius: 5px; overflow-y: auto; cursor: pointer;'>
                                <div style='font-weight: bold;'>{day}</div>
                                <div style='font-size: 0.7rem;'><b>사용량:</b> {usage:,.1f} kWh</div>
                                <div style='font-size: 0.7rem;'><b>요금:</b> {charge:,.0f} 원</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            st.markdown(f"<div class='calendar-cell' style='height: {cell_height}; padding: 5px; border: 1px solid #eee; border-radius: 5px; cursor: pointer;'>{day}</div>", unsafe_allow_html=True)
    
    # === 구분선 1 ===
    st.markdown("---")
    
    # --- 7. 역률 모니터링 ---
    st.markdown("#### 역률 모니터링")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_lag_pf = go.Figure()
        
        # 지상역률 데이터
        fig_lag_pf.add_trace(go.Scatter(
            x=chart_display_data['측정일시'],
            y=chart_display_data['지상역률(%)'],
            name='지상역률 (Lagging)',
            mode='lines',
            line=dict(color='#d62728', width=2)
        ))
        
        # 회색 배경 영역 추가 (23시~09시는 중요하지 않음)
        if len(chart_display_data) > 0:
            # X축 범위 계산
            x_min = chart_display_data['측정일시'].min()
            x_max = chart_display_data['측정일시'].max()
            
            # 각 시간대별로 회색 영역 표시
            current_time = x_min.floor('H')
            while current_time <= x_max + pd.Timedelta(hours=1):
                hour = current_time.hour
                
                # 00시~09시 구간 (중요하지 않은 시간)
                if 0 <= hour < 9:
                    fig_lag_pf.add_vrect(
                        x0=current_time,
                        x1=current_time + pd.Timedelta(hours=1),
                        fillcolor="gray",
                        opacity=0.2,
                        layer="below",
                        line_width=0
                    )
                # 23시 구간 (중요하지 않은 시간)
                elif hour == 23:
                    fig_lag_pf.add_vrect(
                        x0=current_time,
                        x1=current_time + pd.Timedelta(hours=1),
                        fillcolor="gray",
                        opacity=0.2,
                        layer="below",
                        line_width=0
                    )
                
                current_time += pd.Timedelta(hours=1)
            
            # 기준선
            fig_lag_pf.add_shape(
                type='line',
                x0=x_min,
                x1=x_max,
                y0=90,
                y1=90,
                line=dict(color='red', width=2, dash='dash'),
                name='기준선 (90%)'
            )
        
        fig_lag_pf.update_layout(
            title='지상역률 (90% 미만 경보) - 중요시간: 09시~23시',
            template='plotly_white',
            height=300,
            yaxis_title='역률 (%)',
            xaxis_title='측정시간',
            yaxis=dict(range=[78, 101]),
            xaxis=dict(
                dtick=3600000,  # 1시간 단위 (밀리초)
                tickformat='%H:%M'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_lag_pf, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        fig_lead_pf = go.Figure()
        
        # 진상역률 데이터
        fig_lead_pf.add_trace(go.Scatter(
            x=chart_display_data['측정일시'],
            y=chart_display_data['진상역률(%)'],
            name='진상역률 (Leading)',
            mode='lines',
            line=dict(color='#2ca02c', width=2)
        ))
        
        # 회색 배경 영역 추가 (09시~23시는 중요하지 않음)
        if len(chart_display_data) > 0:
            # X축 범위 계산
            x_min = chart_display_data['측정일시'].min()
            x_max = chart_display_data['측정일시'].max()
            
            # 각 시간대별로 회색 영역 표시
            current_time = x_min.floor('H')
            while current_time <= x_max + pd.Timedelta(hours=1):
                hour = current_time.hour
                
                # 09시~23시 구간 (중요하지 않은 시간)
                if 9 <= hour < 23:
                    fig_lead_pf.add_vrect(
                        x0=current_time,
                        x1=current_time + pd.Timedelta(hours=1),
                        fillcolor="gray",
                        opacity=0.2,
                        layer="below",
                        line_width=0
                    )
                
                current_time += pd.Timedelta(hours=1)
            
            # 기준선
            fig_lead_pf.add_shape(
                type='line',
                x0=x_min,
                x1=x_max,
                y0=95,
                y1=95,
                line=dict(color='red', width=2, dash='dash'),
                name='기준선 (95%)'
            )
        
        fig_lead_pf.update_layout(
            title='진상역률 (95% 미만 경보) - 중요시간: 23시~09시',
            template='plotly_white',
            height=300,
            yaxis_title='역률 (%)',
            xaxis_title='측정시간',
            yaxis=dict(range=[88, 101]),
            xaxis=dict(
                dtick=3600000,  # 1시간 단위 (밀리초)
                tickformat='%H:%M'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_lead_pf, use_container_width=True, config={'displayModeBar': False})
    
    # === 구분선 2 ===
    st.markdown("---")
    
    # --- 8. 탄소배출량 ---
    st.markdown("#### 탄소배출량")
    
    col1, col2 = st.columns(2)
    
    if len(all_display_data) > 0:
        with col1:
            current_co2 = all_display_data['탄소배출량(tCO2)'].iloc[-1]
            st.metric(
                "현재 배출량 (tCO2)",
                f"{current_co2:.6f}",
                delta=None
            )
        
        with col2:
            cumulative_co2 = all_display_data['탄소배출량(tCO2)'].sum()
            st.metric(
                "누적 배출량 (tCO2)",
                f"{cumulative_co2:.4f}",
                delta=None
            )
    else:
        with col1: st.metric("현재 배출량 (tCO2)", "0.000000")
        with col2: st.metric("누적 배출량 (tCO2)", "0.0000")
    
    # 시간대별 탄소배출량 추이 그래프
    fig_co2 = go.Figure()
    
    if len(all_display_data) > 0:
        # 기본 탄소배출량 라인
        fig_co2.add_trace(go.Scatter(
            x=all_display_data['측정일시'],
            y=all_display_data['탄소배출량(tCO2)'],
            name='탄소배출량',
            mode='lines',
            line=dict(color='#17becf', width=2),
            fill='tozeroy',
            fillcolor='rgba(23, 190, 207, 0.3)'
        ))
        
        # 이동평균선 추가 (6시간)
        if len(all_display_data) >= 6:
            moving_avg = all_display_data['탄소배출량(tCO2)'].rolling(window=6).mean()
            fig_co2.add_trace(go.Scatter(
                x=all_display_data['측정일시'],
                y=moving_avg,
                name='6시간 이동평균',
                mode='lines',
                line=dict(color='#d62728', width=2, dash='dash')
            ))
        
        # 평균 배출량 기준선 추가
        avg_co2 = all_display_data['탄소배출량(tCO2)'].mean()
        fig_co2.add_shape(
            type='line',
            x0=all_display_data['측정일시'].iloc[0],
            x1=all_display_data['측정일시'].iloc[-1],
            y0=avg_co2,
            y1=avg_co2,
            line=dict(color='orange', width=2, dash='dot'),
            name='평균 배출량'
        )
        
        # 주석 추가
        fig_co2.add_annotation(
            x=all_display_data['측정일시'].iloc[-1],
            y=avg_co2,
            text=f"평균: {avg_co2:.6f} tCO2",
            showarrow=False,
            xanchor='right',
            yanchor='bottom',
            font=dict(color='orange', size=10)
        )
    
    fig_co2.update_layout(
        title='시간대별 탄소배출량 추이',
        template='plotly_white',
        height=300,
        xaxis_title='측정시간',
        yaxis_title='탄소배출량 (tCO2)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        yaxis=dict(autorange=True)
    )
    st.plotly_chart(fig_co2, use_container_width=True, config={'displayModeBar': False})
    
    # === 구분선 3 ===
    st.markdown("---")
    
    # --- 9. 알람 로그 ---
    st.markdown("#### 알람 로그")
    
    if len(all_display_data) > 0:
        all_alerts = []
        for idx, row in all_display_data.iterrows():
            row_dict = row.to_dict()
            row_alerts = check_alerts(row_dict, baseline_metrics)
            all_alerts.extend(row_alerts)
        
        if all_alerts:
            # 최근 20개 알람만 표시
            recent_alerts = sorted(
                all_alerts,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:20]
            
            alerts_df = pd.DataFrame(recent_alerts)
            alerts_df['시간'] = alerts_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            display_df = alerts_df[['시간', 'type', 'value', 'severity']].copy()
            display_df.columns = ['시간', '알람 유형', '값', '심각도']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # 알람 통계
            col1, col2, col3 = st.columns(3)
            
            with col1:
                peak_alerts = len([a for a in all_alerts if a['type'] == '피크 초과'])
                st.metric("피크 초과", peak_alerts)
            
            with col2:
                lag_alerts = len([a for a in all_alerts if a['type'] == '지상역률 경고'])
                st.metric("지상역률 위반", lag_alerts)
            
            with col3:
                lead_alerts = len([a for a in all_alerts if a['type'] == '진상역률 경보'])
                st.metric("진상역률 위반", lead_alerts)
        else:
            st.info("발생한 알람이 없습니다. (기준선 이내)")
    else:
        st.info("스트리밍을 시작하여 알람 모니터링을 시작하세요.")

    # --- 10. 스트리밍 루프 (폴링) ---
    if st.session_state.streaming_active:
        time.sleep(0.5)
        st.rerun()

# --- 앱 실행 (메인 스크립트) ---
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render("tab1")