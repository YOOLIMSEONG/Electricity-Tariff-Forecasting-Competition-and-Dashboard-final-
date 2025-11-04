# dashboard/modules/tab3.py
from pathlib import Path
from io import StringIO
from datetime import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MONTH_OPTIONS = [
    ("2024-01", "2024년 1월"),
    ("2024-02", "2024년 2월"),
    ("2024-03", "2024년 3월"),
    ("2024-04", "2024년 4월"),
    ("2024-05", "2024년 5월"),
    ("2024-06", "2024년 6월"),
    ("2024-07", "2024년 7월"),
    ("2024-08", "2024년 8월"),
    ("2024-09", "2024년 9월"),
    ("2024-10", "2024년 10월"),
    ("2024-11", "2024년 11월"),
]
MONTH_LABELS = dict(MONTH_OPTIONS)
WEEKDAY_LABELS = {
    0: "월",
    1: "화",
    2: "수",
    3: "목",
    4: "금",
    5: "토",
    6: "일",
}
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"
CACHE_VERSION = "midnight_adjust_v3"

TARIFF_BASIC_RATES = {
    "산업용(갑) I": {
        "저압전력": {"": 5550},
        "고압A": {"선택 I": 6490, "선택 II": 7470},
        "고압B": {"선택 I": 6000, "선택 II": 6900},
    },
    "산업용(갑) II": {
        "저압전력": {"": 5550},
        "고압A": {"선택 I": 6490, "선택 II": 7470},
        "고압B": {"선택 I": 6000, "선택 II": 6900},
    },
    "산업용(을)": {
        "고압A": {"선택 I": 7220, "선택 II": 8320, "선택 III": 9810},
        "고압B": {"선택 I": 6630, "선택 II": 7380, "선택 III": 8190},
        "고압C": {"선택 I": 6590, "선택 II": 7520, "선택 III": 8090},
    },
}
DEFAULT_TARIFF_SELECTION = ("산업용(을)", "고압A", "선택 I")
SEASON_GROUPS = {
    "summer": {"label": "하계(6~8월)", "months": {6, 7, 8}},
    "winter": {"label": "동계(11~2월)", "months": {11, 12, 1, 2}},
}
SEASON_STYLE = {
    "summer": {"color": "#F59E0B", "dash": "dash"},
    "winter": {"color": "#0EA5E9", "dash": "dash"},
}
PEAK_MODE_LABELS = {
    "monthly": {"series": "월별 피크 전력(15분)", "avg": "월별 평균 전력(15분)", "x_title": "월"},
    "daily": {"series": "일별 피크 전력(15분)", "avg": "일별 평균 전력(15분)", "x_title": "날짜"},
    "15min": {"series": "피크 전력 (15분)", "avg": "평균 전력(15분)", "x_title": "시간"},
}
PF_TREND_LABELS = {"monthly": "월별 추이", "daily": "일별 추이", "15min": "시간별 추이"}
PF_DAY_START_HOUR = 9
PF_DAY_END_HOUR = 23  # exclusive
PF_DAY_THRESHOLD = 90.0
PF_NIGHT_THRESHOLD = 95.0
PF_MIN_LIMIT = 60.0
PF_DAY_MAX_LIMIT = 95.0
PF_NIGHT_MAX_LIMIT = 95.0
PF_PENALTY_RATE_PER_PERCENT = 0.005


def format_month_label(value: str) -> str:
    return MONTH_LABELS.get(value, value)


@st.cache_data(show_spinner=False)
def load_data(_cache_version: str = CACHE_VERSION):
    df = pd.read_csv(DATA_PATH, parse_dates=["측정일시"])
    # 00:00:00 기록은 전일 24:00과 동일하므로 전날로 1초 이동시켜 월/일 계산 시 포함되도록 조정
    midnight_mask = df["측정일시"].dt.time == time(0, 0, 0)
    df.loc[midnight_mask, "측정일시"] = df.loc[midnight_mask, "측정일시"] - pd.Timedelta(seconds=1)
    if "연월" in df.columns:
        df = df.drop(columns=["연월"])
    df["연월"] = df["측정일시"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("연월")
        .agg(
            총전력사용량=("전력사용량(kWh)", "sum"),
            총전기요금=("전기요금(원)", "sum"),
            총탄소배출량=("탄소배출량(tCO2)", "sum"),
        )
        .sort_index()
    )
    monthly["평균단가"] = monthly["총전기요금"] / monthly["총전력사용량"].replace({0: pd.NA})
    return df, monthly


def format_number(value, decimals=0):
    if pd.isna(value):
        return "-"
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def coalesce_number(value, default=0.0):
    if pd.isna(value):
        return default
    return float(value)


def compute_overall_metrics(df: pd.DataFrame) -> pd.Series:
    total_usage = df["전력사용량(kWh)"].sum()
    total_cost = df["전기요금(원)"].sum()
    total_emission = df["탄소배출량(tCO2)"].sum()
    avg_unit_cost = pd.NA
    if not pd.isna(total_usage) and total_usage != 0:
        avg_unit_cost = total_cost / total_usage
    return pd.Series(
        {
            "총전력사용량": total_usage,
            "총전기요금": total_cost,
            "총탄소배출량": total_emission,
            "평균단가": avg_unit_cost,
        }
    )


def list_tariff_voltages(tariff_type: str):
    return list(TARIFF_BASIC_RATES.get(tariff_type, {}).keys())


def list_tariff_options(tariff_type: str, voltage: str):
    return TARIFF_BASIC_RATES.get(tariff_type, {}).get(voltage, {})


def get_basic_rate(tariff_type: str, voltage: str, option: str):
    voltage_options = list_tariff_options(tariff_type, voltage)
    if not voltage_options:
        return None
    if option in voltage_options:
        return float(voltage_options[option])
    if "" in voltage_options:
        return float(voltage_options[""])
    # If option is missing but only one exists, return it.
    if len(voltage_options) == 1:
        return float(next(iter(voltage_options.values())))
    return None


def format_option_label(option_key: str) -> str:
    if not option_key:
        return "기본"
    return option_key


def dataframe_to_csv(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


def render(title: str = "과거 데이터 분석"):
    st.markdown(
        """
        <style>
        .download-btn-container {
            display: flex;
            align-items: flex-start;
            height: 100%;
            justify-content: flex-end;
            margin-top: 8px;
        }
        .download-btn-container div[data-testid="stDownloadButton"] {
            width: 100%;
        }
        div[data-testid="stDownloadButton"] > button {
            background: #6366F1 !important;
            color: #FFFFFF !important;
            border: 1px solid #6366F1 !important;
            border-radius: 12px !important;
            margin-top: 0 !important;
            width: 100%;
            padding: 0.65rem 1.4rem;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background: #818CF8 !important;
            border-color: #818CF8 !important;
        }
        .tab3-title {
            font-size: 30px !important;
            font-weight: 700 !important;
            color: #1A202C !important;
            margin-bottom: 8px !important;
        }
        .tab3-time-label {
            font-size: 15px !important;
            font-weight: 600 !important;
            color: #1A202C !important;
            margin-bottom: 12px !important;
        }
        .metric-card {
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-height: 120px;
            color: #1A202C;
        }
        .metric-card .metric-body {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 4px;
        }
        .metric-card .metric-value {
            display: inline-flex;
            align-items: baseline;
            gap: 6px;
            font-size: 32px;
            font-weight: 600;
            line-height: 1.1;
            color: #1A202C;
        }
        .metric-card .metric-unit-inline {
            font-size: 16px;
            font-weight: 500;
            color: #4A5568;
        }
        .metric-delta {
            font-size: 13px;
            margin-top: 6px;
        }
        .metric-delta strong {
            font-weight: 600;
        }
        .metric-delta.positive {
            color: #DC2626;
        }
        .metric-delta.negative {
            color: #2563EB;
        }
        .metric-delta.neutral {
            color: #4A5568;
        }
        .metric-delta.metric-delta-empty {
            visibility: hidden;
            margin-top: 6px;
        }
        .detail-toggle-label {
            display: inline-flex;
            align-items: center;
            font-size: 15px;
            font-weight: 500;
            color: #1A202C;
            margin-right: 12px;
        }
        .detail-toggle-wrapper {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            margin-top: 6px;
        }
        .toggle-help {
            display: inline-flex;
            position: relative;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: rgba(99,102,241,0.16);
            border: 1px solid rgba(99,102,241,0.4);
            color: #6366F1;
            font-size: 12px;
            cursor: pointer;
        }
        .toggle-help::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: -38px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(99,102,241,0.25);
            color: #1A202C;
            font-size: 12px;
            padding: 6px 10px;
            border-radius: 8px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.15s ease;
        }
        .toggle-help:hover::after {
            opacity: 1;
        }
        div[data-testid="stSelectbox"] > label {
            color: #4A5568 !important;
            font-weight: 500;
            margin-bottom: 6px;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            background: #FFFFFF;
            border: 1px solid #6366F1;
            border-radius: 12px;
            box-shadow: 0 8px 18px rgba(99,102,241,0.12);
            color: #1A202C;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"]:hover {
            border-color: #818CF8;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: transparent;
            color: inherit;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
            fill: #6366F1;
        }
        div[data-testid="stDateInput"] label,
        div[data-testid="stDateInput"] span,
        div[data-testid="stDateInput"] input,
        label[for^="switch-"] {
            color: #1A202C !important;
        }
        div[data-testid="stToggle"] label div[role="switch"] {
            border-color: rgba(99,102,241,0.4) !important;
        }
        div[data-testid="stToggle"] {
            display: inline-flex;
            align-items: center;
            margin: -4px 0 0 0;
        }
        div[data-testid="stToggle"] > label {
            display: inline-flex;
            align-items: center;
            margin: 0 !important;
        }
        div[data-testid="stToggle"] > label > div:first-child {
            margin: 0;
        }
        div[data-testid="stToggle"] label div[role="switch"] {
            margin: 0;
            transform: translateY(-4px);
        }
        div[data-testid="stDateInput"] input {
            background: #FFFFFF !important;
            border: 1px solid #6366F1 !important;
            color: #1A202C !important;
        }
        div[data-testid="stDateInput"] input:focus {
            border-color: #818CF8 !important;
            box-shadow: 0 0 0 1px #818CF8;
        }
        div[data-testid="stDateInput"] div[data-baseweb="popover"] {
            background: #FFFFFF !important;
        }
        div[data-baseweb="popover"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 28px rgba(26,32,44,0.18) !important;
            overflow: visible !important;
            padding: 0 !important;
        }
        div[data-baseweb="popover"] div[data-baseweb="menu"],
        div[data-baseweb="popover"] ul {
            background: #F7FAFC !important;
            color: #1A202C !important;
            border-radius: 12px !important;
            overflow: auto !important;
            padding: 8px !important;
        }
        div[data-baseweb="popover"] li[role="option"],
        div[data-baseweb="popover"] div[role="option"] {
            background: transparent !important;
            color: #1A202C !important;
            border-radius: 10px !important;
        }
        div[data-baseweb="popover"] li[role="option"]:hover,
        div[data-baseweb="popover"] div[role="option"]:hover {
            background: rgba(99,102,241,0.15) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    raw_df, monthly_summary = load_data()
    monthly_summary = monthly_summary[monthly_summary["총전력사용량"] > 0].sort_index()
    raw_df = raw_df.sort_values("측정일시")
    min_available_date = raw_df["측정일시"].min().date() if not raw_df.empty else None
    max_available_date = raw_df["측정일시"].max().date() if not raw_df.empty else None

    global_peak_kw = pd.NA
    global_peak_kwh = pd.NA
    global_peak_timestamp = None
    global_peak_display = None
    if not raw_df.empty:
        try:
            peak_idx = raw_df["전력사용량(kWh)"].idxmax()
            if pd.notna(peak_idx):
                peak_row = raw_df.loc[peak_idx]
                global_peak_kwh = float(peak_row["전력사용량(kWh)"])
                global_peak_kw = global_peak_kwh * 4
                global_peak_timestamp = peak_row["측정일시"]
                if pd.notna(global_peak_timestamp):
                    global_peak_display = pd.to_datetime(global_peak_timestamp).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass

    top_left, top_right = st.columns([4, 1], gap="large")
    with top_left:
        st.markdown(f'<div class="tab3-title">{title}</div>', unsafe_allow_html=True)
        st.markdown('<div class="tab3-time-label">시간 단위 선택</div>', unsafe_allow_html=True)
        time_mode = st.radio(
            "시간 단위 선택",
            options=("전체", "월별", "사용자 정의"),
            horizontal=True,
            key="tab3_time_mode",
            label_visibility="collapsed",
        )
        selected_month = None
        custom_range = None
        available_months = list(monthly_summary.index)
        if time_mode == "월별":
            if not available_months:
                st.warning("선택 가능한 월 데이터가 없습니다.")
            else:
                selected_month = st.selectbox(
                    "분석 대상 월을 선택하세요",
                    options=available_months,
                    format_func=format_month_label,
                    key="tab3_month_selector",
                )
        elif time_mode == "사용자 정의":
            if min_available_date is None or max_available_date is None:
                st.warning("사용자 정의 범위를 설정할 데이터가 없습니다.")
            else:
                default_month = available_months[0] if available_months else str(pd.Period(min_available_date, freq="M"))
                default_period = pd.Period(default_month, freq="M")
                default_start_date = default_period.to_timestamp().date()
                default_end_timestamp = (default_period + 1).to_timestamp() - pd.Timedelta(seconds=1)
                default_end_date = min(default_end_timestamp.date(), max_available_date)
                custom_range = st.date_input(
                    "사용자 정의 기간",
                    value=(default_start_date, default_end_date),
                    min_value=min_available_date,
                    max_value=max_available_date,
                    key="tab3_custom_range",
                )
    button_container = top_right.container()

    if time_mode == "월별" and available_months and selected_month is None:
        selected_month = available_months[0]

    if raw_df.empty or monthly_summary.empty:
        st.info("과거 데이터가 없습니다.")
        with button_container:
            st.markdown('<div class="download-btn-container tab3-download"></div>', unsafe_allow_html=True)
        return

    summary_series = None
    summary_title = ""
    chart_config = {}
    report_df = None
    download_name = "analysis.csv"
    detail_source = raw_df
    top_records = pd.DataFrame()
    series_usage = pd.Series(dtype=float)
    peak_mode = None
    peak_source = None
    monthly_change_map = {}

    if time_mode == "전체":
        range_df = raw_df.copy()
        summary_series = compute_overall_metrics(range_df)
        first_month = monthly_summary.index.min()
        last_month = monthly_summary.index.max()
        summary_title = f"{format_month_label(first_month)} ~ {format_month_label(last_month)} 주요 지표"

        chart_df = monthly_summary.reset_index().copy().fillna(0)
        monthly_avg = (
            range_df.groupby("연월")
            .agg(
                평균전력사용량=("전력사용량(kWh)", "mean"),
                평균전기요금=("전기요금(원)", "mean"),
            )
            .reset_index()
        )
        chart_df = chart_df.merge(monthly_avg, on="연월", how="left")
        chart_df["평균전력사용량"] = chart_df["평균전력사용량"].fillna(0)
        chart_df["평균전기요금"] = chart_df["평균전기요금"].fillna(0)
        chart_df["표시월"] = chart_df["연월"].apply(format_month_label)
        chart_df["월시작"] = pd.to_datetime(chart_df["연월"])
        chart_df["평균전력사용량_3개월평균"] = (
            chart_df["평균전력사용량"].rolling(window=3, min_periods=1).mean()
        )
        chart_df["평균전기요금_3개월평균"] = (
            chart_df["평균전기요금"].rolling(window=3, min_periods=1).mean()
        )

        report_df = chart_df[
            [
                "연월",
                "표시월",
                "평균전력사용량",
                "평균전력사용량_3개월평균",
                "평균전기요금",
                "평균전기요금_3개월평균",
                "총전력사용량",
                "총전기요금",
                "총탄소배출량",
                "평균단가",
            ]
        ].rename(
            columns={
                "연월": "월",
                "표시월": "월(표시)",
                "평균전력사용량": "평균전력사용량(kWh)",
                "평균전력사용량_3개월평균": "평균전력사용량 3개월 이동평균(kWh)",
                "평균전기요금": "평균전기요금(원)",
                "평균전기요금_3개월평균": "평균전기요금 3개월 이동평균(원)",
                "총전력사용량": "총전력사용량(kWh)",
                "총전기요금": "총전기요금(원)",
                "총탄소배출량": "총탄소배출량(tCO2)",
                "평균단가": "평균단가(원/kWh)",
            }
        )

        overall_avg_usage = range_df["전력사용량(kWh)"].mean()
        overall_avg_cost = range_df["전기요금(원)"].mean()
        totals_row = {
            "월": "전체",
            "월(표시)": f"{format_month_label(first_month)}~{format_month_label(last_month)}",
            "평균전력사용량(kWh)": coalesce_number(overall_avg_usage, 0.0),
            "평균전력사용량 3개월 이동평균(kWh)": 0.0,
            "평균전기요금(원)": coalesce_number(overall_avg_cost, 0.0),
            "평균전기요금 3개월 이동평균(원)": 0.0,
            "총전력사용량(kWh)": coalesce_number(summary_series.get("총전력사용량"), 0.0),
            "총전기요금(원)": coalesce_number(summary_series.get("총전기요금"), 0.0),
            "총탄소배출량(tCO2)": coalesce_number(summary_series.get("총탄소배출량"), 0.0),
            "평균단가(원/kWh)": coalesce_number(summary_series.get("평균단가"), 0.0),
        }
        report_df = pd.concat([report_df, pd.DataFrame([totals_row])], ignore_index=True).fillna(0)

        month_ticks = chart_df["월시작"].tolist()
        chart_config = {
            "x": chart_df["월시작"],
            "xaxis_title": "월",
            "usage_series": chart_df["평균전력사용량"],
            "usage_ma": chart_df["평균전력사용량_3개월평균"],
            "usage_label": "전력사용량 (월평균)",
            "usage_ma_label": "전력사용량 (3개월 이동평균)",
            "cost_series": chart_df["평균전기요금"],
            "cost_ma": chart_df["평균전기요금_3개월평균"],
            "cost_label": "전기요금 (월평균)",
            "cost_ma_label": "전기요금 (3개월 이동평균)",
            "tickvals": month_ticks,
            "ticktext": chart_df["표시월"].tolist(),
        }
        download_name = f"analysis_{first_month}_{last_month}.csv"
        detail_source = range_df
        series_usage = range_df["전력사용량(kWh)"]
        top_records = range_df.nlargest(5, "전력사용량(kWh)").copy()
        if range_df.empty:
            peak_mode = "monthly"
            peak_source = pd.DataFrame()
        else:
            time_delta = range_df["측정일시"].max() - range_df["측정일시"].min()
            span_hours = time_delta.total_seconds() / 3600 if pd.notna(time_delta) else 0
            unique_points = range_df["측정일시"].nunique()
            if span_hours <= 72 or unique_points <= 288:
                peak_mode = "15min"
                fifteen_peak = (
                    range_df.groupby("측정일시")["전력사용량(kWh)"]
                    .agg(["max", "mean"])
                    .reset_index()
                    .rename(columns={"max": "피크전력사용량(kWh)", "mean": "평균전력사용량(kWh)"})
                    .sort_values("측정일시")
                    .reset_index(drop=True)
                )
                peak_source = fifteen_peak.copy()
                peak_source["피크수요전력(kW)"] = peak_source["피크전력사용량(kWh)"] * 4
                peak_source["평균수요전력(kW)"] = peak_source["평균전력사용량(kWh)"] * 4
            elif span_hours <= 24 * 92:
                peak_mode = "daily"
                peak_source = (
                    range_df.assign(날짜=range_df["측정일시"].dt.date)
                    .groupby("날짜")["전력사용량(kWh)"]
                    .agg(["max", "mean"])
                    .reset_index()
                    .rename(columns={"max": "피크전력사용량(kWh)", "mean": "평균전력사용량(kWh)"})
                    .sort_values("날짜")
                    .reset_index(drop=True)
                )
                peak_source["피크수요전력(kW)"] = peak_source["피크전력사용량(kWh)"] * 4
                peak_source["평균수요전력(kW)"] = peak_source["평균전력사용량(kWh)"] * 4
            else:
                peak_mode = "monthly"
                peak_source = (
                    range_df.assign(연월=range_df["측정일시"].dt.to_period("M"))
                    .groupby("연월")
                    .agg(
                        피크전력사용량_kwh=("전력사용량(kWh)", "max"),
                        총전력사용량_kwh=("전력사용량(kWh)", "sum"),
                        계측수=("전력사용량(kWh)", "count"),
                    )
                    .reset_index()
                )
                peak_source["월시작"] = peak_source["연월"].dt.to_timestamp()
                peak_source["표시월"] = peak_source["연월"].astype(str).map(format_month_label)
                peak_source["피크수요전력(kW)"] = peak_source["피크전력사용량_kwh"] * 4
                peak_source["평균수요전력(kW)"] = (
                    peak_source["총전력사용량_kwh"] / peak_source["계측수"].replace({0: pd.NA})
                ) * 4
                peak_source = (
                    peak_source.rename(columns={"피크전력사용량_kwh": "피크전력사용량(kWh)"})
                    .sort_values("월시작")
                    .reset_index(drop=True)
                )

    elif time_mode == "월별":
        if not available_months or selected_month not in monthly_summary.index:
            st.info("선택 가능한 월 데이터가 없습니다.")
            with button_container:
                st.markdown('<div class="download-btn-container tab3-download"></div>', unsafe_allow_html=True)
            return

        summary_series = monthly_summary.loc[selected_month]
        summary_title = f"{format_month_label(selected_month)} 주요 지표"
        monthly_change_map = {}
        if selected_month in available_months:
            sel_index = available_months.index(selected_month)
            if sel_index > 0:
                prev_month_key = available_months[sel_index - 1]
                prev_series = monthly_summary.loc[prev_month_key]
                for metric_key in ("총전력사용량", "총전기요금", "총탄소배출량", "평균단가"):
                    prev_value = prev_series.get(metric_key, pd.NA)
                    curr_value = summary_series.get(metric_key, pd.NA)
                    change_pct = None
                    if pd.notna(prev_value) and prev_value != 0 and pd.notna(curr_value):
                        change_pct = (curr_value - prev_value) / prev_value * 100
                    monthly_change_map[metric_key] = {
                        "change_pct": change_pct,
                        "prev_value": prev_value,
                        "prev_month": prev_month_key,
                    }

        range_df = raw_df[raw_df["연월"] == selected_month].copy()
        if range_df.empty:
            st.info("선택한 월의 세부 데이터가 없습니다.")
            with button_container:
                st.markdown('<div class="download-btn-container tab3-download"></div>', unsafe_allow_html=True)
            return

        range_df = range_df.sort_values("측정일시")
        detail_source = range_df
        series_usage = range_df["전력사용량(kWh)"]
        top_records = range_df.nlargest(5, "전력사용량(kWh)").copy()

        daily_stats = (
            range_df.set_index("측정일시")
            .resample("D")
            .agg(
                {
                    "전력사용량(kWh)": ["sum", "mean"],
                    "전기요금(원)": ["sum", "mean"],
                }
            )
            .reset_index()
        )
        daily_stats.columns = [
            "측정일시",
            "전력사용량_합계",
            "전력사용량_평균",
            "전기요금_합계",
            "전기요금_평균",
        ]
        daily_stats["날짜"] = daily_stats["측정일시"].dt.date
        daily_stats["전력사용량_평균_7일평균"] = (
            daily_stats["전력사용량_평균"].rolling(window=7, min_periods=1).mean()
        )
        daily_stats["전기요금_평균_7일평균"] = (
            daily_stats["전기요금_평균"].rolling(window=7, min_periods=1).mean()
        )
        daily_stats["평균단가(원/kWh)"] = daily_stats["전기요금_합계"] / daily_stats["전력사용량_합계"].replace({0: pd.NA})

        chart_config = {
            "x": daily_stats["날짜"],
            "xaxis_title": "날짜",
            "usage_series": daily_stats["전력사용량_평균"],
            "usage_ma": daily_stats["전력사용량_평균_7일평균"],
            "usage_label": "전력사용량 (일평균)",
            "usage_ma_label": "전력사용량 (7일 이동평균)",
            "cost_series": daily_stats["전기요금_평균"],
            "cost_ma": daily_stats["전기요금_평균_7일평균"],
            "cost_label": "전기요금 (일평균)",
            "cost_ma_label": "전기요금 (7일 이동평균)",
        }

        report_df = daily_stats[
            [
                "날짜",
                "전력사용량_평균",
                "전력사용량_평균_7일평균",
                "전기요금_평균",
                "전기요금_평균_7일평균",
                "전력사용량_합계",
                "전기요금_합계",
                "평균단가(원/kWh)",
            ]
        ].rename(
            columns={
                "전력사용량_평균": "전력사용량(일평균,kWh)",
                "전력사용량_평균_7일평균": "전력사용량(일평균) 7일 이동평균(kWh)",
                "전기요금_평균": "전기요금(일평균,원)",
                "전기요금_평균_7일평균": "전기요금(일평균) 7일 이동평균(원)",
                "전력사용량_합계": "전력사용량(합계,kWh)",
                "전기요금_합계": "전기요금(합계,원)",
            }
        )
        report_df["날짜"] = report_df["날짜"].astype(str)
        totals_row = {
            "날짜": "합계",
            "전력사용량(일평균,kWh)": coalesce_number(range_df["전력사용량(kWh)"].mean(), 0.0),
            "전력사용량(일평균) 7일 이동평균(kWh)": 0.0,
            "전기요금(일평균,원)": coalesce_number(range_df["전기요금(원)"].mean(), 0.0),
            "전기요금(일평균) 7일 이동평균(원)": 0.0,
            "전력사용량(합계,kWh)": coalesce_number(summary_series.get("총전력사용량"), 0.0),
            "전기요금(합계,원)": coalesce_number(summary_series.get("총전기요금"), 0.0),
            "평균단가(원/kWh)": coalesce_number(summary_series.get("평균단가"), 0.0),
        }
        report_df = pd.concat([report_df, pd.DataFrame([totals_row])], ignore_index=True).fillna(0)
        download_name = f"analysis_{selected_month}.csv"
        time_delta_month = range_df["측정일시"].max() - range_df["측정일시"].min()
        span_hours_month = time_delta_month.total_seconds() / 3600 if pd.notna(time_delta_month) else 0
        unique_points_month = range_df["측정일시"].nunique()
        if span_hours_month <= 72 or unique_points_month <= 288:
            peak_mode = "15min"
            fifteen_peak = (
                range_df.groupby("측정일시")["전력사용량(kWh)"]
                .agg(["max", "mean"])
                .reset_index()
                .rename(columns={"max": "피크전력사용량(kWh)", "mean": "평균전력사용량(kWh)"})
                .sort_values("측정일시")
                .reset_index(drop=True)
            )
            peak_source = fifteen_peak.copy()
            peak_source["피크수요전력(kW)"] = peak_source["피크전력사용량(kWh)"] * 4
            peak_source["평균수요전력(kW)"] = peak_source["평균전력사용량(kWh)"] * 4
        else:
            peak_mode = "daily"
            peak_source = (
                range_df.assign(날짜=range_df["측정일시"].dt.date)
                .groupby("날짜")["전력사용량(kWh)"]
                .agg(["max", "mean"])
                .reset_index()
                .rename(columns={"max": "피크전력사용량(kWh)", "mean": "평균전력사용량(kWh)"})
                .sort_values("날짜")
                .reset_index(drop=True)
            )
            peak_source["피크수요전력(kW)"] = peak_source["피크전력사용량(kWh)"] * 4
            peak_source["평균수요전력(kW)"] = peak_source["평균전력사용량(kWh)"] * 4

    else:
        if custom_range is None or not isinstance(custom_range, (list, tuple)) or len(custom_range) != 2:
            st.warning("사용자 정의 기간을 올바르게 선택해주세요.")
            with button_container:
                st.markdown('<div class="download-btn-container tab3-download"></div>', unsafe_allow_html=True)
            return
        custom_start, custom_end = custom_range
        if custom_start is None or custom_end is None:
            st.warning("사용자 정의 기간을 올바르게 선택해주세요.")
            with button_container:
                st.markdown('<div class="download-btn-container tab3-download"></div>', unsafe_allow_html=True)
            return
        if custom_start > custom_end:
            custom_start, custom_end = custom_end, custom_start

        custom_start_ts = pd.Timestamp(custom_start)
        custom_end_ts = pd.Timestamp(custom_end) + pd.Timedelta(days=1)
        range_df = raw_df[(raw_df["측정일시"] >= custom_start_ts) & (raw_df["측정일시"] < custom_end_ts)].copy()
        if range_df.empty:
            st.info("선택한 기간에 해당하는 데이터가 없습니다.")
            with button_container:
                st.markdown('<div class="download-btn-container tab3-download"></div>', unsafe_allow_html=True)
            return

        range_df = range_df.sort_values("측정일시")
        detail_source = range_df
        summary_series = compute_overall_metrics(range_df)
        summary_title = f"{custom_start.strftime('%Y-%m-%d')} ~ {custom_end.strftime('%Y-%m-%d')} 주요 지표"
        series_usage = range_df["전력사용량(kWh)"]
        top_records = range_df.nlargest(5, "전력사용량(kWh)").copy()

        time_delta_custom = range_df["측정일시"].max() - range_df["측정일시"].min()
        span_hours_custom = time_delta_custom.total_seconds() / 3600 if pd.notna(time_delta_custom) else 0
        unique_points_custom = range_df["측정일시"].nunique()

        if span_hours_custom <= 72 or unique_points_custom <= 288:
            fifteen_avg = (
                range_df.groupby("측정일시")
                .agg(
                    전력사용량_평균=("전력사용량(kWh)", "mean"),
                    전기요금_평균=("전기요금(원)", "mean"),
                    탄소배출량=("탄소배출량(tCO2)", "mean"),
                )
                .reset_index()
            )
            fifteen_avg = fifteen_avg.sort_values("측정일시")
            fifteen_avg["전력사용량_평균_이동"] = fifteen_avg["전력사용량_평균"].rolling(window=12, min_periods=1).mean()
            fifteen_avg["전기요금_평균_이동"] = fifteen_avg["전기요금_평균"].rolling(window=12, min_periods=1).mean()
            chart_config = {
                "x": fifteen_avg["측정일시"],
                "xaxis_title": "시간",
                "usage_series": fifteen_avg["전력사용량_평균"],
                "usage_ma": fifteen_avg["전력사용량_평균_이동"],
                "usage_label": "전력사용량 (15분 평균)",
                "usage_ma_label": "전력사용량 (이동평균)",
                "cost_series": fifteen_avg["전기요금_평균"],
                "cost_ma": fifteen_avg["전기요금_평균_이동"],
                "cost_label": "전기요금 (15분 평균)",
                "cost_ma_label": "전기요금 (이동평균)",
            }
            report_df = fifteen_avg[
                ["측정일시", "전력사용량_평균", "전력사용량_평균_이동", "전기요금_평균", "전기요금_평균_이동", "탄소배출량"]
            ].rename(
                columns={
                    "전력사용량_평균": "전력사용량(15분 평균,kWh)",
                    "전력사용량_평균_이동": "전력사용량(15분 평균) 이동평균(kWh)",
                    "전기요금_평균": "전기요금(15분 평균,원)",
                    "전기요금_평균_이동": "전기요금(15분 평균) 이동평균(원)",
                    "탄소배출량": "탄소배출량(15분 평균,tCO2)",
                }
            )
            report_df["측정일시"] = report_df["측정일시"].astype(str)
            totals_row = {
                "측정일시": "합계",
                "전력사용량(15분 평균,kWh)": coalesce_number(range_df["전력사용량(kWh)"].mean(), 0.0),
                "전력사용량(15분 평균) 이동평균(kWh)": 0.0,
                "전기요금(15분 평균,원)": coalesce_number(range_df["전기요금(원)"].mean(), 0.0),
                "전기요금(15분 평균) 이동평균(원)": 0.0,
                "탄소배출량(15분 평균,tCO2)": coalesce_number(range_df["탄소배출량(tCO2)"].mean(), 0.0),
            }
            report_df = pd.concat([report_df, pd.DataFrame([totals_row])], ignore_index=True).fillna(0)
            download_name = f"analysis_custom_{custom_start.strftime('%Y%m%d')}_{custom_end.strftime('%Y%m%d')}.csv"
            peak_mode = "15min"
            fifteen_peak = (
                range_df.groupby("측정일시")["전력사용량(kWh)"]
                .agg(["max", "mean"])
                .reset_index()
                .rename(columns={"max": "피크전력사용량(kWh)", "mean": "평균전력사용량(kWh)"})
                .sort_values("측정일시")
                .reset_index(drop=True)
            )
            peak_source = fifteen_peak.copy()
            peak_source["피크수요전력(kW)"] = peak_source["피크전력사용량(kWh)"] * 4
            peak_source["평균수요전력(kW)"] = peak_source["평균전력사용량(kWh)"] * 4
        elif span_hours_custom <= 24 * 92:
            daily_stats = (
                range_df.set_index("측정일시")
                .resample("D")
                .agg(
                    {
                        "전력사용량(kWh)": ["sum", "mean"],
                        "전기요금(원)": ["sum", "mean"],
                    }
                )
                .reset_index()
            )
            daily_stats.columns = [
                "측정일시",
                "전력사용량_합계",
                "전력사용량_평균",
                "전기요금_합계",
                "전기요금_평균",
            ]
            daily_stats["날짜"] = daily_stats["측정일시"].dt.date
            daily_stats["전력사용량_평균_7일평균"] = (
                daily_stats["전력사용량_평균"].rolling(window=7, min_periods=1).mean()
            )
            daily_stats["전기요금_평균_7일평균"] = (
                daily_stats["전기요금_평균"].rolling(window=7, min_periods=1).mean()
            )
            daily_stats["평균단가(원/kWh)"] = daily_stats["전기요금_합계"] / daily_stats["전력사용량_합계"].replace({0: pd.NA})
            chart_config = {
                "x": daily_stats["날짜"],
                "xaxis_title": "날짜",
                "usage_series": daily_stats["전력사용량_평균"],
                "usage_ma": daily_stats["전력사용량_평균_7일평균"],
                "usage_label": "전력사용량 (일평균)",
                "usage_ma_label": "전력사용량 (7일 이동평균)",
                "cost_series": daily_stats["전기요금_평균"],
                "cost_ma": daily_stats["전기요금_평균_7일평균"],
                "cost_label": "전기요금 (일평균)",
                "cost_ma_label": "전기요금 (7일 이동평균)",
            }
            report_df = daily_stats[
                [
                    "날짜",
                    "전력사용량_평균",
                    "전력사용량_평균_7일평균",
                    "전기요금_평균",
                    "전기요금_평균_7일평균",
                    "전력사용량_합계",
                    "전기요금_합계",
                    "평균단가(원/kWh)",
                ]
            ].rename(
                columns={
                    "전력사용량_평균": "전력사용량(일평균,kWh)",
                    "전력사용량_평균_7일평균": "전력사용량(일평균) 7일 이동평균(kWh)",
                    "전기요금_평균": "전기요금(일평균,원)",
                    "전기요금_평균_7일평균": "전기요금(일평균) 7일 이동평균(원)",
                    "전력사용량_합계": "전력사용량(합계,kWh)",
                    "전기요금_합계": "전기요금(합계,원)",
                }
            )
            report_df["날짜"] = report_df["날짜"].astype(str)
            totals_row = {
                "날짜": "합계",
                "전력사용량(일평균,kWh)": coalesce_number(range_df["전력사용량(kWh)"].mean(), 0.0),
                "전력사용량(일평균) 7일 이동평균(kWh)": 0.0,
                "전기요금(일평균,원)": coalesce_number(range_df["전기요금(원)"].mean(), 0.0),
                "전기요금(일평균) 7일 이동평균(원)": 0.0,
                "전력사용량(합계,kWh)": coalesce_number(summary_series.get("총전력사용량"), 0.0),
                "전기요금(합계,원)": coalesce_number(summary_series.get("총전기요금"), 0.0),
                "평균단가(원/kWh)": coalesce_number(summary_series.get("평균단가"), 0.0),
            }
            report_df = pd.concat([report_df, pd.DataFrame([totals_row])], ignore_index=True).fillna(0)
            download_name = f"analysis_custom_{custom_start.strftime('%Y%m%d')}_{custom_end.strftime('%Y%m%d')}.csv"
            peak_mode = "daily"
            peak_source = (
                range_df.assign(날짜=range_df["측정일시"].dt.date)
                .groupby("날짜")["전력사용량(kWh)"]
                .agg(["max", "mean"])
                .reset_index()
                .rename(columns={"max": "피크전력사용량(kWh)", "mean": "평균전력사용량(kWh)"})
                .sort_values("날짜")
                .reset_index(drop=True)
            )
            peak_source["피크수요전력(kW)"] = peak_source["피크전력사용량(kWh)"] * 4
            peak_source["평균수요전력(kW)"] = peak_source["평균전력사용량(kWh)"] * 4
        else:
            custom_monthly = (
                range_df.assign(연월=range_df["측정일시"].dt.to_period("M"))
                .groupby("연월")
                .agg(
                    총전력사용량=("전력사용량(kWh)", "sum"),
                    총전기요금=("전기요금(원)", "sum"),
                    총탄소배출량=("탄소배출량(tCO2)", "sum"),
                    평균전력사용량=("전력사용량(kWh)", "mean"),
                    평균전기요금=("전기요금(원)", "mean"),
                )
                .sort_index()
            )
            custom_monthly["평균단가"] = custom_monthly["총전기요금"] / custom_monthly["총전력사용량"].replace({0: pd.NA})
            chart_df = custom_monthly.reset_index().copy()
            chart_df["연월"] = chart_df["연월"].astype(str)
            chart_df["표시월"] = chart_df["연월"].apply(format_month_label)
            chart_df["월시작"] = pd.to_datetime(chart_df["연월"])
            chart_df["평균전력사용량_3개월평균"] = chart_df["평균전력사용량"].rolling(window=3, min_periods=1).mean()
            chart_df["평균전기요금_3개월평균"] = chart_df["평균전기요금"].rolling(window=3, min_periods=1).mean()
            chart_config = {
                "x": chart_df["월시작"],
                "xaxis_title": "월",
                "usage_series": chart_df["평균전력사용량"],
                "usage_ma": chart_df["평균전력사용량_3개월평균"],
                "usage_label": "전력사용량 (월평균)",
                "usage_ma_label": "전력사용량 (3개월 이동평균)",
                "cost_series": chart_df["평균전기요금"],
                "cost_ma": chart_df["평균전기요금_3개월평균"],
                "cost_label": "전기요금 (월평균)",
                "cost_ma_label": "전기요금 (3개월 이동평균)",
                "tickvals": chart_df["월시작"].tolist(),
                "ticktext": chart_df["표시월"].tolist(),
            }
            report_df = chart_df[
                [
                    "연월",
                    "표시월",
                    "평균전력사용량",
                    "평균전력사용량_3개월평균",
                    "평균전기요금",
                    "평균전기요금_3개월평균",
                    "총전력사용량",
                    "총전기요금",
                    "총탄소배출량",
                    "평균단가",
                ]
            ].rename(
                columns={
                    "연월": "월",
                    "표시월": "월(표시)",
                    "평균전력사용량": "평균전력사용량(kWh)",
                    "평균전력사용량_3개월평균": "평균전력사용량 3개월 이동평균(kWh)",
                    "평균전기요금": "평균전기요금(원)",
                    "평균전기요금_3개월평균": "평균전기요금 3개월 이동평균(원)",
                    "총전력사용량": "총전력사용량(kWh)",
                    "총전기요금": "총전기요금(원)",
                    "총탄소배출량": "총탄소배출량(tCO2)",
                    "평균단가": "평균단가(원/kWh)",
                }
            )
            totals_row = {
                "월": f"{custom_start.strftime('%Y-%m-%d')}~{custom_end.strftime('%Y-%m-%d')}",
                "월(표시)": f"{chart_df['표시월'].iloc[0]}~{chart_df['표시월'].iloc[-1]}",
                "평균전력사용량(kWh)": coalesce_number(range_df["전력사용량(kWh)"].mean(), 0.0),
                "평균전력사용량 3개월 이동평균(kWh)": 0.0,
                "평균전기요금(원)": coalesce_number(range_df["전기요금(원)"].mean(), 0.0),
                "평균전기요금 3개월 이동평균(원)": 0.0,
                "총전력사용량(kWh)": coalesce_number(summary_series.get("총전력사용량"), 0.0),
                "총전기요금(원)": coalesce_number(summary_series.get("총전기요금"), 0.0),
                "총탄소배출량(tCO2)": coalesce_number(summary_series.get("총탄소배출량"), 0.0),
                "평균단가(원/kWh)": coalesce_number(summary_series.get("평균단가"), 0.0),
            }
        report_df = pd.concat([report_df, pd.DataFrame([totals_row])], ignore_index=True).fillna(0)
        download_name = f"analysis_custom_{custom_start.strftime('%Y%m%d')}_{custom_end.strftime('%Y%m%d')}.csv"
        if peak_mode is None or peak_source is None:
            peak_mode = "monthly"
            peak_source = (
                range_df.assign(연월=range_df["측정일시"].dt.to_period("M"))
                .groupby("연월")
                .agg(
                    피크전력사용량_kwh=("전력사용량(kWh)", "max"),
                    총전력사용량_kwh=("전력사용량(kWh)", "sum"),
                    계측수=("전력사용량(kWh)", "count"),
                )
                .reset_index()
            )
            peak_source["월시작"] = peak_source["연월"].dt.to_timestamp()
            peak_source["표시월"] = peak_source["연월"].astype(str).map(format_month_label)
            peak_source["피크수요전력(kW)"] = peak_source["피크전력사용량_kwh"] * 4
            peak_source["평균수요전력(kW)"] = (
                peak_source["총전력사용량_kwh"] / peak_source["계측수"].replace({0: pd.NA})
            ) * 4
            peak_source = (
                peak_source.rename(columns={"피크전력사용량_kwh": "피크전력사용량(kWh)"})
                .sort_values("월시작")
                .reset_index(drop=True)
            )

    peak_max_value = series_usage.max() if not series_usage.empty else pd.NA
    peak_avg_value = series_usage.mean() if not series_usage.empty else pd.NA

    if summary_series is None:

        summary_series = pd.Series(
            {
                "총전력사용량": pd.NA,
                "총전기요금": pd.NA,
                "총탄소배출량": pd.NA,
                "평균단가": pd.NA,
            }
        )

    if report_df is not None:
        if "월" in report_df.columns:
            report_df["월"] = report_df["월"].astype(str)
        if "월(표시)" in report_df.columns:
            report_df["월(표시)"] = report_df["월(표시)"].astype(str)
        report_csv = dataframe_to_csv(report_df)
    else:
        report_csv = ""

    with button_container:
        st.markdown('<div class="download-btn-container tab3-download">', unsafe_allow_html=True)
        st.download_button(
            label="분석 보고서 다운로드",
            data=report_csv,
            file_name=download_name,
            mime="text/csv",
            key="tab3_report_download",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"#### {summary_title}")
    metric_cols = st.columns(4, gap="large")
    metrics = [
        ("총 전력사용량(kWh)", "총전력사용량", 0, "kWh"),
        ("총 전기요금(원)", "총전기요금", 0, "원"),
        ("총 탄소배출량(tCO2)", "총탄소배출량", 2, "tCO2"),
        ("평균 단가(원/kWh)", "평균단가", 2, "원/kWh"),
    ]
    for column, (title_text, key, decimals, unit) in zip(metric_cols, metrics):
        with column:
            if key in summary_series.index:
                raw_value = summary_series[key]
            else:
                raw_value = pd.NA
            formatted_value = format_number(raw_value, decimals=decimals)
            change_html = ""
            if monthly_change_map and key in monthly_change_map:
                change_info = monthly_change_map[key]
                change_pct = change_info.get("change_pct")
                if change_pct is None:
                    change_html = ""
                else:
                    if change_pct > 0:
                        delta_class = "positive"
                        pct_label = f"+{change_pct:.1f}%"
                        direction_label = "증가"
                    elif change_pct < 0:
                        delta_class = "negative"
                        pct_label = f"{change_pct:.1f}%"
                        direction_label = "감소"
                    else:
                        delta_class = "neutral"
                        pct_label = "0.0%"
                        direction_label = "변화 없음"
                    change_html = (
                        f'<div class="metric-delta {delta_class}">전월 대비 '
                        f'<strong>{pct_label}</strong> {direction_label}</div>'
                    )
            card_segments = [
                '<div class="card metric-card">',
                f'    <div class="card-title">{title_text}</div>',
                '    <div class="metric-body">',
                f'        <div class="metric-value">{formatted_value}<span class="metric-unit-inline">{unit}</span></div>',
            ]
            if change_html:
                card_segments.append(f'        {change_html}')
            else:
                card_segments.append('        <div class="metric-delta metric-delta-empty">&nbsp;</div>')
            card_segments.extend(
                [
                    '    </div>',
                    '</div>',
                ]
            )
            card_html = "\n".join(card_segments)
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("#### 월별 세부 추이")

    daily_fig = go.Figure()
    daily_fig.add_trace(
        go.Scatter(
            x=chart_config["x"],
            y=chart_config["usage_series"],
            mode="lines+markers",
            name=chart_config["usage_label"],
            line=dict(color="#14B8A6", width=2.4),
            marker=dict(size=5),
        )
    )
    daily_fig.add_trace(
        go.Scatter(
            x=chart_config["x"],
            y=chart_config["usage_ma"],
            mode="lines",
            name=chart_config["usage_ma_label"],
            line=dict(color="#2DD4BF", width=2, dash="dash"),
        )
    )
    daily_fig.add_trace(
        go.Scatter(
            x=chart_config["x"],
            y=chart_config["cost_series"],
            mode="lines+markers",
            name=chart_config["cost_label"],
            line=dict(color="#6366F1", width=2.4),
            marker=dict(size=5),
            yaxis="y2",
        )
    )
    daily_fig.add_trace(
        go.Scatter(
            x=chart_config["x"],
            y=chart_config["cost_ma"],
            mode="lines",
            name=chart_config["cost_ma_label"],
            line=dict(color="#A5B4FF", width=2, dash="dash"),
            yaxis="y2",
        )
    )
    xaxis_settings = dict(
        title=dict(text=chart_config["xaxis_title"], font=dict(color="#1A202C")),
        tickfont=dict(color="#1A202C"),
    )
    if "tickvals" in chart_config:
        xaxis_settings["tickvals"] = chart_config["tickvals"]
    if "ticktext" in chart_config:
        xaxis_settings["ticktext"] = chart_config["ticktext"]

    daily_fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=45, b=20),
        hovermode="x unified",
        font=dict(color="#1A202C"),
        xaxis=xaxis_settings,
        yaxis=dict(
            title=dict(text="전력사용량 (kWh)", font=dict(color="#14B8A6")),
            color="#14B8A6",
            tickfont=dict(color="#4A5568"),
        ),
        yaxis2=dict(
            title=dict(text="전기요금 (원)", font=dict(color="#6366F1")),
            overlaying="y",
            side="right",
            color="#6366F1",
            tickfont=dict(color="#4A5568"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#1A202C"),
        ),
        paper_bgcolor="#F7FAFC",
        plot_bgcolor="#F7FAFC",
    )
    st.plotly_chart(daily_fig, config={"displayModeBar": True})

    if detail_source.empty:
        return

    col_label, col_toggle, col_help, _ = st.columns([0.2, 0.05, 0.03, 0.72], gap="small")
    with col_label:
        st.markdown(
            '<div class="detail-toggle-label">상세 보기 (15분 단위)</div>',
            unsafe_allow_html=True,
        )
    with col_toggle:
        detail_toggle = st.toggle(
            "상세 보기 (15분 단위)",
            value=st.session_state.get("tab3_detail_toggle", False),
            key="tab3_detail_toggle",
            label_visibility="collapsed",
        )
    with col_help:
        st.markdown(
            '<div class="toggle-help" data-tooltip="체크하면 선택 기간의 15분 단위 데이터가 표시됩니다.">?</div>',
            unsafe_allow_html=True,
        )
    if detail_toggle:
        start_date = detail_source["측정일시"].min().date()
        end_date = detail_source["측정일시"].max().date()
        date_range = st.date_input(
            "상세 조회 기간",
            value=(start_date, end_date),
            min_value=start_date,
            max_value=end_date,
            format="YYYY-MM-DD",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            detail_start, detail_end = date_range
        else:
            detail_start = start_date
            detail_end = end_date

        detail_df = detail_source[
            (detail_source["측정일시"] >= pd.Timestamp(detail_start))
            & (detail_source["측정일시"] < pd.Timestamp(detail_end) + pd.Timedelta(days=1))
        ]

        if detail_df.empty:
            st.warning("선택한 기간에 해당하는 상세 데이터가 없습니다.")
        else:
            detail_fig = go.Figure()
            detail_fig.add_trace(
                go.Scatter(
                    x=detail_df["측정일시"],
                    y=detail_df["전력사용량(kWh)"],
                    mode="lines",
                    name="전력사용량(15분 단위)",
                    line=dict(color="#14B8A6", width=1.8),
                )
            )
            detail_fig.add_trace(
                go.Scatter(
                    x=detail_df["측정일시"],
                    y=detail_df["전기요금(원)"],
                    mode="lines",
                    name="전기요금(15분 단위)",
                    line=dict(color="#6366F1", width=1.8),
                    yaxis="y2",
                )
            )
            detail_fig.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, t=45, b=20),
                hovermode="x unified",
                font=dict(color="#1A202C"),
                xaxis=dict(
                    title=dict(text="시간", font=dict(color="#1A202C")),
                    tickfont=dict(color="#1A202C"),
                ),
                yaxis=dict(
                    title=dict(text="전력사용량 (kWh)", font=dict(color="#14B8A6")),
                    color="#14B8A6",
                    tickfont=dict(color="#4A5568"),
                ),
                yaxis2=dict(
                    title=dict(text="전기요금 (원)", font=dict(color="#6366F1")),
                    overlaying="y",
                    side="right",
                    color="#6366F1",
                    tickfont=dict(color="#4A5568"),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#1A202C"),
                ),
                paper_bgcolor="#F7FAFC",
                plot_bgcolor="#F7FAFC",
            )
            st.plotly_chart(detail_fig, config={"displayModeBar": True})

    effective_end_ts = None
    if time_mode == "전체":
        effective_end_ts = raw_df["측정일시"].max() if not raw_df.empty else None
    elif not range_df.empty:
        effective_end_ts = range_df["측정일시"].max()

    eligible_df = raw_df
    if effective_end_ts is not None:
        eligible_df = raw_df[raw_df["측정일시"] <= effective_end_ts]

    seasonal_peak_kwh_map = {key: pd.NA for key in SEASON_GROUPS}
    seasonal_peak_kw_map = {key: pd.NA for key in SEASON_GROUPS}
    if not eligible_df.empty:
        for season_key, season_info in SEASON_GROUPS.items():
            season_df = eligible_df[eligible_df["측정일시"].dt.month.isin(season_info["months"])]
            if season_df.empty:
                continue
            season_val = season_df["전력사용량(kWh)"].max()
            if pd.isna(season_val):
                continue
            season_val = float(season_val)
            seasonal_peak_kwh_map[season_key] = season_val
            seasonal_peak_kw_map[season_key] = season_val * 4

    peak_focus_months = []
    if not eligible_df.empty:
        monthly_peak_series = (
            eligible_df.assign(__연월=eligible_df["측정일시"].dt.to_period("M"))
            .groupby("__연월")["전력사용량(kWh)"]
            .max()
            .sort_values(ascending=False)
        )
        for period_key, peak_value in monthly_peak_series.head(2).items():
            label = format_month_label(str(period_key))
            if pd.isna(peak_value):
                peak_focus_months.append(label)
                continue
            peak_kw_value = float(peak_value) * 4
            peak_focus_months.append(f"{label} ({format_number(peak_kw_value, 1)} kW)")

    st.markdown("#### 피크타임 분석")
    peak_col_left, peak_col_right = st.columns([2, 1], gap="large")

    if peak_source is None or (hasattr(peak_source, "empty") and peak_source.empty):
        with peak_col_left:
            st.info("피크 전력 데이터를 찾을 수 없습니다.")
    else:
        peak_fig = go.Figure()
        seasonal_line_colors = {key: SEASON_STYLE.get(key, {}) for key in SEASON_GROUPS}

        def add_seasonal_reference_lines(x_values):
            x_list = list(x_values)
            if not x_list:
                return
            if len(x_list) == 1:
                x_list = x_list + x_list
            for season_key, season_info in SEASON_GROUPS.items():
                season_value = seasonal_peak_kwh_map.get(season_key, pd.NA)
                if pd.isna(season_value):
                    continue
                style = seasonal_line_colors.get(season_key, {})
                line_kwargs = {
                    "color": style.get("color", "#94A3B8"),
                    "width": 1.6,
                    "dash": style.get("dash", "dot"),
                }
                peak_fig.add_trace(
                    go.Scatter(
                        x=x_list,
                        y=[season_value] * len(x_list),
                        mode="lines",
                        name=f"{season_info['label']} 최대선",
                        line=line_kwargs,
                        legendgroup=f"season_{season_key}",
                        legendrank=2 if season_key == "summer" else 3,
                        hovertemplate=f"{season_info['label']} 최대선: %{{y:.1f}} kWh<extra></extra>",
                    )
                )

        def add_peak_reference_line(x_values, y_series):
            if y_series.empty:
                return
            peak_max = y_series.max()
            if pd.isna(peak_max):
                return
            x_list = list(x_values)
            if not x_list:
                return
            if len(x_list) == 1:
                x_list = x_list + x_list
            peak_fig.add_trace(
                go.Scatter(
                    x=x_list,
                    y=[peak_max] * len(x_list),
                    mode="lines",
                    name="최대 피크",
                    line=dict(color="#DC2626", width=2.6, dash="dashdot"),
                    legendgroup="global_peak",
                    legendrank=4,
                    hovertemplate="최대 피크: %{y:.1f} kWh<extra></extra>",
                )
            )
            max_idx = y_series.idxmax()
            try:
                marker_x = x_values.iloc[max_idx]
            except AttributeError:
                marker_x = x_values[max_idx]
            marker_y = peak_max
            peak_fig.add_trace(
                go.Scatter(
                    x=[marker_x],
                    y=[marker_y],
                    mode="markers",
                    marker=dict(size=9, color="#DC2626", symbol="diamond"),
                    name="최대 피크 지점",
                    legendgroup="global_peak",
                    showlegend=False,
                    hovertemplate="최대 피크 지점<br>전력: %{y:.1f} kWh<extra></extra>",
                )
            )
        label_config = PEAK_MODE_LABELS.get(peak_mode, PEAK_MODE_LABELS["monthly"])
        if peak_mode == "monthly":
            peak_x = peak_source["월시작"]
            peak_y = peak_source["피크전력사용량(kWh)"]
            peak_fig.add_trace(
                go.Scatter(
                    x=peak_x,
                    y=peak_y,
                    mode="lines+markers",
                    name=label_config["series"],
                    line=dict(color="#14B8A6", width=2.4),
                    marker=dict(size=6),
                )
            )
            avg_series = peak_source.get("평균전력사용량(kWh)")
            if avg_series is not None and not avg_series.isna().all():
                peak_fig.add_trace(
                    go.Scatter(
                        x=peak_x,
                        y=avg_series,
                        mode="lines",
                        name=label_config["avg"],
                        line=dict(color="#6366F1", width=2, dash="dot"),
                    )
                )
            add_seasonal_reference_lines(peak_x)
            add_peak_reference_line(peak_x, peak_y)
            ticktext = peak_source.get("표시월")
            if ticktext is not None:
                ticktext = ticktext.tolist()
            else:
                ticktext = [format_month_label(str(x)) for x in peak_source["월시작"].dt.strftime("%Y-%m")]
            y_lower_bound = 120.0
            if not peak_y.isna().all():
                min_val = float(peak_y.min())
                candidate = min(120.0, min_val - 5.0)
                y_lower_bound = max(candidate, 0.0)
            peak_fig.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, t=60, b=40),
                font=dict(color="#1A202C"),
                xaxis=dict(
                    title=label_config["x_title"],
                    tickvals=peak_x,
                    ticktext=ticktext,
                    tickfont=dict(color="#1A202C"),
                ),
                yaxis=dict(
                    title="전력사용량 (kWh, 15분)",
                    tickfont=dict(color="#1A202C"),
                    range=[y_lower_bound, 160],
                ),
                legend=dict(
                    orientation="h",
                    y=1.10,
                    x=0.5,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(color="#1A202C"),
                ),
                paper_bgcolor="#F7FAFC",
                plot_bgcolor="#F7FAFC",
            )
        elif peak_mode == "daily":
            peak_x = peak_source["날짜"]
            peak_y = peak_source["피크전력사용량(kWh)"]
            peak_fig.add_trace(
                go.Scatter(
                    x=peak_x,
                    y=peak_y,
                    mode="lines+markers",
                    name=label_config["series"],
                    line=dict(color="#14B8A6", width=2.4),
                    marker=dict(size=6),
                )
            )
            avg_series = peak_source.get("평균전력사용량(kWh)")
            if avg_series is not None and not avg_series.isna().all():
                peak_fig.add_trace(
                    go.Scatter(
                        x=peak_x,
                        y=avg_series,
                        mode="lines",
                        name=label_config["avg"],
                        line=dict(color="#6366F1", width=2, dash="dot"),
                    )
            )
            add_seasonal_reference_lines(peak_x)
            add_peak_reference_line(peak_x, peak_y)
            y_lower_bound_daily = 120.0
            if not peak_y.isna().all():
                min_val = float(peak_y.min())
                candidate = min(120.0, min_val - 5.0)
                y_lower_bound_daily = max(candidate, 0.0)
            peak_fig.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, t=60, b=40),
                font=dict(color="#1A202C"),
                xaxis=dict(
                    title=label_config["x_title"],
                    tickfont=dict(color="#1A202C"),
                ),
                yaxis=dict(
                    title="전력사용량 (kWh, 15분)",
                    tickfont=dict(color="#1A202C"),
                    range=[y_lower_bound_daily, 160],
                ),
                legend=dict(
                    orientation="h",
                    y=1.10,
                    x=0.5,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(color="#1A202C"),
                ),
                paper_bgcolor="#F7FAFC",
                plot_bgcolor="#F7FAFC",
            )
        else:  # 15분 단위
            peak_x = peak_source["측정일시"]
            peak_y = peak_source["피크전력사용량(kWh)"]
            peak_fig.add_trace(
                go.Scatter(
                    x=peak_x,
                    y=peak_y,
                    mode="lines",
                    name=label_config["series"],
                    line=dict(color="#14B8A6", width=1.8),
                )
            )
            avg_kwh_line = None
            if "평균전력사용량(kWh)" in peak_source.columns and not peak_source.empty:
                avg_kwh_line = peak_source["평균전력사용량(kWh)"].iloc[0]
            if pd.notna(avg_kwh_line):
                peak_fig.add_trace(
                    go.Scatter(
                        x=peak_x,
                        y=[avg_kwh_line] * len(peak_x),
                        mode="lines",
                        name=label_config["avg"],
                        line=dict(color="#6366F1", width=2, dash="dot"),
                    )
                )
            add_seasonal_reference_lines(peak_x)
            add_peak_reference_line(peak_x, peak_y)
            y_lower_bound_15 = 120.0
            if not peak_y.isna().all():
                min_val = float(peak_y.min())
                candidate = min(120.0, min_val - 5.0)
                y_lower_bound_15 = max(candidate, 0.0)
            peak_fig.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, t=60, b=40),
                font=dict(color="#1A202C"),
                xaxis=dict(
                    title=label_config["x_title"],
                    tickfont=dict(color="#1A202C"),
                ),
                yaxis=dict(
                    title="전력사용량 (kWh, 15분)",
                    tickfont=dict(color="#1A202C"),
                    range=[y_lower_bound_15, 160],
                ),
                legend=dict(
                    orientation="h",
                    y=1.10,
                    x=0.5,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(color="#1A202C"),
                ),
                paper_bgcolor="#F7FAFC",
                plot_bgcolor="#F7FAFC",
            )
        with peak_col_left:
            st.plotly_chart(peak_fig, config={"displayModeBar": True})

    if top_records.empty:
        with peak_col_right:
            st.info("피크 전력 상위 데이터를 표시할 수 없습니다.")
    else:
        ratio = None
        if pd.notna(peak_max_value) and pd.notna(peak_avg_value) and peak_avg_value != 0:
            ratio = peak_max_value / peak_avg_value * 100
        cards_html = """
            <div style="display:flex; gap:12px; margin-bottom:12px;">
                <div class="card" style="flex:1; padding:12px 14px;">
                    <div class="card-title" style="font-size:13px;">최대 전력량 (kWh)</div>
                    <div style="font-size:24px; font-weight:600;">{max_val}</div>
                </div>
                <div class="card" style="flex:1; padding:12px 14px;">
                    <div class="card-title" style="font-size:13px;">평균 대비</div>
                    <div style="font-size:24px; font-weight:600;">{ratio_val}</div>
                </div>
            </div>
        """.format(
            max_val=format_number(peak_max_value, decimals=2),
            ratio_val=f"{ratio:,.1f}%" if ratio is not None else "-",
        )
        styled_df = top_records.copy()
        styled_df["요일"] = styled_df["측정일시"].dt.dayofweek.map(WEEKDAY_LABELS)
        styled_df["측정일시"] = styled_df["측정일시"].dt.strftime("%Y-%m-%d %H:%M")
        styled_df["전력사용량(kWh)"] = styled_df["전력사용량(kWh)"].apply(lambda v: format_number(v, 2))
        display_df = styled_df[["id", "측정일시", "요일", "전력사용량(kWh)"]].rename(
            columns={
                "id": "ID",
                "측정일시": "측정일시",
                "요일": "요일",
                "전력사용량(kWh)": "전력사용량(kWh)",
            }
        )
        table_css = """
            <style>
            .peak-table-wrap {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 14px;
                padding: 8px 10px 2px 10px;
                overflow: hidden;
            }
            .peak-table-wrap table {
                width: 100%;
                border-collapse: collapse;
                color: #1A202C;
                font-size: 13px;
            }
            .peak-table-wrap thead th {
                text-align: left;
                padding: 8px 10px;
                font-weight: 600;
                color: #6366F1;
                border-bottom: 1px solid rgba(148, 163, 184, 0.25);
            }
            .peak-table-wrap tbody td {
                padding: 10px 10px;
                border-bottom: 1px solid rgba(226, 232, 240, 0.5);
            }
            .peak-table-wrap tbody tr:last-child td {
                border-bottom: none;
            }
            </style>
        """
        table_html = display_df.to_html(index=False, border=0, classes="peak-table-table")
        with peak_col_right:
            st.markdown(cards_html, unsafe_allow_html=True)
            st.markdown(table_css, unsafe_allow_html=True)
            st.markdown(f'<div class="peak-table-wrap">{table_html}</div>', unsafe_allow_html=True)

    st.markdown("### 피크 비용 분석")
    with st.container():
        tariff_categories = list(TARIFF_BASIC_RATES.keys())
        default_category = DEFAULT_TARIFF_SELECTION[0] if DEFAULT_TARIFF_SELECTION[0] in tariff_categories else tariff_categories[0]

        config_cols = st.columns([1.3, 1.3, 1.3, 1.0], gap="large")
        with config_cols[0]:
            selected_tariff = st.selectbox(
                "요금제",
                options=tariff_categories,
                index=tariff_categories.index(default_category),
                key="tab2_peak_tariff_category",
            )

        available_voltages = list_tariff_voltages(selected_tariff)
        if not available_voltages:
            st.warning("선택한 요금제에서 전압 정보를 찾을 수 없습니다.")
            return

        default_voltage = DEFAULT_TARIFF_SELECTION[1] if DEFAULT_TARIFF_SELECTION[1] in available_voltages else available_voltages[0]
        with config_cols[1]:
            selected_voltage = st.selectbox(
                "전압 등급",
                options=available_voltages,
                index=available_voltages.index(default_voltage),
                key="tab2_peak_tariff_voltage",
            )

        option_map = list_tariff_options(selected_tariff, selected_voltage)
        option_keys = list(option_map.keys())
        option_placeholder = "-"
        if not option_keys:
            selected_option = ""
        elif len(option_keys) == 1 and option_keys[0] == "":
            selected_option = ""
        else:
            default_option = DEFAULT_TARIFF_SELECTION[2] if DEFAULT_TARIFF_SELECTION[2] in option_keys else option_keys[0]
            with config_cols[2]:
                selected_option = st.selectbox(
                    "옵션",
                    options=option_keys,
                    index=option_keys.index(default_option),
                    format_func=format_option_label,
                    key="tab2_peak_tariff_option",
                )
        if not option_map or (len(option_keys) == 1 and option_keys[0] == ""):
            with config_cols[2]:
                st.markdown("옵션")
                st.markdown("<div style='font-size:14px;color:#4A5568;'>기본</div>", unsafe_allow_html=True)

        base_rate = get_basic_rate(selected_tariff, selected_voltage, selected_option if option_keys else "")
        with config_cols[3]:
            if base_rate is not None:
                st.metric("기본요금 단가", f"{format_number(base_rate, 0)} 원/kW")
            else:
                st.metric("기본요금 단가", "-")

        if base_rate is None:
            st.warning("선택한 조건에 해당하는 기본요금 단가를 확인할 수 없습니다.")
            return

        period_peak_kwh = pd.NA
        period_avg_kwh = pd.NA
        if not range_df.empty:
            peak_val = range_df["전력사용량(kWh)"].max()
            mean_val = range_df["전력사용량(kWh)"].mean()
            if pd.notna(peak_val):
                period_peak_kwh = float(peak_val)
            if pd.notna(mean_val):
                period_avg_kwh = float(mean_val)

        billing_peak_kwh = pd.NA
        billing_peak_label = "누적 최대"
        billing_peak_display = None
        if not eligible_df.empty:
            try:
                billing_idx = eligible_df["전력사용량(kWh)"].idxmax()
            except ValueError:
                billing_idx = None
            if billing_idx is not None and pd.notna(billing_idx):
                billing_row = eligible_df.loc[billing_idx]
                billing_peak_kwh = float(billing_row["전력사용량(kWh)"])
                billing_ts = billing_row["측정일시"]
                if pd.notna(billing_ts):
                    billing_peak_display = pd.to_datetime(billing_ts).strftime("%Y-%m-%d %H:%M")
                    billing_peak_label = f"누적 최대 ({billing_peak_display})"

        current_peak_kw_for_charge = billing_peak_kwh * 4 if pd.notna(billing_peak_kwh) else pd.NA
        avg_kw_current = period_avg_kwh * 4 if pd.notna(period_avg_kwh) else pd.NA

        demand_candidates = []
        if pd.notna(current_peak_kw_for_charge):
            demand_candidates.append((billing_peak_label, current_peak_kw_for_charge))

        seasonal_peaks = seasonal_peak_kwh_map
        for season_key, season_info in SEASON_GROUPS.items():
            season_kw = seasonal_peak_kw_map.get(season_key, pd.NA)
            if pd.notna(season_kw):
                demand_candidates.append((season_info["label"], season_kw))

    if pd.notna(global_peak_kw) and global_peak_timestamp is not None:
        if effective_end_ts is None or global_peak_timestamp <= effective_end_ts:
            global_label = "연중 최대"
            if global_peak_display:
                global_label = f"{global_label} ({global_peak_display})"
            demand_candidates.append((global_label, global_peak_kw))

    if not demand_candidates:
        st.warning("청구 수요전력을 계산할 데이터를 찾을 수 없습니다.")
        return

    chargeable_label, chargeable_kw = max(demand_candidates, key=lambda item: item[1])
    peak_charge = chargeable_kw * base_rate if pd.notna(chargeable_kw) else pd.NA
    avg_charge = avg_kw_current * base_rate if pd.notna(avg_kw_current) else pd.NA
    charge_gap = pd.NA
    if pd.notna(peak_charge) and pd.notna(avg_charge):
        charge_gap = peak_charge - avg_charge

    ratio_against_avg = None
    if pd.notna(avg_kw_current) and avg_kw_current != 0 and pd.notna(chargeable_kw):
        ratio_against_avg = (chargeable_kw / avg_kw_current - 1) * 100

    def render_metric_card(column, title, value, unit, decimals=1, extra_html=""):
        with column:
            if pd.isna(value):
                display_value = "-"
            else:
                display_value = format_number(value, decimals=decimals)
            card_parts = [
                '<div class="card metric-card">',
                f'  <div class="card-title">{title}</div>',
                '  <div class="metric-body">',
                f'    <div class="metric-value">{display_value}<span class="metric-unit-inline">{unit}</span></div>',
            ]
            if extra_html:
                card_parts.append(f"    {extra_html}")
            else:
                card_parts.append('    <div class="metric-delta metric-delta-empty">&nbsp;</div>')
            card_parts.extend(["  </div>", "</div>"])
            st.markdown("\n".join(card_parts), unsafe_allow_html=True)

    card_row_one = st.columns(3, gap="large")
    render_metric_card(
        card_row_one[0],
        "기간 최대수요전력",
        period_peak_kwh,
        "kWh (15분)",
        decimals=1,
    )

    if pd.notna(period_peak_kwh) and pd.notna(period_avg_kwh) and period_peak_kwh != 0:
        diff_pct = (period_avg_kwh / period_peak_kwh - 1) * 100
        delta_class = "positive" if diff_pct > 0 else "negative" if diff_pct < 0 else "neutral"
        avg_delta_html = f'<div class="metric-delta {delta_class}">기간 최대 대비 <strong>{diff_pct:+.1f}%</strong></div>'
    else:
        avg_delta_html = ""

    render_metric_card(
        card_row_one[1],
        "평균 수요전력",
        period_avg_kwh,
        "kWh (15분)",
        decimals=1,
        extra_html=avg_delta_html,
    )

    charge_extra_html = f'<div class="metric-delta neutral">기준: {chargeable_label}</div>'
    render_metric_card(
        card_row_one[2],
        "청구 수요전력",
        chargeable_kw,
        "kW",
        decimals=1,
        extra_html=charge_extra_html,
    )

    card_row_two = st.columns(3, gap="large")
    render_metric_card(
        card_row_two[0],
        "기본요금 (청구 기준)",
        peak_charge,
        "원",
        decimals=0,
    )
    render_metric_card(
        card_row_two[1],
        "기본요금 (평균 가정)",
        avg_charge,
        "원",
        decimals=0,
    )

    gap_html = ""
    if pd.notna(charge_gap):
        if charge_gap > 0:
            gap_class = "positive"
            label = "추가 비용"
        elif charge_gap < 0:
            gap_class = "negative"
            label = "절감 효과"
        else:
            gap_class = "neutral"
            label = "변동 없음"
        gap_html = f'<div class="metric-delta {gap_class}">{label}</div>'
    render_metric_card(
        card_row_two[2],
        "피크로 인한 비용 차이",
        charge_gap,
        "원",
        decimals=0,
        extra_html=gap_html,
    )

    scenario_container = st.container()
    with scenario_container:
        st.markdown(
            "<div style='font-size:18px;font-weight:600;color:#1A202C;margin-bottom:8px;'>피크 변동 가정</div>",
            unsafe_allow_html=True,
        )
        slider_cols = st.columns([2.2, 1.8], gap="large")
        with slider_cols[0]:
            change_pct = st.slider(
                "피크 변동 가정",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                format="%+.1f%%",
                key="tab2_peak_change_pct",
                help="연간 최대 수요전력이 얼마나 변한다고 가정할지 설정하세요. 감소는 음수, 증가는 양수를 사용합니다.",
                label_visibility="collapsed",
            )
        with slider_cols[1]:
            if peak_focus_months:
                focus_text = " · ".join(peak_focus_months)
                st.markdown(
                    f"<div style='font-size:13px;color:#4A5568;line-height:1.6;'><strong>현재 데이터 기준 피크 집중 월</strong><br>{focus_text}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='font-size:13px;color:#4A5568;line-height:1.6;'>현재 데이터에서 피크 집중 월 정보를 확인할 수 없습니다.</div>",
                    unsafe_allow_html=True,
                )

    change_factor = 1 + change_pct / 100.0
    scenario_chargeable_kw = chargeable_kw * change_factor if pd.notna(chargeable_kw) else pd.NA
    scenario_billing_peak_kwh = billing_peak_kwh * change_factor if pd.notna(billing_peak_kwh) else pd.NA
    scenario_peak_charge = scenario_chargeable_kw * base_rate if pd.notna(scenario_chargeable_kw) else pd.NA

    delta_kw = pd.NA
    if pd.notna(chargeable_kw) and pd.notna(scenario_chargeable_kw):
        delta_kw = scenario_chargeable_kw - chargeable_kw

    delta_charge = pd.NA
    if pd.notna(peak_charge) and pd.notna(scenario_peak_charge):
        delta_charge = scenario_peak_charge - peak_charge

    scenario_kw_delta_html = ""
    if pd.notna(delta_kw):
        if abs(delta_kw) > 1e-6:
            kw_direction = "증가" if delta_kw > 0 else "감소"
            kw_class = "negative" if delta_kw > 0 else "positive"
            scenario_kw_delta_html = f'<div class="metric-delta {kw_class}">{kw_direction} {format_number(abs(delta_kw), 1)} kW</div>'
        else:
            scenario_kw_delta_html = '<div class="metric-delta neutral">변동 없음</div>'

    scenario_charge_delta_html = ""
    scenario_cost_effect_html = ""

    scenario_cards = st.columns(3, gap="large")
    render_metric_card(
        scenario_cards[0],
        "변동 적용 수요전력",
        scenario_chargeable_kw,
        "kW",
        decimals=1,
        extra_html=scenario_kw_delta_html,
    )
    render_metric_card(
        scenario_cards[1],
        "기본요금 (변동 가정)",
        scenario_peak_charge,
        "원",
        decimals=0,
        extra_html=scenario_charge_delta_html,
    )
    render_metric_card(
        scenario_cards[2],
        "비용 영향",
        delta_charge,
        "원",
        decimals=0,
        extra_html=scenario_cost_effect_html,
    )

    note_lines = []
    if pd.notna(billing_peak_kwh):
        if billing_peak_display:
            note_lines.append(f"- 현재 범위까지 누적된 최대수요전력(15분): {format_number(billing_peak_kwh, 1)} kWh ({billing_peak_display})")
        else:
            note_lines.append(f"- 현재 범위까지 누적된 최대수요전력(15분): {format_number(billing_peak_kwh, 1)} kWh")
    note_lines.append(f"- 청구 수요전력 기준: {chargeable_label} ({format_number(chargeable_kw, 1)} kW)")
    if pd.notna(global_peak_kw):
        if global_peak_display:
            note_lines.append(f"- 연중 최대수요전력: {format_number(global_peak_kw, 1)} kW ({global_peak_display})")
        else:
            note_lines.append(f"- 연중 최대수요전력: {format_number(global_peak_kw, 1)} kW")
    for season_key, season_info in SEASON_GROUPS.items():
        value = seasonal_peaks.get(season_key, pd.NA)
        if pd.notna(value):
            note_lines.append(f"- {season_info['label']} 최대수요전력(15분): {format_number(value, 1)} kWh")
        else:
            note_lines.append(f"- {season_info['label']} 최대수요전력: 데이터 없음")
    note_lines.append("- 청구 수요전력은 연중 최대, 하계, 동계 최대 중 가장 큰 값으로 가정했습니다.")
    if peak_focus_months:
        note_lines.append(f"- 최근 피크 집중 월: {', '.join(peak_focus_months)}")
    if not eligible_df.empty:
        eligible_start = eligible_df["측정일시"].min()
        eligible_end = eligible_df["측정일시"].max()
        note_lines.append(f"- 분석 데이터 범위: {eligible_start:%Y-%m-%d} ~ {eligible_end:%Y-%m-%d}")
    if change_pct != 0:
        if pd.notna(scenario_billing_peak_kwh):
            note_lines.append(f"- 변동 가정 최대수요전력(15분): {format_number(scenario_billing_peak_kwh, 1)} kWh")
        if pd.notna(scenario_chargeable_kw):
            direction_word = "감소" if change_pct < 0 else "증가"
            note_lines.append(f"- {abs(change_pct):.1f}% {direction_word} 시 청구 수요전력: {format_number(scenario_chargeable_kw, 1)} kW")
        if pd.notna(scenario_peak_charge):
            note_lines.append(f"- 변동 가정 기본요금: {format_number(scenario_peak_charge, 0)} 원")
        if pd.notna(delta_charge):
            if delta_charge > 0:
                note_lines.append(f"- 예상 추가 비용: {format_number(delta_charge, 0)} 원")
            elif delta_charge < 0:
                note_lines.append(f"- 예상 절감액: {format_number(-delta_charge, 0)} 원")
    else:
        note_lines.append("- 피크 변동 슬라이더로 증가·감소 시나리오를 조정하며 비용 변화를 확인할 수 있습니다.")
    if ratio_against_avg is not None:
        note_lines.append(f"- 청구 수요전력은 평균 대비 약 {ratio_against_avg:+.1f}% 차이가 있습니다.")

    note_html = "<br>".join(note_lines)
    st.markdown(
        f"""
        <div class="card" style="margin-top:12px;">
            <div class="card-title">피크 정책 참고</div>
            <div style="font-size:13px; color:#4A5568; line-height:1.6;">
                {note_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 역률 분석 & 페널티")
    with st.container():
        required_columns = {"지상역률(%)", "진상역률(%)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)"}
        if range_df.empty or not required_columns.intersection(range_df.columns):
            st.info("역률 분석을 위한 데이터를 찾을 수 없습니다.")
        else:
            pf_df = range_df.copy()
            for col in ["지상역률(%)", "진상역률(%)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)"]:
                if col in pf_df.columns:
                    pf_df[col] = pd.to_numeric(pf_df[col], errors="coerce")
            pf_df["hour"] = pf_df["측정일시"].dt.hour

            day_mask = pf_df["hour"].between(PF_DAY_START_HOUR, PF_DAY_END_HOUR - 1)
            night_mask = ~day_mask

            day_pf = pf_df.loc[day_mask, "지상역률(%)"].dropna()
            night_pf = pf_df.loc[night_mask, "진상역률(%)"].dropna()

            day_pf_capped = day_pf.clip(lower=PF_MIN_LIMIT, upper=PF_DAY_MAX_LIMIT) if not day_pf.empty else day_pf
            night_pf_capped = night_pf.clip(lower=PF_MIN_LIMIT, upper=PF_NIGHT_MAX_LIMIT) if not night_pf.empty else night_pf

            day_avg_actual = float(day_pf.mean()) if not day_pf.empty else pd.NA
            night_avg_actual = float(night_pf.mean()) if not night_pf.empty else pd.NA
            day_avg_capped = float(day_pf_capped.mean()) if not day_pf_capped.empty else pd.NA
            night_avg_capped = float(night_pf_capped.mean()) if not night_pf_capped.empty else pd.NA

            day_violation_ratio = float((day_pf < PF_DAY_THRESHOLD).mean() * 100) if not day_pf.empty else pd.NA
            night_violation_ratio = float((night_pf < PF_NIGHT_THRESHOLD).mean() * 100) if not night_pf.empty else pd.NA

            day_df = pf_df.loc[day_mask].copy()
            night_df = pf_df.loc[night_mask].copy()
            if not day_df.empty:
                day_df["month"] = day_df["측정일시"].dt.to_period("M")
                day_df["weekday"] = day_df["측정일시"].dt.weekday
            if not night_df.empty:
                night_df["month"] = night_df["측정일시"].dt.to_period("M")
                night_df["weekday"] = night_df["측정일시"].dt.weekday

            pf_plot_mode = "monthly"
            monthly_fig = None
            if not pf_df.empty:
                total_rows = len(pf_df)
                pf_span = pf_df["측정일시"].max() - pf_df["측정일시"].min()
                if total_rows <= 288:
                    pf_plot_mode = "15min"
                elif pf_span <= pd.Timedelta(days=92):
                    pf_plot_mode = "daily"

            if (not day_df.empty) or (not night_df.empty):
                monthly_fig = go.Figure()
                if pf_plot_mode == "monthly":
                    x_ticks = None
                    if not day_df.empty:
                        monthly_day = day_df.groupby("month")["지상역률(%)"].mean().dropna()
                        if not monthly_day.empty:
                            month_index = monthly_day.index.sort_values()
                            month_x = [m.to_timestamp() for m in month_index]
                            x_ticks = month_x
                            monthly_fig.add_trace(
                            go.Scatter(
                                x=month_x,
                                y=monthly_day.values,
                                mode="lines+markers",
                                name="지상 평균 역률",
                                line=dict(color="#14B8A6", width=2.4),
                                marker=dict(size=6),
                                    hovertemplate="%{x|%Y-%m} 지상 평균 %{y:.1f}%<extra></extra>",
                                )
                            )
                            monthly_fig.add_hline(
                                y=PF_DAY_THRESHOLD,
                                line=dict(color="#F97316", dash="dash"),
                                annotation_text=f"지상 기준 {PF_DAY_THRESHOLD:.0f}%",
                                annotation_position="top left",
                            )
                    if not night_df.empty:
                        monthly_night = night_df.groupby("month")["진상역률(%)"].mean().dropna()
                        if not monthly_night.empty:
                            month_index_night = monthly_night.index.sort_values()
                            month_x_night = [m.to_timestamp() for m in month_index_night]
                            if x_ticks is None:
                                x_ticks = month_x_night
                            monthly_fig.add_trace(
                            go.Scatter(
                                x=month_x_night,
                                y=monthly_night.values,
                                mode="lines+markers",
                                name="진상 평균 역률",
                                line=dict(color="#6366F1", width=2.4),
                                marker=dict(size=6),
                                    hovertemplate="%{x|%Y-%m} 진상 평균 %{y:.1f}%<extra></extra>",
                                )
                            )
                            monthly_fig.add_hline(
                                y=PF_NIGHT_THRESHOLD,
                                line=dict(color="#DC2626", dash="dot"),
                                annotation_text=f"진상 기준 {PF_NIGHT_THRESHOLD:.0f}%",
                                annotation_position="top right",
                            )
                    monthly_fig.update_layout(
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=50, b=40),
                        font=dict(color="#1A202C"),
                        xaxis=dict(
                            title="월",
                            tickmode="array" if x_ticks else "auto",
                            tickvals=x_ticks,
                            ticktext=[
                                format_month_label(pd.Timestamp(x).strftime("%Y-%m")) for x in x_ticks
                            ]
                            if x_ticks
                            else None,
                            tickfont=dict(color="#1A202C"),
                        ),
                        yaxis=dict(title="역률 (%)", tickfont=dict(color="#1A202C"), range=[50, 105]),
                        legend=dict(
                            orientation="h",
                            y=1.10,
                            x=0.5,
                            xanchor="center",
                            yanchor="bottom",
                            font=dict(color="#1A202C"),
                        ),
                        paper_bgcolor="#F7FAFC",
                        plot_bgcolor="#F7FAFC",
                    )
                elif pf_plot_mode == "daily":
                    if not day_df.empty:
                        daily_day = (
                            day_df.groupby(day_df["측정일시"].dt.date)["지상역률(%)"].mean().dropna()
                        )
                        if not daily_day.empty:
                            day_x = pd.to_datetime(daily_day.index)
                            monthly_fig.add_trace(
                            go.Scatter(
                                x=day_x,
                                y=daily_day.values,
                                mode="lines+markers",
                                name="지상 평균 역률",
                                line=dict(color="#14B8A6", width=2.4),
                                marker=dict(size=6),
                                    hovertemplate="%{x|%Y-%m-%d} 지상 평균 %{y:.1f}%<extra></extra>",
                                )
                            )
                            monthly_fig.add_hline(
                                y=PF_DAY_THRESHOLD,
                                line=dict(color="#F97316", dash="dash"),
                                annotation_text=f"지상 기준 {PF_DAY_THRESHOLD:.0f}%",
                                annotation_position="top left",
                            )
                    if not night_df.empty:
                        daily_night = (
                            night_df.groupby(night_df["측정일시"].dt.date)["진상역률(%)"].mean().dropna()
                        )
                        if not daily_night.empty:
                            night_x = pd.to_datetime(daily_night.index)
                            monthly_fig.add_trace(
                            go.Scatter(
                                x=night_x,
                                y=daily_night.values,
                                mode="lines+markers",
                                name="진상 평균 역률",
                                line=dict(color="#6366F1", width=2.4),
                                marker=dict(size=6),
                                    hovertemplate="%{x|%Y-%m-%d} 진상 평균 %{y:.1f}%<extra></extra>",
                                )
                            )
                            monthly_fig.add_hline(
                                y=PF_NIGHT_THRESHOLD,
                                line=dict(color="#DC2626", dash="dot"),
                                annotation_text=f"진상 기준 {PF_NIGHT_THRESHOLD:.0f}%",
                                annotation_position="top right",
                            )
                    monthly_fig.update_layout(
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=50, b=40),
                        font=dict(color="#1A202C"),
                        xaxis=dict(title="날짜", tickformat="%m-%d", tickfont=dict(color="#1A202C")),
                        yaxis=dict(title="역률 (%)", tickfont=dict(color="#1A202C"), range=[50, 105]),
                        legend=dict(
                            orientation="h",
                            y=1.10,
                            x=0.5,
                            xanchor="center",
                            yanchor="bottom",
                            font=dict(color="#1A202C"),
                        ),
                        paper_bgcolor="#F7FAFC",
                        plot_bgcolor="#F7FAFC",
                    )
                else:  # 15분 단위
                    if not day_df.empty:
                        day_line_df = day_df[["측정일시", "지상역률(%)"]].dropna()
                        if not day_line_df.empty:
                            monthly_fig.add_trace(
                                go.Scatter(
                                    x=day_line_df["측정일시"],
                                    y=day_line_df["지상역률(%)"],
                                    mode="lines+markers",
                                    name="지상 역률",
                                    line=dict(color="#14B8A6", width=2.4),
                                    marker=dict(size=6, color="#14B8A6"),
                                    hovertemplate="%{x|%m-%d %H:%M} 지상 %{y:.1f}%<extra></extra>",
                                )
                            )
                            monthly_fig.add_hline(
                                y=PF_DAY_THRESHOLD,
                                line=dict(color="#F97316", dash="dash"),
                                annotation_text=f"지상 기준 {PF_DAY_THRESHOLD:.0f}%",
                                annotation_position="top left",
                            )
                    if not night_df.empty:
                        night_line_df = night_df[["측정일시", "진상역률(%)"]].dropna()
                        if not night_line_df.empty:
                            monthly_fig.add_trace(
                                go.Scatter(
                                    x=night_line_df["측정일시"],
                                    y=night_line_df["진상역률(%)"],
                                    mode="lines+markers",
                                    name="진상 역률",
                                    line=dict(color="#6366F1", width=2.4),
                                    marker=dict(size=6, color="#6366F1"),
                                    hovertemplate="%{x|%m-%d %H:%M} 진상 %{y:.1f}%<extra></extra>",
                                )
                            )
                            monthly_fig.add_hline(
                                y=PF_NIGHT_THRESHOLD,
                                line=dict(color="#DC2626", dash="dot"),
                                annotation_text=f"진상 기준 {PF_NIGHT_THRESHOLD:.0f}%",
                                annotation_position="top right",
                            )
                    monthly_fig.update_layout(
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=50, b=40),
                        font=dict(color="#1A202C"),
                        xaxis=dict(title="시간", tickfont=dict(color="#1A202C"), tickformat="%m-%d %H:%M"),
                        yaxis=dict(title="역률 (%)", tickfont=dict(color="#1A202C"), range=[50, 105]),
                        legend=dict(
                            orientation="h",
                            y=1.10,
                            x=0.5,
                            xanchor="center",
                            yanchor="bottom",
                            font=dict(color="#1A202C"),
                        ),
                        paper_bgcolor="#F7FAFC",
                        plot_bgcolor="#F7FAFC",
                    )
                if not monthly_fig.data:
                    monthly_fig = None

            day_hour_series = (
                day_df.groupby("hour")["지상역률(%)"].mean().dropna() if not day_df.empty else pd.Series(dtype=float)
            )
            night_hour_series = (
                night_df.groupby("hour")["진상역률(%)"].mean().dropna() if not night_df.empty else pd.Series(dtype=float)
            )

            trend_tab_title = PF_TREND_LABELS.get(pf_plot_mode, "월별 추이")
            pf_tabs = st.tabs([trend_tab_title, "시간대 분석"])
            with pf_tabs[0]:
                if monthly_fig and monthly_fig.data:
                    st.plotly_chart(monthly_fig, config={"displayModeBar": True}, use_container_width=True)
                else:
                    if pf_plot_mode == "daily":
                        st.info("선택한 기간에 대한 일별 역률 데이터를 표시할 수 없습니다.")
                    elif pf_plot_mode == "15min":
                        st.info("선택한 기간에 대한 15분 단위 역률 데이터를 표시할 수 없습니다.")
                    else:
                        st.info("선택한 기간에 대한 월별 역률 데이터를 표시할 수 없습니다.")
            with pf_tabs[1]:
                if day_hour_series.empty and night_hour_series.empty:
                    st.info("시간대별 역률 데이터를 표시할 수 없습니다.")
                else:
                    hour_cols = st.columns(2, gap="large")
                    if not day_hour_series.empty:
                        day_colors = [
                            "#F87171" if value < PF_DAY_THRESHOLD else "#22C55E" for value in day_hour_series.values
                        ]
                        day_fig = go.Figure()
                        day_fig.add_trace(
                            go.Scatter(
                                x=day_hour_series.index,
                                y=day_hour_series.values,
                                mode="lines+markers",
                                name="지상 역률",
                                line=dict(color="#0EA5E9", width=2.2),
                                marker=dict(size=8, color=day_colors),
                                hovertemplate="%{x}시 평균 %{y:.1f}%<extra></extra>",
                            )
                        )
                        day_fig.add_hline(
                            y=PF_DAY_THRESHOLD,
                            line=dict(color="#F97316", dash="dash"),
                            annotation_text=f"기준 {PF_DAY_THRESHOLD:.0f}%",
                            annotation_position="top left",
                        )
                        day_fig.update_layout(
                            template="plotly_dark",
                            margin=dict(l=10, r=10, t=50, b=40),
                            font=dict(color="#1A202C"),
                            xaxis=dict(title="시간 (09~23시)", tickmode="linear", tick0=PF_DAY_START_HOUR, dtick=1),
                            yaxis=dict(title="역률 (%)", range=[50, 105]),
                            paper_bgcolor="#F7FAFC",
                            plot_bgcolor="#F7FAFC",
                            showlegend=False,
                        )
                        with hour_cols[0]:
                            st.plotly_chart(day_fig, config={"displayModeBar": True}, use_container_width=True)
                    else:
                        with hour_cols[0]:
                            st.info("주간 역률 데이터를 찾을 수 없습니다.")

                    if not night_hour_series.empty:
                        night_colors = [
                            "#F87171" if value < PF_NIGHT_THRESHOLD else "#22C55E"
                            for value in night_hour_series.values
                        ]
                        night_fig = go.Figure()
                        night_fig.add_trace(
                            go.Scatter(
                                x=night_hour_series.index,
                                y=night_hour_series.values,
                                mode="lines+markers",
                                name="진상 역률",
                                line=dict(color="#6366F1", width=2.2),
                                marker=dict(size=8, color=night_colors),
                                hovertemplate="%{x}시 평균 %{y:.1f}%<extra></extra>",
                            )
                        )
                        night_fig.add_hline(
                            y=PF_NIGHT_THRESHOLD,
                            line=dict(color="#DC2626", dash="dot"),
                            annotation_text=f"기준 {PF_NIGHT_THRESHOLD:.0f}%",
                            annotation_position="top left",
                        )
                        night_fig.update_layout(
                            template="plotly_dark",
                            margin=dict(l=10, r=10, t=50, b=40),
                            font=dict(color="#1A202C"),
                            xaxis=dict(title="시간 (23~08시)", tickmode="linear", tick0=0, dtick=1),
                            yaxis=dict(title="역률 (%)", range=[50, 105]),
                            paper_bgcolor="#F7FAFC",
                            plot_bgcolor="#F7FAFC",
                            showlegend=False,
                        )
                        with hour_cols[1]:
                            st.plotly_chart(night_fig, config={"displayModeBar": True}, use_container_width=True)
                    else:
                        with hour_cols[1]:
                            st.info("야간 역률 데이터를 찾을 수 없습니다.")

            basic_charge_reference = peak_charge
            if pd.isna(basic_charge_reference) and pd.notna(base_rate) and pd.notna(chargeable_kw):
                basic_charge_reference = base_rate * chargeable_kw
            if pd.isna(basic_charge_reference):
                basic_charge_reference = 0.0

            day_shortfall = max(0.0, PF_DAY_THRESHOLD - day_avg_capped) if pd.notna(day_avg_capped) else 0.0
            day_discount_pct = max(0.0, min(day_avg_capped, PF_DAY_MAX_LIMIT) - PF_DAY_THRESHOLD) if pd.notna(day_avg_capped) else 0.0
            night_shortfall = max(0.0, PF_NIGHT_THRESHOLD - night_avg_capped) if pd.notna(night_avg_capped) else 0.0

            day_penalty_amount = basic_charge_reference * PF_PENALTY_RATE_PER_PERCENT * day_shortfall if basic_charge_reference else 0.0
            day_discount_amount = basic_charge_reference * PF_PENALTY_RATE_PER_PERCENT * day_discount_pct if basic_charge_reference else 0.0
            night_penalty_amount = basic_charge_reference * PF_PENALTY_RATE_PER_PERCENT * night_shortfall if basic_charge_reference else 0.0

            total_penalty_amount = day_penalty_amount + night_penalty_amount
            total_discount_amount = day_discount_amount
            net_effect_amount = total_penalty_amount - total_discount_amount
            savings_if_compliant = total_penalty_amount

            pf_card_row_one = st.columns(4, gap="large")

            day_delta_html = ""
            if pd.notna(day_avg_actual):
                diff = day_avg_actual - PF_DAY_THRESHOLD
                if diff > 0:
                    day_delta_html = f'<div class="metric-delta positive">기준 대비 +{diff:.1f}p</div>'
                elif diff < 0:
                    day_delta_html = f'<div class="metric-delta negative">기준 대비 {diff:.1f}p</div>'
                else:
                    day_delta_html = '<div class="metric-delta neutral">기준과 동일</div>'

            night_delta_html = ""
            if pd.notna(night_avg_actual):
                diff = night_avg_actual - PF_NIGHT_THRESHOLD
                if diff > 0:
                    night_delta_html = f'<div class="metric-delta positive">기준 대비 +{diff:.1f}p</div>'
                elif diff < 0:
                    night_delta_html = f'<div class="metric-delta negative">기준 대비 {diff:.1f}p</div>'
                else:
                    night_delta_html = '<div class="metric-delta neutral">기준과 동일</div>'

            render_metric_card(
                pf_card_row_one[0],
                "지상 평균 역률",
                day_avg_actual,
                "%",
                decimals=1,
                extra_html=day_delta_html,
            )
            render_metric_card(
                pf_card_row_one[1],
                "진상 평균 역률",
                night_avg_actual,
                "%",
                decimals=1,
                extra_html=night_delta_html,
            )

            day_violation_html = ""
            if pd.notna(day_violation_ratio):
                if day_violation_ratio > 0:
                    day_violation_html = f'<div class="metric-delta negative">위반 {day_violation_ratio:.1f}%</div>'
                else:
                    day_violation_html = '<div class="metric-delta positive">위반 없음</div>'

            night_violation_html = ""
            if pd.notna(night_violation_ratio):
                if night_violation_ratio > 0:
                    night_violation_html = f'<div class="metric-delta negative">위반 {night_violation_ratio:.1f}%</div>'
                else:
                    night_violation_html = '<div class="metric-delta positive">위반 없음</div>'

            render_metric_card(
                pf_card_row_one[2],
                "지상 위반 비중",
                day_violation_ratio,
                "%",
                decimals=1,
                extra_html=day_violation_html,
            )
            render_metric_card(
                pf_card_row_one[3],
                "진상 위반 비중",
                night_violation_ratio,
                "%",
                decimals=1,
                extra_html=night_violation_html,
            )

            pf_card_row_two = st.columns(4, gap="large")
            render_metric_card(
                pf_card_row_two[0],
                "추가 요금 예상",
                total_penalty_amount,
                "원",
                decimals=0,
            )
            render_metric_card(
                pf_card_row_two[1],
                "할인 예상",
                total_discount_amount,
                "원",
                decimals=0,
            )

            net_effect_html = ""
            if net_effect_amount > 0:
                net_effect_html = '<div class="metric-delta negative">순 추가 비용</div>'
            elif net_effect_amount < 0:
                net_effect_html = '<div class="metric-delta positive">순 할인</div>'
            else:
                net_effect_html = '<div class="metric-delta neutral">변동 없음</div>'

            render_metric_card(
                pf_card_row_two[2],
                "순 영향",
                net_effect_amount,
                "원",
                decimals=0,
                extra_html=net_effect_html,
            )

            savings_html = ""
            if savings_if_compliant > 0:
                savings_html = '<div class="metric-delta positive">추가 요금 절감 여지</div>'
            render_metric_card(
                pf_card_row_two[3],
                "기준 준수 시 절감액",
                savings_if_compliant,
                "원",
                decimals=0,
                extra_html=savings_html,
            )

            pf_summary_lines = []
            if pd.notna(day_avg_capped):
                pf_summary_lines.append(f"- 지상 평균 역률(캡 적용): {format_number(day_avg_capped, 1)}% (기준 {PF_DAY_THRESHOLD:.0f}%)")
            if pd.notna(night_avg_capped):
                pf_summary_lines.append(f"- 진상 평균 역률(캡 적용): {format_number(night_avg_capped, 1)}% (기준 {PF_NIGHT_THRESHOLD:.0f}%)")
            if basic_charge_reference:
                pf_summary_lines.append(f"- 역률 요금 계산 기준 기본요금: {format_number(basic_charge_reference, 0)} 원")
            pf_summary_lines.append(
                "- 추가/할인 요금은 기준 대비 1%당 기본요금의 0.5%를 적용합니다. (지상: 09~23시, 진상: 23~09시)"
            )
            st.markdown(
                f"""
                <div class="card" style="margin-top:12px;">
                    <div class="card-title">역률 계산 참고</div>
                    <div style="font-size:13px; color:#4A5568; line-height:1.6;">
                        {"<br>".join(pf_summary_lines)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
