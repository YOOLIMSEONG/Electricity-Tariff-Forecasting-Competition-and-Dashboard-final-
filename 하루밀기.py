# build_features_train_test.py
import pandas as pd
import numpy as np
import os

# ==========================================================
# 설정
# ==========================================================
TRAIN_PATH = "./data/train.csv"
TEST_PATH  = "./data/test.csv"
SAVE_TRAIN = "./data/train2_features.csv"
SAVE_TEST  = "./data/test2_features.csv"

# ==========================================================
# 공통 함수 정의
# ==========================================================
def find_timestamp_column(df):
    candidates = ["측정일시", "측정일시_x", "측정일시_y", "timestamp", "datetime", "date"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("❌ 타임스탬프 컬럼을 찾을 수 없습니다. (예: 측정일시, 측정일시_x 등)")

def make_season(m):
    if m in [3,4,5]: return 0  # 봄
    elif m in [6,7,8]: return 1  # 여름
    elif m in [9,10,11]: return 2  # 가을
    else: return 3  # 겨울

def add_time_features(df, ts_col):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).reset_index(drop=True)

    # 기본 시간 컬럼
    df["month"] = df[ts_col].dt.month
    df["day"] = df[ts_col].dt.day
    df["hour"] = df[ts_col].dt.hour
    df["minute"] = df[ts_col].dt.minute
    df["day_of_week"] = df[ts_col].dt.dayofweek

    # sin/cos 변환
    df["sin_day"]  = np.sin(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["cos_day"]  = np.cos(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["sin_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # season
    if "season" not in df.columns:
        df["season"] = df["month"].apply(make_season)

    # is_holiday
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    return df

def clean_columns(df, ts_col):
    """최종 컬럼 구조 정리"""
    keep_cols = [
        "id", ts_col,
        "전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)",
        "지상역률(%)", "진상역률(%)",
        "작업유형", "전기요금(원)",
        "sin_day", "cos_day", "sin_week", "cos_week", "season", "is_holiday"
    ]
    return df[[c for c in keep_cols if c in df.columns]]

# ==========================================================
# 1️⃣ Train: 00시 00분 하루 뒤로 밀기 + 파생 컬럼 생성
# ==========================================================
train = pd.read_csv(TRAIN_PATH)
ts_col_train = find_timestamp_column(train)
train[ts_col_train] = pd.to_datetime(train[ts_col_train], errors="coerce")

mask = (train[ts_col_train].dt.hour == 0) & (train[ts_col_train].dt.minute == 0)
train.loc[mask, ts_col_train] = train.loc[mask, ts_col_train] + pd.Timedelta(days=1)

train = add_time_features(train, ts_col_train)
train_final = clean_columns(train, ts_col_train)
train_final.to_csv(SAVE_TRAIN, index=False, encoding="utf-8-sig")
print(f"✅ Train 전처리 완료 → {SAVE_TRAIN}")

# ==========================================================
# 2️⃣ Test: 하루 밀지 않고 파생 컬럼만 동일 생성
# ==========================================================
test = pd.read_csv(TEST_PATH)
ts_col_test = find_timestamp_column(test)
test = add_time_features(test, ts_col_test)
test_final = clean_columns(test, ts_col_test)
test_final.to_csv(SAVE_TEST, index=False, encoding="utf-8-sig")
print(f"✅ Test 전처리 완료 → {SAVE_TEST}")
