import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
# import os

# print(os.getcwd())
# os.chdir("..")
# 한글 폰트 설정 (Windows 기본 값: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[0]  # py파일 기준
DATA_DIR = BASE_DIR / "data"

# 데이터 로드
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
weather_df = pd.read_csv("data/청주_기상_2024년도.csv", encoding='cp949')

# 측정일시 데이터 타입 변환
train_df["측정일시"] = pd.to_datetime(train_df["측정일시"], format="%Y-%m-%d %H:%M:%S")
test_df["측정일시"] = pd.to_datetime(test_df["측정일시"], format="%Y-%m-%d %H:%M:%S")

# 모든 측정일시 데이터 년도 18년도로 변경
train_df["측정일시"] = train_df["측정일시"].apply(lambda dt: dt.replace(year=2018))
test_df["측정일시"] = test_df["측정일시"].apply(lambda dt: dt.replace(year=2018))


# 관측 단가(원/kWh) 분포 요약
eps = 1e-9  # 0 나눗셈 방지용 작은 수
unit_price = train_df["전기요금(원)"] / (train_df["전력사용량(kWh)"] + eps)
print("\nUnit price describe:\n", unit_price.describe(percentiles=[.1,.5,.9]))

unit_price = train_df["전기요금(원)"] / (train_df["전력사용량(kWh)"] + eps)

# 월별로 시간대(시)별 중앙값
month = train_df["측정일시"].dt.month
hourly_monthly_med = unit_price.groupby([train_df["측정일시"].dt.hour, month]).median().unstack()
print("\nHourly-Monthly median unit price (first 3 months):\n", hourly_monthly_med)

# 시간대의 월별 단가 변화 시각화
plt.figure(figsize=(10,6))
for m in range(1,12):
    plt.plot(hourly_monthly_med.index, hourly_monthly_med[m], label=f'Month {m}')
plt.xlabel('Hour of Day')
plt.ylabel('Median Unit Price (원/kWh)')
plt.legend(title='월', bbox_to_anchor=(1.02, 1), loc='upper left')

# 월별로 시간대(시)별 평균
hourly_monthly_mean = unit_price.groupby([train_df["측정일시"].dt.hour, month]).mean().unstack()
print("\nHourly-Monthly median unit price (first 3 months):\n", hourly_monthly_mean)

# 시간대의 월별 단가 변화 시각화
plt.figure(figsize=(10,6))
for m in range(1,12):
    plt.plot(hourly_monthly_mean.index, hourly_monthly_mean[m], label=f'Month {m}')
plt.xlabel('Hour of Day')
plt.ylabel('Median Unit Price (원/kWh)')
plt.legend(title='월', bbox_to_anchor=(1.02, 1), loc='upper left')

# 21시 파생변수 고려해보면 좋을거같음.

####################################################################################################
# 전처리
####################################################################################################

####################################################################################################
# 시간 관련 파생변수 생성
####################################################################################################
train_df["year"] = train_df["측정일시"].dt.year
train_df["month"] = train_df["측정일시"].dt.month
train_df["day"] = train_df["측정일시"].dt.day
train_df["hour"] = train_df["측정일시"].dt.hour
train_df["minute"] = train_df["측정일시"].dt.minute
train_df["second"] = train_df["측정일시"].dt.second
train_df["weekday"] = train_df["측정일시"].dt.weekday
train_df["is_weekend"] = train_df["weekday"].isin([5,6]).astype(int)
pd.set_option('display.max_rows', None)
train_df.loc[(train_df["hour"] == 12), :]

highlight_periods_2018 = [
    ("2018-01-01", "2018-01-01"),
    ("2018-02-15", "2018-02-17"),
    ("2018-03-01", "2018-03-01"),
    ("2018-05-05", "2018-05-05"),
    ("2018-05-07", "2018-05-07"),
    ("2018-05-22", "2018-05-22"),
    ("2018-06-06", "2018-06-06"),
    ("2018-06-13", "2018-06-13"),
    ("2018-08-15", "2018-08-15"),
    ("2018-09-23", "2018-09-25"),
    ("2018-09-26", "2018-09-26"),
    ("2018-10-03", "2018-10-03"),
    ("2018-10-09", "2018-10-09"),
    ("2018-12-25", "2018-12-25"),
]

highlight_dates_2018 = set()
for start, end in highlight_periods_2018:
    highlight_dates_2018.update(pd.date_range(start=start, end=end, freq="D").date)

train_df["is_special_day"] = train_df["측정일시"].dt.date.apply(lambda d: int(d in highlight_dates_2018))

# 계절 컬럼 생성
season_mapping = {
    11: 'winter', 12: 'winter', 1: 'winter',
    2: 'spring', 3: 'spring', 4: 'spring',
    5: 'summer', 6: 'summer', 7: 'summer', 8: 'summer',
    9: 'autumn', 10: 'autumn'
}
train_df["season"] = train_df["month"].map(season_mapping)

# year, second 컬럼 제거 (단일값)
train_df["year"].nunique()
train_df["second"].nunique()
train_df.drop(columns=["year", "second"], inplace=True)

####################################################################################################
# 기상 데이터 병합
####################################################################################################
weather_df.drop(columns=["지점", "지점명"], inplace=True)
weather_df["일시"] = pd.to_datetime(weather_df["일시"], format="%Y-%m-%d %H:%M")

train_df["측정일시_분"] = train_df["측정일시"].dt.floor("H")
train_df = pd.merge(train_df, weather_df, left_on="측정일시_분", right_on="일시", how="left")
train_df.drop(columns=["측정일시_분", "일시"], inplace=True)

# 결측치 재확인
train_df.isna().sum()
# 강수량 결측치는 0으로 대체
train_df["강수량(mm)"].fillna(0, inplace=True)

# 기상 데이터에서 누락 (9월19일, 9월20일)
train_df.loc[train_df.isna().sum(axis=1)>0, :]

# # 결측치 처리 - 전날 데이터로 채우기
# for idx in train_df[train_df["기온(°C)"].isna()].index:
#     prev_day = train_df.loc[idx - 24*3]
#     train_df.loc[idx, ["기온(°C)", "습도(%)", "지면온도(°C)"]] = prev_day[["기온(°C)", "습도(%)", "지면온도(°C)"]]

train_df.fillna(0, inplace=True)

train_df.to_csv('data/train_new.csv', index=False)

# # 11월 데이터 기준 train/valid 분리 저장
# train_processed = train_df[train_df["month"] != 11].copy()
# valid_processed = train_df[train_df["month"] == 11].copy()

# train_processed.to_csv(BASE_DIR / 'data' / 'processed' / 'yt' / 'v1_train_split_.csv', index=False)
# valid_processed.to_csv(BASE_DIR / 'data' / 'processed' / 'yt' / 'v1_valid_split.csv', index=False)
