# max_power_pipeline.py
# 
# "최대 성능" 목표의 하이브리드 파이프라인 (LSTM + LightGBM + 스태킹 + Optuna)
# - Stage0: (옵션) 날씨 중요도 추정 (Optuna+LightGBM) → 공통 외생변수 셋 도출
# - Stage1: 5타깃 예측 (LightGBM 개별 + LSTM 멀티타깃) → 검증기반 스태킹/블렌드
# - Stage2: 전기요금(원) 예측 (LightGBM) — 입력에 Stage1 *_pred 사용
# - 모든 분할은 시간기반, 누설 방지. id/측정일시 유지. test에 날씨 실측 없으면 자동 제외
#
# 입출력
#   train: ./data/train2_features.csv   (id, 측정일시*, 타깃5 + 전기요금 + 기타피처)
#   test : ./data/test2_features.csv    (id, 측정일시*, 기타피처; 날씨는 없을 수도 있음)
#   out  : ./data/stage1_preds_max.csv (test에 5타깃 *_pred 추가)
#          ./data/stage2_submission_max.csv (id,target)
#          ./data/stage2_full_max.csv (test + target)
#
# 실행 예시:
#   python max_power_pipeline.py --stage0 --trials1 30 --trials2 40 --lstm_trials 20
#   python max_power_pipeline.py --no-stage0  (Stage0 건너뜀)
#
# 주의: 이 스크립트는 단일 파일로 전체 파이프라인을 실행합니다.
#       GPU가 있으면 LSTM이 자동으로 cuda 사용.

import os, json, argparse, warnings, math, gc
from typing import List, Tuple, Dict
from lightgbm import log_evaluation
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from lightgbm import LGBMRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# =========================
# 설정
# =========================
TRAIN_PATH = "./data/train2_features.csv"
TEST_PATH  = "./data/test2_features.csv"

STAGE1_SAVE = "./data/stage1_preds_max.csv"
STAGE2_SUB  = "./data/stage2_submission_max.csv"
STAGE2_FULL = "./data/stage2_full_max.csv"

ID_COL   = "id"
COST_COL = "전기요금(원)"
TARGET5  = [
    "전력사용량(kWh)",
    "지상무효전력량(kVarh)",
    "진상무효전력량(kVarh)",
    "지상역률(%)",
    "진상역률(%)",
]

# 후보 타임스탬프 컬럼명
TS_CANDIDATES = ["측정일시","측정일시_x","측정일시_y","timestamp","datetime","date"]

# 후보 날씨 컬럼명 (train에만 있을 수 있음)
WEATHER_CANDS = [
    "기온","기온(°C)","temperature",
    "습도","습도(%)","humidity",
    "풍속","풍속(m/s)","wind_speed",
    "기압","기압(hPa)","pressure",
    "강수량","강수량(mm)","precip","rain",
    "일사","일사(W/m2)","solar",
    "날씨코드","weather_code"
]

# 시퀀스/학습 설정
LOOKBACK  = 96
HORIZON   = 1
VAL_RATIO = 0.1
BATCH_SIZE = 256
EPOCHS     = 50
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Optuna trials 기본값
TRIALS_STAGE1 = 25   # LGBM per target
TRIALS_STAGE2 = 30   # LGBM cost
TRIALS_LSTM   = 20   # LSTM multi-target

# =========================
# 유틸
# =========================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_ts_col(df: pd.DataFrame) -> str:
    for c in TS_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError("타임스탬프 컬럼을 찾을 수 없습니다.")


def to_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == "object":
            out[c] = out[c].astype("category").cat.codes
    return out


def add_time_feats(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out["hour"]  = out[ts_col].dt.hour
    out["dow"]   = out[ts_col].dt.dayofweek
    out["month"] = out[ts_col].dt.month
    out["sin_hour"] = np.sin(2*np.pi*out["hour"]/24)
    out["cos_hour"] = np.cos(2*np.pi*out["hour"]/24)
    out["sin_dow"]  = np.sin(2*np.pi*out["dow"]/7)
    out["cos_dow"]  = np.cos(2*np.pi*out["dow"]/7)
    out["sin_year"] = np.sin(2*np.pi*out["month"]/12)
    out["cos_year"] = np.cos(2*np.pi*out["month"]/12)
    return out


def build_windows_multistep(X: np.ndarray, Y: np.ndarray, lookback=48, horizon=1) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    N = len(X)
    for t in range(lookback, N - horizon + 1):
        xs.append(X[t-lookback:t])
        fut = []
        for h in range(horizon):
            fut.append(Y[t + h])
        ys.append(np.concatenate(fut, axis=-1))
    return np.asarray(xs), np.asarray(ys)


def safe_make_test_windows(context: np.ndarray, X_te: np.ndarray, lookback=48, horizon=1, expected_rows=None) -> np.ndarray:
    ctx = np.asarray(context, dtype=np.float32)
    te  = np.asarray(X_te, dtype=np.float32)
    assert ctx.ndim == 2 and te.ndim == 2
    assert ctx.shape[1] == te.shape[1], f"feature mismatch: {ctx.shape[1]} != {te.shape[1]}"
    X_te_ctx = np.concatenate([ctx, te], axis=0)
    avail = X_te_ctx.shape[0] - lookback - horizon + 1
    if expected_rows is None:
        expected_rows = te.shape[0]
    n_win = min(expected_rows, max(0, avail))
    if n_win <= 0:
        raise ValueError(f"윈도우 불가: avail={avail}, lookback={lookback}, horizon={horizon}, expected={expected_rows}")
    X_te_win = np.stack([X_te_ctx[t-lookback:t] for t in range(lookback, lookback + n_win)], axis=0).astype(np.float32)
    if X_te_win.shape[0] != expected_rows and X_te_win.shape[0] > expected_rows:
        X_te_win = X_te_win[:expected_rows]
    return X_te_win

# =========================
# PyTorch LSTM
# =========================
class SeqDataset(Dataset):
    def __init__(self, Xw, Yw):
        self.Xw = torch.tensor(Xw, dtype=torch.float32)
        self.Yw = torch.tensor(Yw, dtype=torch.float32)
    def __len__(self): return self.Xw.shape[0]
    def __getitem__(self, i): return self.Xw[i], self.Yw[i]

class LSTMHead(nn.Module):
    def __init__(self, n_features, out_dim, h1=128, h2=64, dropout=0.2):
        super().__init__()
        self.l1 = nn.LSTM(n_features, h1, batch_first=True)
        self.do1 = nn.Dropout(dropout)
        self.l2 = nn.LSTM(h1, h2, batch_first=True)
        self.do2 = nn.Dropout(dropout)
        self.fc = nn.Linear(h2, out_dim)
    def forward(self, x):
        x,_ = self.l1(x); x = self.do1(x)
        x,_ = self.l2(x); x = self.do2(x[:, -1, :])
        return self.fc(x)

# =========================
# Optuna 목적함수들
# =========================

def lgbm_objective_factory(X: np.ndarray, y: np.ndarray, n_splits=3):
    def obj(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=250),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=32),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 3.0, log=True),
            "random_state": SEED,
            "n_jobs": -1,
            "objective": "regression",
        }
        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes = []
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            m = LGBMRegressor(**params)
            m.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="l1",
    callbacks=[log_evaluation(period=0)]
)
            p = m.predict(X_va)
            maes.append(mean_absolute_error(y_va, p))
        return float(np.mean(maes))
    return obj


def lstm_objective_factory(X: np.ndarray, Y: np.ndarray, lookback: int, horizon: int, val_ratio: float):
    # Y: (N, K)
    N = len(X)
    val_len = max(lookback + horizon, int(N * val_ratio))
    tr_end  = N - val_len

    X_tr_win, Y_tr_win = build_windows_multistep(X[:tr_end], Y[:tr_end], lookback, horizon)
    X_va_win, Y_va_win = build_windows_multistep(X[tr_end-lookback:], Y[tr_end-lookback:], lookback, horizon)

    def obj(trial):
        h1 = trial.suggest_int("h1", 64, 256, step=32)
        h2 = trial.suggest_int("h2", 32, 192, step=32)
        dr = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [128, 256, 512])
        ep = trial.suggest_int("epochs", 15, 45)

        m = LSTMHead(n_features=X.shape[1], out_dim=Y.shape[1]*horizon, h1=h1, h2=h2, dropout=dr).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        crit = nn.L1Loss()
        dl_tr = DataLoader(SeqDataset(X_tr_win, Y_tr_win), batch_size=bs, shuffle=False)
        dl_va = DataLoader(SeqDataset(X_va_win, Y_va_win), batch_size=bs, shuffle=False)

        best = np.inf; patience=0; limit=6
        for e in range(1, ep+1):
            m.train(); tl=0.0
            for xb, yb in dl_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); p = m(xb); loss = crit(p, yb); loss.backward(); opt.step()
                tl += loss.item()*xb.size(0)
            tl /= len(dl_tr.dataset)

            m.eval(); vl=0.0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    p = m(xb); loss = crit(p, yb); vl += loss.item()*xb.size(0)
            vl /= len(dl_va.dataset)

            trial.report(vl, e)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if vl + 1e-6 < best:
                best = vl; best_state = {k:v.cpu().clone() for k,v in m.state_dict().items()}; patience=0
            else:
                patience+=1
                if patience>=limit:
                    break
        if 'best_state' in locals():
            m.load_state_dict(best_state)
        # return validation loss (MAE)
        return float(best)
    return obj

# =========================
# 메인 파이프라인
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0", dest="stage0", action="store_true", help="날씨 외생변수 중요도 도출 실행")
    parser.add_argument("--no-stage0", dest="stage0", action="store_false", help="Stage0 건너뜀")
    parser.set_defaults(stage0=False)
    parser.add_argument("--trials1", type=int, default=TRIALS_STAGE1)
    parser.add_argument("--trials2", type=int, default=TRIALS_STAGE2)
    parser.add_argument("--lstm_trials", type=int, default=TRIALS_LSTM)
    args = parser.parse_args()

    set_seed(SEED)

    # ----- 데이터 로드 -----
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    ts_col = find_ts_col(train)
    assert ts_col == find_ts_col(test), "train/test 타임스탬프 컬럼명이 다릅니다."

    # 시계열 정렬, id/측정일시 유지
    test_ids = test[ID_COL].values
    train = train.sort_values(ts_col).reset_index(drop=True)
    test  = test.sort_values(ts_col).reset_index(drop=True)

    # 공통 시간 피처 추가 (원본에 보존)
    train_tf = add_time_feats(train, ts_col)
    test_tf  = add_time_feats(test, ts_col)

    # 수치화(객체형만 코드화). id/ts 제외
    train_num = to_numeric(train_tf, exclude=[ID_COL, ts_col])
    test_num  = to_numeric(test_tf,  exclude=[ID_COL, ts_col])

    # ============= Stage0: 날씨 외생변수 후보 (옵션) =============
    if args.stage0:
        print("\n[Stage0] 날씨 외생변수 중요도 도출 (LightGBM + Optuna)")
        weather_in_train = [c for c in WEATHER_CANDS if c in train_num.columns]
        exo_pool = []
        for wc in weather_in_train:
            # 타깃 wc 제외, 사용 가능한 피처 교집합만
            feat = [c for c in train_num.columns if c not in [ID_COL, ts_col] + TARGET5 + [COST_COL] + [wc]]
            X = train_num[feat].values; y = train_num[wc].values
            study = optuna.create_study(direction="minimize", pruner=SuccessiveHalvingPruner())
            study.optimize(lgbm_objective_factory(X, y, n_splits=3), n_trials=max(10, args.trials1//3), show_progress_bar=False)
            best = study.best_trial.params
            m = LGBMRegressor(**best)
            m.fit(X, y)
            imp = pd.Series(m.feature_importances_, index=feat).sort_values(ascending=False)
            topk = list(imp.head(20).index)
            exo_pool.extend(topk)
        # 중복 제거
        exo_pool = list(dict.fromkeys(exo_pool))
    else:
        exo_pool = []

    # test에 존재하지 않는 컬럼은 제외 (날씨 실측이 없을 수 있음)
    common_cols = set(train_num.columns).intersection(set(test_num.columns))
    exo_pool = [c for c in exo_pool if c in common_cols]

    # ============= Stage1: 5타깃 하이브리드 =============
    print("\n[Stage1] 5타깃: LightGBM(개별) + LSTM(멀티) + 검증기반 스태킹")

    # Stage1 입력 피처: 원본 5타깃과 전기요금 제외, id/ts 제외
    drop_X1 = [ID_COL, ts_col, COST_COL] + TARGET5
    base_feat_1 = [c for c in train_num.columns if c not in drop_X1]
    # Stage0에서 고른 exo 우선 편입 (중복 자동 제거)
    feature_cols_1 = list(dict.fromkeys(exo_pool + base_feat_1))
    feature_cols_1 = [c for c in feature_cols_1 if c in common_cols]

    X1_tr_df = train_num[feature_cols_1].copy()
    X1_te_df = test_num[feature_cols_1].copy()

    sx1 = StandardScaler()
    X1_tr = sx1.fit_transform(X1_tr_df.values)
    X1_te = sx1.transform(X1_te_df.values)

    # Target5 표준화(각각)
    Y5 = train_num[TARGET5].astype(float).values
    sy5 = [StandardScaler() for _ in range(len(TARGET5))]
    Y5_sc = np.column_stack([sy5[i].fit_transform(Y5[:, i].reshape(-1,1)).ravel() for i in range(len(TARGET5))])

    # 시간 분할
    N = len(X1_tr)
    val_len = max(LOOKBACK + HORIZON, int(N * VAL_RATIO))
    tr_end  = N - val_len

    # ---- LGBM per target (Optuna) ----
    lgbm_preds_va = []  # 검증 구간 예측 (스태킹 학습용)
    lgbm_preds_te = []  # test 예측
    for i, tgt in enumerate(TARGET5):
        print(f"  - LGBM 튜닝: {tgt}")
        y = train_num[tgt].values
        X = X1_tr
        study = optuna.create_study(direction="minimize", pruner=SuccessiveHalvingPruner())
        study.optimize(lgbm_objective_factory(X, y, n_splits=3), n_trials=args.trials1, show_progress_bar=False)
        best = study.best_trial.params
        # 최종 적합 (학습+검증 전체)
        m = LGBMRegressor(**best)
        m.fit(X[:tr_end], y[:tr_end])
        p_va = m.predict(X[tr_end:])
        p_te = m.predict(X1_te)
        lgbm_preds_va.append(p_va)
        lgbm_preds_te.append(p_te)
    lgbm_preds_va = np.column_stack(lgbm_preds_va)  # (val_len, 5)
    lgbm_preds_te = np.column_stack(lgbm_preds_te)  # (len(test), 5)

    # ---- LSTM multi-target (Optuna) ----
    print("  - LSTM 튜닝: multi-target")
    study_lstm = optuna.create_study(direction="minimize", pruner=SuccessiveHalvingPruner())
    study_lstm.optimize(
        lstm_objective_factory(X1_tr, Y5_sc, LOOKBACK, HORIZON, VAL_RATIO),
        n_trials=args.lstm_trials, show_progress_bar=False
    )
    lstm_best = study_lstm.best_trial.params
    h1 = lstm_best.get("h1",128); h2 = lstm_best.get("h2",64)
    dr = lstm_best.get("dropout",0.2); lr = lstm_best.get("lr",1e-3)
    bs = lstm_best.get("batch_size",256); ep = lstm_best.get("epochs",30)

    out_dim_1 = HORIZON * len(TARGET5)
    X1_tr_win, Y1_tr_win = build_windows_multistep(X1_tr[:tr_end], Y5_sc[:tr_end], LOOKBACK, HORIZON)
    X1_va_win, Y1_va_win = build_windows_multistep(X1_tr[tr_end-LOOKBACK:], Y5_sc[tr_end-LOOKBACK:], LOOKBACK, HORIZON)

    m1 = LSTMHead(n_features=X1_tr.shape[1], out_dim=out_dim_1, h1=h1, h2=h2, dropout=dr).to(DEVICE)
    opt1 = torch.optim.Adam(m1.parameters(), lr=lr)
    crit1 = nn.L1Loss()

    dl1_tr = DataLoader(SeqDataset(X1_tr_win, Y1_tr_win), batch_size=bs, shuffle=False)
    dl1_va = DataLoader(SeqDataset(X1_va_win, Y1_va_win), batch_size=bs, shuffle=False)

    best_val = np.inf; patience=0; limit=8
    for e in range(1, ep+1):
        m1.train(); tl=0.0
        for xb,yb in dl1_tr:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt1.zero_grad(); p=m1(xb); loss=crit1(p,yb); loss.backward(); opt1.step()
            tl += loss.item()*xb.size(0)
        tl/=len(dl1_tr.dataset)
        m1.eval(); vl=0.0
        with torch.no_grad():
            for xb,yb in dl1_va:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                p=m1(xb); loss=crit1(p,yb); vl += loss.item()*xb.size(0)
        vl/=len(dl1_va.dataset)
        print(f"    [LSTM] ep{e:02d} tr:{tl:.4f} va:{vl:.4f}")
        if vl + 1e-6 < best_val:
            best_val = vl; best_state1 = {k:v.cpu().clone() for k,v in m1.state_dict().items()}; patience=0
        else:
            patience+=1
            if patience>=limit: break
    if 'best_state1' in locals():
        m1.load_state_dict(best_state1)

    # LSTM 검증 예측 (스텝1)
    m1.eval()
    with torch.no_grad():
        va_pred1 = []
        for xb,_ in dl1_va:
            va_pred1.append(m1(xb.to(DEVICE)).cpu().numpy())
        va_pred1 = np.concatenate(va_pred1, axis=0)
    va_pred1_step1 = va_pred1[:, :len(TARGET5)]

    # 역변환하여 스태킹 학습에 사용
    va_pred1_inv = np.column_stack([
        sy5[i].inverse_transform(va_pred1_step1[:, i].reshape(-1,1)).ravel()
        for i in range(len(TARGET5))
    ])

    # LSTM test 예측
    context1 = X1_tr[-LOOKBACK:]
    X1_te_win = safe_make_test_windows(context1, X1_te, LOOKBACK, HORIZON, expected_rows=len(test))
    dl1_te = DataLoader(SeqDataset(X1_te_win, np.zeros((len(X1_te_win), out_dim_1), dtype=np.float32)), batch_size=bs, shuffle=False)
    with torch.no_grad():
        te_pred1 = []
        for xb,_ in dl1_te:
            te_pred1.append(m1(xb.to(DEVICE)).cpu().numpy())
        te_pred1 = np.concatenate(te_pred1, axis=0)
    te_pred1_step1 = te_pred1[:, :len(TARGET5)]
    te_pred1_inv = np.column_stack([
        sy5[i].inverse_transform(te_pred1_step1[:, i].reshape(-1,1)).ravel()
        for i in range(len(TARGET5))
    ])

    # ---- 스태킹: 검증구간에서 LGBM vs LSTM를 결합해 스태커(릿지)를 학습 → test에도 적용 ----
    print("  - 스태킹/블렌딩")
    # 검증구간의 정답(스텝1)
    Y_va_true = Y5[tr_end:][:va_pred1_inv.shape[0], :]

    # 스태킹 입력: [LGBM, LSTM] concat
    Z_va = np.concatenate([lgbm_preds_va, va_pred1_inv], axis=1)  # (val_len, 10)
    Z_te = np.concatenate([lgbm_preds_te, te_pred1_inv], axis=1)  # (len(test), 10)

    # 타깃별로 별도 리지 회귀로 메타모델 학습
    stack_preds_te = []
    for i, tgt in enumerate(TARGET5):
        meta = Ridge(alpha=1.0, random_state=SEED)
        meta.fit(Z_va, Y_va_true[:, i])
        p_te = meta.predict(Z_te)
        stack_preds_te.append(p_te)
    stack_preds_te = np.column_stack(stack_preds_te)  # (len(test), 5)

    # Stage1 저장 (id/측정일시 유지 + *_pred 추가)
    test_stage1 = test.copy()
    for i, tgt in enumerate(TARGET5):
        test_stage1[f"{tgt}_pred"] = stack_preds_te[:, i]
    os.makedirs(os.path.dirname(STAGE1_SAVE), exist_ok=True)
    test_stage1.to_csv(STAGE1_SAVE, index=False, encoding="utf-8-sig")
    print(f"[Stage1] 저장: {STAGE1_SAVE}")

    # ============= Stage2: 전기요금(원) (LightGBM + Optuna) =============
    print("\n[Stage2] 전기요금(원): LightGBM + Optuna (Stage1 *_pred 사용)")

    # Train에는 *_pred = 실제값 (과거만 입력되므로 누설 아님)
    train_s2 = train_num.copy()
    for tgt in TARGET5:
        train_s2[f"{tgt}_pred"] = train_s2[tgt].values

    # Test는 Stage1 예측 사용
    test_s2 = to_numeric(test_stage1, exclude=[ID_COL, ts_col])

    # 입력 피처: 원본 타깃5 제외, *_pred 포함, id/ts/COST 제외
    drop_X2 = [ID_COL, ts_col, COST_COL] + TARGET5
    feat2 = [c for c in train_s2.columns if c not in drop_X2 and c in test_s2.columns]

    X2_tr = train_s2[feat2].values
    X2_te = test_s2[feat2].values

    y_cost = train_num[COST_COL].astype(float).values

    study2 = optuna.create_study(direction="minimize", pruner=SuccessiveHalvingPruner())
    study2.optimize(lgbm_objective_factory(X2_tr, y_cost, n_splits=3), n_trials=args.trials2, show_progress_bar=False)
    best2 = study2.best_trial.params

    m2 = LGBMRegressor(**best2)
    # 시간기반 홀드아웃으로 간단 검증 찍어보기
    split2 = int(len(X2_tr)*(1-VAL_RATIO))
    m2.fit(X2_tr[:split2], y_cost[:split2])
    p2 = m2.predict(X2_tr[split2:])
    mae2 = mean_absolute_error(y_cost[split2:], p2)
    print(f"  Stage2 홀드아웃 MAE: {mae2:.4f}")

    # 전체 학습 후 test 예측
    m2.fit(X2_tr, y_cost)
    cost_pred = m2.predict(X2_te)

    # 저장 (id/측정일시 유지, 제출 파일은 id,target)
    sub = pd.DataFrame({"id": test_ids, "target": cost_pred})
    os.makedirs(os.path.dirname(STAGE2_SUB), exist_ok=True)
    sub.to_csv(STAGE2_SUB, index=False, encoding="utf-8-sig")

    test_full = test_stage1.copy()
    test_full["target"] = cost_pred
    test_full.to_csv(STAGE2_FULL, index=False, encoding="utf-8-sig")

    print(f"[Stage2] 제출 저장: {STAGE2_SUB}")
    print(f"[Stage2] 풀 저장: {STAGE2_FULL}")
    print("\n✅ 파이프라인 완료 (Stage1: 하이브리드 + 스태킹 → Stage2: LGBM)")


if __name__ == "__main__":
    main()
