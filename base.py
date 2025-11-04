# ==========================================================
# ⚡ stage1_lstm_cv_light_tb_safe.py
# ✅ 안전한 TensorBoard 경로 + 시간 기록 + 경량 LSTM
# ==========================================================
import os, time, json, math, random, datetime, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import optuna, joblib

# ==========================================================
# 설정
# ==========================================================
train_path = "./data/fixed_train_clean_v2.csv"
test_path  = "./data/fixed_test_weather_full.csv"

TARGETS = [
    "전력사용량(kWh)",
    "지상무효전력량(kVarh)",
    "진상무효전력량(kVarh)",
    "지상역률(%)",
    "진상역률(%)",
]

N_SPLITS = 3
N_TRIALS = 5
SEED = 42

LOG_BASE = "logs_light_tb"
os.makedirs(LOG_BASE, exist_ok=True)
os.makedirs("models_light", exist_ok=True)
os.makedirs("artifacts_light", exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
set_seed()

# ==========================================================
# ✅ 안전한 TensorBoard 디렉토리 생성 함수
# ==========================================================
def tb_dir(target, fold):
    # 특수문자, 한글, 공백 제거 → 영문+숫자+_ 만 남김
    safe_target = re.sub(r"[^a-zA-Z0-9_]+", "_", str(target))
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(LOG_BASE, f"{safe_target}_fold{fold}_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path

# ==========================================================
# 유틸 함수
# ==========================================================
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def build_light_lstm(input_shape, params):
    inp = Input(shape=input_shape)
    x = LSTM(params["units"], return_sequences=True, dropout=params["dropout"])(inp)
    if params["n_layers"] == 2:
        x = LSTM(params["units"]//2, return_sequences=True, dropout=params["dropout"])(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(params["dropout"])(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]), loss="mae")
    return model

def preprocess_dataframe(df):
    df = df.copy()
    for col in ["측정일시_x", "측정일시_y"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.sort_values(col).reset_index(drop=True)
            break
    drop_cols = ["id", "측정일시_x", "측정일시_y"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "season" in df.columns:
        df["season"] = df["season"].map({"spring":0,"summer":1,"autumn":2,"fall":2,"winter":3}).fillna(3).astype(int)
        df = pd.get_dummies(df, columns=["season"], prefix="season")

    if "작업유형" in df.columns:
        df = pd.get_dummies(df, columns=["작업유형"], prefix="작업유형")

    if "날씨코드" in df.columns:
        le = LabelEncoder()
        df["날씨코드"] = le.fit_transform(df["날씨코드"].astype(str))

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# ==========================================================
# 데이터 로드
# ==========================================================
print("📂 Loading...")
train = preprocess_dataframe(pd.read_csv(train_path))
ALL_NUM = train.select_dtypes(include=[np.number]).columns.tolist()
feature_pool = [c for c in ALL_NUM if c not in TARGETS + ["전기요금(원)"]]

# ==========================================================
# Optuna Objective
# ==========================================================
def make_objective(target):
    X_full = train[feature_pool].values.astype(float)
    y_full = train[target].values.astype(float)

    def objective(trial):
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 2),
            "units": trial.suggest_int("units", 16, 64, step=16),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4, step=0.1),
            "seq_len": trial.suggest_int("seq_len", 12, 48, step=12),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "epochs": trial.suggest_int("epochs", 20, 40, step=10),
        }

        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        fold_mae = []
        start_time = time.time()

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_full)):
            x_scaler = StandardScaler(); y_scaler = StandardScaler()
            X_tr = x_scaler.fit_transform(X_full[tr_idx])
            X_va = x_scaler.transform(X_full[va_idx])
            y_tr = y_full[tr_idx].reshape(-1,1); y_va = y_full[va_idx].reshape(-1,1)
            y_tr_s = y_scaler.fit_transform(y_tr).ravel(); y_va_s = y_scaler.transform(y_va).ravel()

            seq_len = min(params["seq_len"], len(X_tr)//3)
            X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr_s, seq_len)
            X_va_seq, y_va_seq = create_sequences(X_va, y_va_s, seq_len)

            if len(X_tr_seq)==0 or len(X_va_seq)==0:
                return 1e9

            model = build_light_lstm((seq_len, X_tr_seq.shape[-1]), params)
            tb_logdir = tb_dir(target, fold)
            callbacks = [
                TensorBoard(log_dir=tb_logdir),
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
            ]

            print(f"🟢 [{target}] Fold {fold+1}/{N_SPLITS} start ({params['epochs']} epochs)...")
            t0 = time.time()
            model.fit(
                X_tr_seq, y_tr_seq,
                validation_data=(X_va_seq, y_va_seq),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                verbose=0,
                shuffle=False,
                callbacks=callbacks
            )
            duration = time.time() - t0
            print(f"⏱️ Fold {fold+1} 완료: {duration/60:.2f}분")

            y_hat_s = model.predict(X_va_seq, verbose=0).ravel()
            y_hat = y_scaler.inverse_transform(y_hat_s.reshape(-1,1)).ravel()
            y_true = y_scaler.inverse_transform(y_va_seq.reshape(-1,1)).ravel()
            fold_mae.append(mean_absolute_error(y_true, y_hat))

        total_time = time.time() - start_time
        print(f"⏰ 총 학습시간: {total_time/60:.2f}분")
        return float(np.mean(fold_mae))
    return objective

# ==========================================================
# 실행
# ==========================================================
metrics = {}
for tgt in TARGETS:
    print(f"\n================== Target: {tgt} ==================")
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(tgt), n_trials=N_TRIALS, show_progress_bar=True)
    metrics[tgt] = {
        "best_mae": study.best_value,
        "best_params": study.best_trial.params
    }
    print(f"🎯 {tgt} | Best MAE: {study.best_value:.4f}")
    print(f"🧩 Params: {study.best_trial.params}")

with open("artifacts_light/light_tb_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\n✅ 학습 완료! TensorBoard에서 시각화 가능.")
print(f"▶ tensorboard --logdir {LOG_BASE}")
