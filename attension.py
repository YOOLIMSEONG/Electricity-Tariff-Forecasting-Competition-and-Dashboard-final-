# ==========================================================
# ⚡ stage1_lstm_cv_light_torch_mlflow.py
# ✅ PyTorch + MLflow 버전 (전처리모델 + 손실기록 포함)
# ==========================================================
import os, time, json, math, random, datetime, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import optuna, joblib, mlflow

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = "preprocessing_models"
os.makedirs(BASE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ==========================================================
# 🧩 데이터셋
# ==========================================================
class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y, self.seq_len = X, y, seq_len
    def __len__(self):
        return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len], dtype=torch.float32)
        )

# ==========================================================
# 🧠 모델 정의
# ==========================================================
# ==========================================================
# 🧠 LSTM + Multi-Head Attention 통합 모델
# ==========================================================
class LSTMAttention(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        hidden = params["units"]
        layers = params["n_layers"]
        dropout = params["dropout"]
        n_heads = params.get("n_heads", 4)

        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # [B, T, H]
        # Self-Attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        # Residual + Normalize
        out = self.norm(attn_out + lstm_out)
        out = self.dropout(out)
        # Temporal pooling
        out = out.mean(dim=1)
        return self.fc(out).squeeze(-1)


# ==========================================================
# 🔧 전처리 함수
# ==========================================================
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
# 🧩 데이터 로드
# ==========================================================
print("📂 Loading...")
train = preprocess_dataframe(pd.read_csv(train_path))
ALL_NUM = train.select_dtypes(include=[np.number]).columns.tolist()
feature_pool = [c for c in ALL_NUM if c not in TARGETS + ["전기요금(원)"]]

# ==========================================================
# ⚙️ 학습 함수
# ==========================================================
def train_one_fold(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, fold_name):
    best_loss = np.inf
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()

        mlflow.log_metrics({
            f"{fold_name}_train_loss": train_loss,
            f"{fold_name}_val_loss": val_loss
        }, step=epoch)

    model.load_state_dict(best_state)
    return best_loss, train_losses, val_losses

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
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        }

        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        fold_mae = []
        fold_results = {}

        with mlflow.start_run(run_name=f"{target}_optuna_run"):
            mlflow.log_params(params)
            for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_full)):
                fold_name = f"fold{fold+1}"

                x_scaler = StandardScaler(); y_scaler = StandardScaler()
                X_tr = x_scaler.fit_transform(X_full[tr_idx])
                X_va = x_scaler.transform(X_full[va_idx])
                y_tr = y_full[tr_idx].reshape(-1,1); y_va = y_full[va_idx].reshape(-1,1)
                y_tr_s = y_scaler.fit_transform(y_tr).ravel(); y_va_s = y_scaler.transform(y_va).ravel()

                seq_len = min(params["seq_len"], len(X_tr)//3)
                train_ds = SeqDataset(X_tr, y_tr_s, seq_len)
                val_ds = SeqDataset(X_va, y_va_s, seq_len)
                train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=False)
                val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

                model = LightLSTM(X_tr.shape[1], params).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
                criterion = nn.L1Loss()

                best_loss, tr_losses, va_losses = train_one_fold(
                    model, train_loader, val_loader, optimizer, scheduler, criterion, params["epochs"], fold_name
                )

                # 예측
                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb = Xb.to(DEVICE)
                        y_pred = model(Xb).cpu().numpy()
                        preds.append(y_pred)
                        trues.append(yb.numpy())
                y_pred = np.concatenate(preds)
                y_true = np.concatenate(trues)
                y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()
                y_true_inv = y_scaler.inverse_transform(y_true.reshape(-1,1)).ravel()

                mae = mean_absolute_error(y_true_inv, y_pred_inv)
                fold_mae.append(mae)
                fold_results[fold_name] = {"train_loss": tr_losses, "val_loss": va_losses}

            mean_mae = float(np.mean(fold_mae))
            mlflow.log_metric("mean_mae", mean_mae)

            # 모델 및 전처리 저장
            model_dir = os.path.join(BASE_DIR, f"model_{TARGETS.index(target)+1}")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            joblib.dump(x_scaler, os.path.join(model_dir, "scaler_x.pkl"))
            joblib.dump(y_scaler, os.path.join(model_dir, "scaler_y.pkl"))
            with open(os.path.join(model_dir, "losses.json"), "w", encoding="utf-8") as f:
                json.dump(fold_results, f, indent=2, ensure_ascii=False)

        return mean_mae

    return objective

# ==========================================================
# 실행
# ==========================================================
if __name__ == "__main__":
    mlflow.set_experiment("Preprocessing_LSTM_PyTorch")

    for tgt in TARGETS:
        print(f"\n================== Target: {tgt} ==================")
        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(tgt), n_trials=N_TRIALS, show_progress_bar=True)
        print(f"🎯 {tgt} | Best MAE: {study.best_value:.4f}")
        print(f"🧩 Params: {study.best_trial.params}")

    print("\n✅ 모든 모델 학습 및 저장 완료")
    print(f"📁 폴더 구조: {BASE_DIR}")
    print("🧪 MLflow 대시보드 실행:  mlflow ui --port 5000")
