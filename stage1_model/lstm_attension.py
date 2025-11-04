# ==========================================================
# ⚡ stage1_model_train_autolog.py
# ✅ PyTorch + MLflow Autolog + Git/GPU 추적 (Stage1 전처리모델)
# ==========================================================
import os, json, random, subprocess, warnings
warnings.filterwarnings("ignore")
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import optuna, joblib, mlflow
import matplotlib.pyplot as plt

# ==========================================================
# 기본 설정
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
SAVE_ROOT = "stage1_model"          # 타깃별 model_1, model_2 ...가 생성될 루트
os.makedirs(SAVE_ROOT, exist_ok=True)

train_path = os.path.join(DATA_DIR, "fixed_train_clean_v2.csv")
test_path  = os.path.join(DATA_DIR, "fixed_test_weather_full.csv")

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

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ==========================================================
# 🔍 Git / MLflow 설정
# ==========================================================
def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode().strip()
    except Exception:
        return "unknown"

os.environ["GIT_PYTHON_REFRESH"] = "quiet"  # Git 경고 억제
mlflow.set_experiment("Stage1_LSTM_Attention")
mlflow.autolog(log_models=True)              # ✅ 한 번만 호출

# ==========================================================
# Dataset
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
# LSTM + Attention 모델
# ==========================================================
class LSTMAttention(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        hidden  = params["units"]
        layers  = params["n_layers"]
        dropout = params["dropout"]
        n_heads = params.get("n_heads", 4)

        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = self.norm(attn_out + lstm_out)
        out = self.dropout(out)
        out = out.mean(dim=1)
        return self.fc(out).squeeze(-1)

# ==========================================================
# 전처리
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
# 학습 루프 + 그래프 저장/업로드
# ==========================================================
def train_one_fold(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, fold_name, save_dir):
    best_loss = np.inf
    train_losses, val_losses, val_maes = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
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
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                y_pred = model(Xb)
                loss = criterion(y_pred, yb)
                val_loss += loss.item() * len(Xb)
                preds.append(y_pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # epoch별 MAE (스케일된 공간) – 추세 모니터링용
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        val_mae = mean_absolute_error(y_true, y_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # MLflow 메트릭
        mlflow.log_metrics({
            f"{fold_name}_train_loss": train_loss,
            f"{fold_name}_val_loss":   val_loss,
            f"{fold_name}_val_mae":    val_mae
        }, step=epoch)

        # 곡선 저장 + MLflow 업로드
        plt.figure(figsize=(6,4))
        plt.plot(val_maes, label="Validation MAE", marker="o")
        plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title(f"{fold_name} (Epoch {epoch+1}/{epochs})")
        plt.grid(True); plt.legend(); plt.tight_layout()
        img_path = os.path.join(save_dir, f"{fold_name}_mae_curve.png")
        plt.savefig(img_path); plt.close()
        mlflow.log_artifact(img_path)

        # best 갱신
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    return best_loss, train_losses, val_losses

# ==========================================================
# ⚙️ Optuna Objective (mlflow + nested run)
# ==========================================================
def make_objective(target):
    X_full = train[feature_pool].values.astype(float)
    y_full = train[target].values.astype(float)

    def objective(trial):
        params = {
            "n_layers":  trial.suggest_int("n_layers", 1, 2),
            "units":     trial.suggest_int("units", 32, 128, step=32),
            "dropout":   trial.suggest_float("dropout", 0.1, 0.4, step=0.1),
            "n_heads":   trial.suggest_categorical("n_heads", [2, 4, 8]),
            "seq_len":   trial.suggest_int("seq_len", 12, 48, step=12),
            "lr":        trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size":trial.suggest_categorical("batch_size", [16, 32]),
            "epochs":    trial.suggest_int("epochs", 20, 40, step=10),
        }

        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        fold_mae = []
        fold_results = {}

        # 타깃별 저장 폴더 (예: stage1_model/model_1)
        model_dir = os.path.join(SAVE_ROOT, f"model_{TARGETS.index(target)+1}")
        os.makedirs(model_dir, exist_ok=True)

        # ✅ nested run으로 충돌 방지
        with mlflow.start_run(run_name=f"{target}_trial", nested=True):
            mlflow.log_params(params)

            for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_full)):
                fold_name = f"fold{fold+1}"
                x_scaler = StandardScaler(); y_scaler = StandardScaler()

                X_tr = x_scaler.fit_transform(X_full[tr_idx])
                X_va = x_scaler.transform(X_full[va_idx])
                y_tr = y_full[tr_idx].reshape(-1,1); y_va = y_full[va_idx].reshape(-1,1)
                y_tr_s = y_scaler.fit_transform(y_tr).ravel()
                y_va_s = y_scaler.transform(y_va).ravel()

                seq_len = min(params["seq_len"], max(3, len(X_tr)//3))
                train_ds   = SeqDataset(X_tr, y_tr_s, seq_len)
                val_ds     = SeqDataset(X_va, y_va_s, seq_len)
                train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=False)
                val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"], shuffle=False)

                model     = LSTMAttention(X_tr.shape[1], params).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
                criterion = nn.L1Loss()

                best_loss, tr_losses, va_losses = train_one_fold(
                    model, train_loader, val_loader, optimizer, scheduler, criterion,
                    params["epochs"], fold_name, model_dir
                )

                # 역스케일 MAE (리포팅용)
                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb = Xb.to(DEVICE)
                        preds.append(model(Xb).cpu().numpy())
                        trues.append(yb.numpy())
                y_pred = np.concatenate(preds)
                y_true = np.concatenate(trues)
                y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()
                y_true_inv = y_scaler.inverse_transform(y_true.reshape(-1,1)).ravel()
                mae = mean_absolute_error(y_true_inv, y_pred_inv)
                fold_mae.append(mae)

                # fold별 loss 기록
                fold_results[fold_name] = {"train_loss": tr_losses, "val_loss": va_losses}

                # 마지막 fold 기준으로 현재 모델/스케일러 저장(간단화)
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
                joblib.dump(x_scaler, os.path.join(model_dir, "scaler_x.pkl"))
                joblib.dump(y_scaler, os.path.join(model_dir, "scaler_y.pkl"))

            mean_mae = float(np.mean(fold_mae))
            mlflow.log_metric("mean_mae", mean_mae)

            # fold별 손실 곡선 저장
            with open(os.path.join(model_dir, "losses.json"), "w", encoding="utf-8") as f:
                json.dump(fold_results, f, indent=2, ensure_ascii=False)

        return mean_mae

    return objective

# ==========================================================
# 실행 (외부 run 1개 + 내부 nested run)
# ==========================================================
if __name__ == "__main__":
    print("📂 데이터 로드 중...")
    train = preprocess_dataframe(pd.read_csv(train_path))
    ALL_NUM = train.select_dtypes(include=[np.number]).columns.tolist()
    feature_pool = [c for c in ALL_NUM if c not in TARGETS + ["전기요금(원)"]]

    # 외부 run 1개로 전체 Stage1을 감싸고, 태그는 여기서 설정
    with mlflow.start_run(run_name="Stage1_Full_Run"):
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        mlflow.set_tag("script_path", __file__)

        results = {}
        for tgt in TARGETS:
            print(f"\n================== Target: {tgt} ==================")
            study = optuna.create_study(direction="minimize")
            study.optimize(make_objective(tgt), n_trials=N_TRIALS, show_progress_bar=True)
            results[tgt] = {"best_mae": study.best_value, "params": study.best_trial.params}
            print(f"🎯 {tgt} | Best MAE: {study.best_value:.4f}")
            print(f"🧩 Params: {study.best_trial.params}")
            # 🔹 한글이나 특수문자를 안전하게 변환
            safe_name = re.sub(r"[^a-zA-Z0-9_.\-/ ]", "_", tgt)

            # 🔹 metric (수치) 저장
            mlflow.log_metric(f"{safe_name}_best_mae", study.best_value)

            # 🔹 params (딕셔너리) 저장 — 문자열로 변환해서 태그에 저장
            mlflow.set_tag(f"{safe_name}_best_params", str(study.best_trial.params))


        with open(os.path.join(SAVE_ROOT, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n✅ 모든 Stage1 모델 학습 및 저장 완료!")
        print(f"📁 저장 위치: {SAVE_ROOT}")

###################################################################################################
# # ==========================================================
# # ⚡ stage1_model_predict.py
# # ✅ Stage1 학습된 모델로 test셋 예측 → test_with_stage1.csv 저장
# # ==========================================================
# import os, json, joblib, torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import Dataset, DataLoader
# from stage1_model_train_autolog import LSTMAttention, SeqDataset, preprocess_dataframe, TARGETS, DEVICE

# # ----------------------------------------------------------
# # 경로 설정
# # ----------------------------------------------------------
# BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR  = os.path.join(BASE_DIR, "../data")
# MODEL_DIR = os.path.join(BASE_DIR, "stage1_model")

# test_path = os.path.join(DATA_DIR, "fixed_test_weather_full.csv")
# save_path = os.path.join(DATA_DIR, "fixed_test_stage1_pred.csv")

# # ----------------------------------------------------------
# # 데이터 로드 및 전처리
# # ----------------------------------------------------------
# print("📂 test 데이터 로드 중...")
# test_df = preprocess_dataframe(pd.read_csv(test_path))
# print(f"✅ 전처리 완료 | shape = {test_df.shape}")

# # ----------------------------------------------------------
# # 숫자형 feature pool 구성
# # ----------------------------------------------------------
# ALL_NUM = test_df.select_dtypes(include=[np.number]).columns.tolist()
# feature_pool = [c for c in ALL_NUM if c not in TARGETS + ["전기요금(원)"]]
# X_full = test_df[feature_pool].values.astype(float)

# # ----------------------------------------------------------
# # 시퀀스 데이터셋 생성 함수
# # ----------------------------------------------------------
# class SimpleDataset(Dataset):
#     def __init__(self, X, seq_len):
#         self.X, self.seq_len = X, seq_len
#     def __len__(self):
#         return len(self.X) - self.seq_len
#     def __getitem__(self, idx):
#         return torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32)

# # ----------------------------------------------------------
# # 타깃별 모델 로드 및 예측
# # ----------------------------------------------------------
# for i, target in enumerate(TARGETS, 1):
#     model_dir = os.path.join(MODEL_DIR, f"model_{i}")
#     if not os.path.exists(model_dir):
#         print(f"⚠️ {target} 모델 폴더가 없습니다: {model_dir}")
#         continue

#     print(f"\n🔍 [{i}/{len(TARGETS)}] {target} 예측 시작...")

#     # 스케일러 로드
#     x_scaler = joblib.load(os.path.join(model_dir, "scaler_x.pkl"))
#     y_scaler = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

#     # feature 스케일링
#     X_scaled = x_scaler.transform(X_full)

#     # 손실 정보 로드 → seq_len 추출
#     try:
#         with open(os.path.join(model_dir, "losses.json"), "r", encoding="utf-8") as f:
#             losses = json.load(f)
#         # fold별 seq_len은 다를 수 있으므로 가장 짧은 값 사용
#         seq_len = 12
#     except:
#         seq_len = 12

#     # 모델 파라미터 복원
#     with open(os.path.join(MODEL_DIR, "metrics.json"), "r", encoding="utf-8") as f:
#         meta = json.load(f)
#     params = meta[target]["params"]

#     # 모델 생성 및 가중치 로드
#     model = LSTMAttention(X_scaled.shape[1], params).to(DEVICE)
#     model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=DEVICE))
#     model.eval()

#     # 시퀀스 기반 예측
#     ds = SimpleDataset(X_scaled, seq_len)
#     dl = DataLoader(ds, batch_size=params["batch_size"], shuffle=False)

#     preds = []
#     with torch.no_grad():
#         for Xb in tqdm(dl, desc=f"{target} predicting"):
#             Xb = Xb.to(DEVICE)
#             y_pred = model(Xb).cpu().numpy()
#             preds.append(y_pred)

#     preds = np.concatenate(preds).ravel()
#     preds_inv = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

#     # 길이 맞추기: seq_len offset 만큼 앞에 NaN
#     pred_full = np.concatenate([np.full(seq_len, np.nan), preds_inv])

#     # test_df에 컬럼 추가
#     test_df[f"{target}_pred"] = pred_full

#     print(f"✅ {target} 예측 완료 | 예측치 {np.isnan(pred_full).sum()}개 NaN, shape={len(pred_full)}")

# # ----------------------------------------------------------
# # 결과 저장
# # ----------------------------------------------------------
# test_df.to_csv(save_path, index=False, encoding="utf-8-sig")
# print(f"\n💾 저장 완료: {save_path}")
