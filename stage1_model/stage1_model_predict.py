# ==========================================================
# ⚡ stage1_model_predict.py
# ✅ Stage1 학습된 모델로 test셋 예측 → test_with_stage1.csv 저장
# ==========================================================
import os, json, joblib, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from stage1_model_train_autolog import LSTMAttention, SeqDataset, preprocess_dataframe, TARGETS, DEVICE

# ----------------------------------------------------------
# 경로 설정
# ----------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "../data")
MODEL_DIR = os.path.join(BASE_DIR, "stage1_model")

test_path = os.path.join(DATA_DIR, "fixed_test_weather_full.csv")
save_path = os.path.join(DATA_DIR, "fixed_test_stage1_pred.csv")

# ----------------------------------------------------------
# 데이터 로드 및 전처리
# ----------------------------------------------------------
print("📂 test 데이터 로드 중...")
test_df = preprocess_dataframe(pd.read_csv(test_path))
print(f"✅ 전처리 완료 | shape = {test_df.shape}")

# ----------------------------------------------------------
# 숫자형 feature pool 구성
# ----------------------------------------------------------
ALL_NUM = test_df.select_dtypes(include=[np.number]).columns.tolist()
feature_pool = [c for c in ALL_NUM if c not in TARGETS + ["전기요금(원)"]]
X_full = test_df[feature_pool].values.astype(float)

# ----------------------------------------------------------
# 시퀀스 데이터셋 생성 함수
# ----------------------------------------------------------
class SimpleDataset(Dataset):
    def __init__(self, X, seq_len):
        self.X, self.seq_len = X, seq_len
    def __len__(self):
        return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32)

# ----------------------------------------------------------
# 타깃별 모델 로드 및 예측
# ----------------------------------------------------------
for i, target in enumerate(TARGETS, 1):
    model_dir = os.path.join(MODEL_DIR, f"model_{i}")
    if not os.path.exists(model_dir):
        print(f"⚠️ {target} 모델 폴더가 없습니다: {model_dir}")
        continue

    print(f"\n🔍 [{i}/{len(TARGETS)}] {target} 예측 시작...")

    # 스케일러 로드
    x_scaler = joblib.load(os.path.join(model_dir, "scaler_x.pkl"))
    y_scaler = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

    # feature 스케일링
    X_scaled = x_scaler.transform(X_full)

    # 손실 정보 로드 → seq_len 추출
    try:
        with open(os.path.join(model_dir, "losses.json"), "r", encoding="utf-8") as f:
            losses = json.load(f)
        # fold별 seq_len은 다를 수 있으므로 가장 짧은 값 사용
        seq_len = 12
    except:
        seq_len = 12

    # 모델 파라미터 복원
    with open(os.path.join(MODEL_DIR, "metrics.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    params = meta[target]["params"]

    # 모델 생성 및 가중치 로드
    model = LSTMAttention(X_scaled.shape[1], params).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=DEVICE))
    model.eval()

    # 시퀀스 기반 예측
    ds = SimpleDataset(X_scaled, seq_len)
    dl = DataLoader(ds, batch_size=params["batch_size"], shuffle=False)

    preds = []
    with torch.no_grad():
        for Xb in tqdm(dl, desc=f"{target} predicting"):
            Xb = Xb.to(DEVICE)
            y_pred = model(Xb).cpu().numpy()
            preds.append(y_pred)

    preds = np.concatenate(preds).ravel()
    preds_inv = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    # 길이 맞추기: seq_len offset 만큼 앞에 NaN
    pred_full = np.concatenate([np.full(seq_len, np.nan), preds_inv])

    # test_df에 컬럼 추가
    test_df[f"{target}_pred"] = pred_full

    print(f"✅ {target} 예측 완료 | 예측치 {np.isnan(pred_full).sum()}개 NaN, shape={len(pred_full)}")

# ----------------------------------------------------------
# 결과 저장
# ----------------------------------------------------------
test_df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"\n💾 저장 완료: {save_path}")
