# make_submission.py
# 최종 예측 CSV를 읽어서 [name(or id), target] 두 컬럼만 뽑아 제출파일로 저장

import os
import sys
import pandas as pd

# ----- 경로 설정 (원하면 argv로 받을 수 있음) -----
# 사용 예: python make_submission.py input.csv output.csv
in_path  = sys.argv[1] if len(sys.argv) > 1 else "data/final_stage2_lstm_predictions.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "data/submission_stage2.csv"

df = pd.read_csv(in_path)

# ----- 식별자 컬럼 자동 탐색: name 또는 id -----
cand_id = [c for c in df.columns if c.lower() in ("name", "id")]
if not cand_id:
    raise KeyError("식별자 컬럼(name 또는 id)을 찾지 못했습니다.")
id_col = cand_id[0]

# ----- 타깃 컬럼 자동 탐색: 'target' 또는 전기요금(원)_pred -----
cand_target = []
if "target" in df.columns:
    cand_target = ["target"]
elif "전기요금(원)_pred" in df.columns:
    cand_target = ["전기요금(원)_pred"]

if not cand_target:
    # *_pred 형식에서 하나 고르기 (최후 fallback)
    preds = [c for c in df.columns if c.endswith("_pred")]
    if not preds:
        raise KeyError("타깃 컬럼(target 또는 *_pred)을 찾지 못했습니다.")
    cand_target = [preds[0]]

tcol = cand_target[0]

# ----- 제출 형식: [name(or id), target] -----
sub = df[[id_col, tcol]].copy()
if tcol != "target":
    sub.rename(columns={tcol: "target"}, inplace=True)

# name 컬럼이 아닌 경우 id를 name으로 바꿔달라는 요청이면 아래 주석 해제
# if id_col != "name":
#     sub.rename(columns={id_col: "name"}, inplace=True)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
sub.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"✅ Saved submission: {out_path}  (cols: {sub.columns.tolist()})")
