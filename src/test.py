import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tab_transformer_pytorch import TabTransformer

######################################
# 0) 환경 설정
######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################
# 1) 테스트 데이터 로드
######################################
test_df = pd.read_csv("data/test.csv")

######################################
# 2) 학습 때와 동일한 전처리 적용
######################################
# (a) 특정 열 제거
cols_to_drop_explicit = ["난자 채취 경과일", "난자 혼합 경과일"]
for c in cols_to_drop_explicit:
    if c in test_df.columns:
        test_df.drop(columns=[c], inplace=True)

# (b) DI 시술 무관 변수 결측치 -> 0 처리
cols_di = [
    '단일 배아 이식 여부',
    '착상 전 유전 진단 사용 여부',
    '배아 생성 주요 이유',
    '총 생성 배아 수',
    '미세주입된 난자 수',
    '미세주입에서 생성된 배아 수',
    '이식된 배아 수',
    '미세주입 배아 이식 수',
    '저장된 배아 수',
    '미세주입 후 저장된 배아 수',
    '해동된 배아 수',
    '해동 난자 수',
    '수집된 신선 난자 수',
    '저장된 신선 난자 수',
    '혼합된 난자 수',
    '파트너 정자와 혼합된 난자 수',
    '기증자 정자와 혼합된 난자 수',
    '동결 배아 사용 여부',
    '신선 배아 사용 여부',
    '기증 배아 사용 여부',
    '대리모 여부'
]
condition = test_df[cols_di].isnull().all(axis=1)
cols_to_update = cols_di + [
    'PGD 시술 여부', 'PGS 시술 여부',
    '난자 해동 경과일', '배아 이식 경과일', '배아 해동 경과일'
]
test_df.loc[condition, cols_to_update] = 0

test_df[cols_di] = test_df[cols_di].fillna(0)
if '배아 이식 경과일' in test_df.columns:
    test_df['배아 이식 경과일'] = test_df['배아 이식 경과일'].fillna(5.0)

######################################
# 3) label_encoders 로드
######################################
label_encoders = joblib.load('label_encoders.pkl')
print("label_encoders 로드 완료")

########################################
# (중요) 테스트 데이터 unseen label → 'unknown' 치환
########################################
def transform_with_unknown_in_test(series: pd.Series, le: LabelEncoder):
    known_classes = set(le.classes_)
    def map_func(x):
        return x if x in known_classes else "unknown"
    mapped_series = series.apply(map_func)
    return le.transform(mapped_series)

# (d) 범주형 컬럼 목록 (학습 때와 동일)
categorical_columns = [
    "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형",
    "배란 자극 여부", "배란 유도 유형", "단일 배아 이식 여부", 
    "착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", 
    "여성 부 불임 원인", "부부 주 불임 원인", "부부 부 불임 원인",
    "불명확 불임 원인", "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", 
    "불임 원인 - 배란 장애", "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인", 
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
    "배아 생성 주요 이유", "총 시술 횟수", "클리닉 내 총 시술 횟수", 
    "IVF 시술 횟수", "DI 시술 횟수", "총 임신 횟수", "IVF 임신 횟수", 
    "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
    "난자 출처", "정자 출처", "난자 기증자 나이", "정자 기증자 나이",
    "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부",
    "대리모 여부", "PGD 시술 여부", "PGS 시술 여부"
]

for col in categorical_columns:
    if col in test_df.columns:
        test_df[col] = test_df[col].astype(str)
        le = label_encoders[col]
        test_df[col] = transform_with_unknown_in_test(test_df[col], le)

######################################
# 4) scaler_mm 로드 + 스케일링
######################################
cols_to_scale = [
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
    "미세주입 후 저장된 배아 수", "해동된 배아 수", "혼합된 난자 수",
    "수집된 신선 난자 수", "저장된 신선 난자 수", "배아 이식 경과일",
    "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수"
]
scaler_mm = joblib.load('scaler_mm.pkl')
print("scaler_mm 로드 완료")

test_df[cols_to_scale] = scaler_mm.transform(test_df[cols_to_scale])

######################################
# 5) 최종 X_cat, X_cont 생성
######################################
final_cat_cols = [c for c in categorical_columns if c in test_df.columns]
final_cont_cols = [c for c in cols_to_scale if c in test_df.columns]

X_cat_test_infer = test_df[final_cat_cols].values
X_cont_test_infer = test_df[final_cont_cols].values

######################################
# 6) cat_cardinalities 로드 (or 재계산)
######################################
# (A) 만약 학습 시점에 cat_cardinalities.pkl을 저장했다면:
# cat_cardinalities = joblib.load('cat_cardinalities.pkl')

# (B) 또는, test_df 기준으로 재계산 (주의: 학습 때와 정확히 같아야 mismatch가 안 남)
cat_cardinalities = []
for i, col in enumerate(final_cat_cols):
    max_val = X_cat_test_infer[:, i].max()
    cat_cardinalities.append(int(max_val) + 1)

######################################
# 7) 모델 로드
######################################
model = TabTransformer(
    categories = cat_cardinalities,
    num_continuous = len(final_cont_cols),
    dim = 32,
    depth = 4,
    heads = 4,
    dim_head = 16,
    dim_out = 2,
    attn_dropout = 0.1,
    ff_dropout = 0.1,
)

best_model_state = torch.load('model.pth', map_location='cpu')
model.load_state_dict(best_model_state)
print("model.pth 로드 및 state_dict 주입 완료")

model.to(device)
model.eval()

######################################
# 8) 추론
######################################
with torch.no_grad():
    cat_feats = torch.tensor(X_cat_test_infer, dtype=torch.long).to(device)
    cont_feats = torch.tensor(X_cont_test_infer, dtype=torch.float).to(device)

    logits = model(cat_feats, cont_feats)  # shape: (N, 2)
    probs = nn.functional.softmax(logits, dim=1)[:, 1]  # 임신 성공(클래스=1) 확률
    pred_probs = probs.cpu().numpy()  # GPU->CPU->numpy

######################################
# 9) CSV로 저장
######################################
if "ID" in test_df.columns:
    output_df = pd.DataFrame({
        "ID": test_df["ID"],
        "probability": pred_probs
    })
else:
    # ID가 없다면, index 사용 등
    output_df = pd.DataFrame({
        "probability": pred_probs
    })

output_df.to_csv("submission.csv", index=False)
print("submission.csv 저장 완료!")
