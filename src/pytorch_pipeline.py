import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tab_transformer_pytorch import TabTransformer
import joblib

########################################
# 0) 환경 설정
########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
# 1) 데이터 로드
########################################
data_path = 'data/train.csv'
df = pd.read_csv(data_path)
print("데이터 로드 완료:", df.shape)

########################################
# 2) 열 제거, 결측치 처리 등 (예시)
########################################
# (예) "난자 채취 경과일", "난자 혼합 경과일" 제거
cols_to_drop_explicit = ["난자 채취 경과일", "난자 혼합 경과일"]
for c in cols_to_drop_explicit:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
print("열 제거 후:", df.shape)

# (예) DI 시술 무관 변수 결측치 0 처리
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
if set(cols_di).issubset(df.columns):
    condition = df[cols_di].isnull().all(axis=1)
    cols_to_update = cols_di + [
        'PGD 시술 여부', 'PGS 시술 여부',
        '난자 해동 경과일',
        '배아 이식 경과일',
        '배아 해동 경과일'
    ]
    for col in cols_to_update:
        if col in df.columns:
            df.loc[condition, col] = 0

for c in cols_di:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# (예) '배아 이식 경과일' NaN → 5.0
if '배아 이식 경과일' in df.columns:
    df['배아 이식 경과일'] = df['배아 이식 경과일'].fillna(5.0)

########################################
# 3) 범주형 변수 인코딩 (unknown 범주 추가)
########################################

categorical_columns = [
    "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형",
    "배란 자극 여부", "배란 유도 유형",
    "단일 배아 이식 여부", "착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인", "남성 부 불임 원인",
    "여성 주 불임 원인", "여성 부 불임 원인",
    "부부 주 불임 원인", "부부 부 불임 원인",
    "불명확 불임 원인", "불임 원인 - 난관 질환", "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
    "난자 출처", "정자 출처",
    "난자 기증자 나이", "정자 기증자 나이",
    "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부",
    "대리모 여부", "PGD 시술 여부", "PGS 시술 여부",
    "임신 성공 여부"
]

# '총 생성 배아 수'가 0이면 PGD/PGS=0
if '총 생성 배아 수' in df.columns:
    cond_zero = (df['총 생성 배아 수'] == 0)
    if 'PGD 시술 여부' in df.columns:
        df.loc[cond_zero, 'PGD 시술 여부'] = 0
    if 'PGS 시술 여부' in df.columns:
        df.loc[cond_zero, 'PGS 시술 여부'] = 0

# 불필요 열 제거 (예: "불임 원인 - 여성 요인")
if "불임 원인 - 여성 요인" in df.columns:
    df.drop(columns=["불임 원인 - 여성 요인"], inplace=True)

# unknown 범주 추가용 함수
label_encoders = {}

def add_unknown_and_fit_encoder(series: pd.Series):
    temp_s = series.copy()
    # 'unknown' 추가
    temp_s = pd.concat([temp_s, pd.Series(["unknown"])], ignore_index=True)
    le = LabelEncoder()
    le.fit(temp_s)
    return le

def transform_with_unknown(series: pd.Series, le: LabelEncoder):
    known_classes = set(le.classes_)
    def map_func(x):
        return x if x in known_classes else "unknown"
    mapped_series = series.apply(map_func)
    return le.transform(mapped_series)

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype(str)
        le = add_unknown_and_fit_encoder(df[col])
        df[col] = transform_with_unknown(df[col], le)
        label_encoders[col] = le

########################################
# 4) 결측치 비율 확인 + 특정 임계값(80%) 열 삭제
########################################
missing_counts = df.isnull().sum()
missing_ratio = (missing_counts / len(df)) * 100
threshold = 80.0
high_missing_cols = missing_ratio[missing_ratio >= threshold].index
print(f"\n결측치 80% 이상 열: {high_missing_cols.tolist()}")

print(f"열 삭제 전: {df.shape}")
df.drop(columns=high_missing_cols, inplace=True, errors='ignore')
print(f"열 삭제 후: {df.shape}")

########################################
# (데이터 검증)
########################################
print("\n==================== 데이터 검증 ====================")
print("\n[검증 A] 범주형 카디널리티 확인")
for col in categorical_columns:
    if col in df.columns and col != "임신 성공 여부":
        max_val = df[col].max()
        min_val = df[col].min()
        card_actual = int(max_val) + 1
        print(f"  - {col}: min={min_val}, max={max_val}, actual_card={card_actual}")

print("\n[검증 B] 수치형 NaN / Inf / Outlier 확인")

# 스케일링할 열 목록
cols_to_scale = [
    "총 생성 배아 수", "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수", "이식된 배아 수",
    "미세주입 배아 이식 수", "저장된 배아 수",
    "미세주입 후 저장된 배아 수", "해동된 배아 수",
    "혼합된 난자 수", "수집된 신선 난자 수",
    "저장된 신선 난자 수", "배아 이식 경과일",
    "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수"
]
cols_to_scale = [c for c in cols_to_scale if c in df.columns]

if cols_to_scale:
    print("=== df[cols_to_scale].describe() ===")
    print(df[cols_to_scale].describe())

    nan_counts = df[cols_to_scale].isnull().sum()
    print("\nNaN 개수:\n", nan_counts)

    inf_counts = df[cols_to_scale].apply(lambda s: np.isinf(s).sum())
    print("\nInf 개수:\n", inf_counts)

    for col in cols_to_scale:
        outliers = df.loc[df[col].abs() > 1e6, col]
        if len(outliers) > 0:
            print(f"  -> {col}에서 {len(outliers)}개 Outlier(>1e6) 발견.")
else:
    print("수치형 열이 없습니다.")

print("\n[검증 C] 타깃 라벨(임신 성공 여부) 값 확인")
target_vals = df["임신 성공 여부"].unique()
print("임신 성공 여부 고유값:", target_vals)
if set(target_vals) == {0, 1}:
    print("  -> 0/1만 존재합니다. ✅")
else:
    print("  -> ❗️0/1 이외 값 발견:", target_vals)

########################################
# 5) MinMaxScaler
########################################
scaler_mm = MinMaxScaler()
df[cols_to_scale] = scaler_mm.fit_transform(df[cols_to_scale])

########################################
# 6) 데이터셋 분리
########################################
target_col = "임신 성공 여부"
final_cat_cols = [c for c in categorical_columns if c in df.columns and c != target_col]
final_cont_cols = [c for c in cols_to_scale if c in df.columns]

X_cat = df[final_cat_cols].values
X_cont = df[final_cont_cols].values
y = df[target_col].values.astype(int)

X_cat_train, X_cat_temp, X_cont_train, X_cont_temp, y_train, y_temp = train_test_split(
    X_cat, X_cont, y, test_size=0.2, random_state=42, stratify=y
)
X_cat_val, X_cat_test, X_cont_val, X_cont_test, y_val, y_test = train_test_split(
    X_cat_temp, X_cont_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

########################################
# 7) PyTorch Dataset / DataLoader
########################################
class FertilityDataset(Dataset):
    def __init__(self, cat_data, cont_data, targets):
        self.cat_data = cat_data
        self.cont_data = cont_data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        cat_feats = torch.tensor(self.cat_data[idx], dtype=torch.long)
        cont_feats = torch.tensor(self.cont_data[idx], dtype=torch.float)
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return cat_feats, cont_feats, label

batch_size = 32
train_dataset = FertilityDataset(X_cat_train, X_cont_train, y_train)
val_dataset   = FertilityDataset(X_cat_val,   X_cont_val,   y_val)
test_dataset  = FertilityDataset(X_cat_test,  X_cont_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

########################################
# 8) TabTransformer 모델
########################################
cat_cardinalities = []
for i, col in enumerate(final_cat_cols):
    max_val = X_cat[:, i].max()
    cat_cardinalities.append(int(max_val) + 1)

print("\n카디널리티 목록:", cat_cardinalities)

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
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

########################################
# 9) 학습/평가 함수
########################################
def train_one_epoch(model, loader, optimizer, criterion, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    for cat_feats, cont_feats, labels in loader:
        cat_feats, cont_feats, labels = cat_feats.to(device), cont_feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(cat_feats, cont_feats)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for cat_feats, cont_feats, labels in loader:
            cat_feats, cont_feats, labels = cat_feats.to(device), cont_feats.to(device), labels.to(device)
            logits = model(cat_feats, cont_feats)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

########################################
# 10) 조기 종료(Early Stopping) 로직
########################################
patience = 3  # 개선 없는 epoch가 patience 이상이면 중단
delta = 1e-4  # 개선 판단 기준 (val_loss가 delta만큼 줄어야 개선)

best_val_loss = float('inf')
best_model_state = None
best_epoch = 0
wait_count = 0

epochs = 50  # 최대 epoch 수
for epoch in range(1, epochs+1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, clip_norm=1.0)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    improvement = best_val_loss - val_loss
    if improvement > delta:
        # 개선됨
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        best_epoch = epoch
        wait_count = 0
    else:
        # 개선 안됨
        wait_count += 1
        if wait_count >= patience:
            print(f"=> 조기 종료 발생! [epoch={best_epoch}], Val Loss={best_val_loss:.4f} 가 최적")
            break

# 만약 best_model_state가 None이거나 dict가 아니면, 마지막 모델 state_dict로 대체
if not isinstance(best_model_state, dict):
    best_model_state = model.state_dict()

model.load_state_dict(best_model_state)
print(f"Best model loaded (epoch={best_epoch}, Val Loss={best_val_loss:.4f})")

########################################
# 11) 최종 테스트 평가
########################################
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\n[Test] Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

########################################
# 12) cat_cardinalities·label_encoders·scaler_mm·model 저장
########################################
# (a) model.pth
torch.save(best_model_state, 'model.pth')
print("model.pth 저장 완료")

# (b) label_encoders.pkl
joblib.dump(label_encoders, 'label_encoders.pkl')
print("label_encoders.pkl 저장 완료")

# (c) scaler_mm.pkl
joblib.dump(scaler_mm, 'scaler_mm.pkl')
print("scaler_mm.pkl 저장 완료")

# (d) cat_cardinalities.pkl
joblib.dump(cat_cardinalities, 'cat_cardinalities.pkl')
print("cat_cardinalities.pkl 저장 완료")
