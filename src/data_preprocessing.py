import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# 1. 데이터 파일 로드
data_path = 'data/train.csv'
df = pd.read_csv(data_path)

# 2. 데이터 탐색 (EDA)
print("### 데이터 정보 ###")
print(df.info())

print("\n### 기술 통계 (수치형 변수) ###")
print(df.describe())

print("\n### 결측치 확인 ###")
pd.set_option('display.max_rows', None)  # 모든 행을 출력하도록 설정
print(df.isnull().sum())
pd.reset_option('display.max_rows')  # 설정을 원래대로 돌림

# 결측치 비율 계산 예시
missing_ratio = df.isnull().mean() * 100
print("각 항목별 결측치 비율 (%)")
pd.set_option('display.max_rows', None)  # 모든 항목을 출력하도록 설정
print(missing_ratio)
pd.reset_option('display.max_rows')  # 설정을 원래대로 돌림

# 3. 범주형 변수 인코딩
# 예시로 '시술 시기 코드', '시술 당시 나이', '시술 유형' 등 범주형 변수의 이름을 사용합니다.
categorical_columns = categorical_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "PGD 시술 여부",
    "PGS 시술 여부",
    "임신 성공 여부"
]

label_encoders = {}

for col in categorical_columns:
    # 결측치를 문자열로 변환하여 처리 (필요에 따라 다른 처리 방법 적용)
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"\n'{col}' 변수의 범주 (인코딩 전):")
    print(le.classes_)

# 4. 수치형 변수 스케일링
# 예시로 '임신 시도 또는 마지막 임신 경과 연수', '총 생성 배아 수' 등을 사용합니다.
numerical_columns = [
    "임신 시도 또는 마지막 임신 경과 연수",
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "난자 채취 경과일",
    "난자 해동 경과일",
    "난자 혼합 경과일",
    "배아 이식 경과일",
    "배아 해동 경과일"
]

# 각 수치형 변수에 대해 요약 통계, 고유값 목록과 함께 각 고유값이 몇 번 나타나는지를 보여주는 코드 예시
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"\n'{col}' 변수 요약:")
    print(df[col].describe())
    print("Unique values:")
    print(df[col].unique())
    print("Value counts:")
    print(df[col].value_counts(dropna=False))


scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"\n'{col}' 변수 요약:")
    print(df[col].describe())
    print("Unique values:")
    print(df[col].unique())


# 5. 전처리 결과 확인
print("\n### 전처리 후 데이터의 일부 출력 ###")
print(df.head())



# PGS 시술 여부 항목별 값 개수
print("### PGS 시술 여부 - 항목별 값 개수 ###")
print(df["PGS 시술 여부"].value_counts(dropna=False))

# PGD 시술 여부 항목별 값 개수
print("### PGD 시술 여부 - 항목별 값 개수 ###")
print(df["PGD 시술 여부"].value_counts(dropna=False))


# CSV 읽기 (빈 문자열 및 공백을 NaN으로 인식)
df = pd.read_csv("data/train.csv", na_values=["", " "], keep_default_na=True)

# 확인할 열 목록 (열 이름이 정확히 일치하는지 확인)
cols = [
    "단일 배아 이식 여부",
    "착상 전 유전 진단 사용 여부",
    "배아 생성 주요 이유",
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부"
]

# 이미 CSV 읽기 시 na_values를 지정했으므로, 빈 문자열는 NaN으로 처리됨

# 각 열에 대해 결측치(True/False) 마스크 생성
missing_mask = df[cols].isnull()

# 모든 지정 열에서 동시에 결측치인 행 찾기
all_missing = missing_mask.all(axis=1)

# 결과 출력
common_missing_count = all_missing.sum()
print("모든 항목에서 결측치인 행의 수:", common_missing_count)
print("해당 행의 인덱스:", df.index[all_missing].tolist())

# 디버깅: 일부 행의 원본 데이터를 출력해서 결측치가 제대로 인식되는지 확인
print("상위 10개 행:")
print(df[cols].head(10))

missing_count = df["단일 배아 이식 여부"].isnull().sum()
print("단일 배아 이식 여부의 결측치 개수:", missing_count)
