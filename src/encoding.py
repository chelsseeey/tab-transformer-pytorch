from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# 1. 데이터 파일 로드
data_path = 'data/train.csv'
df = pd.read_csv(data_path)

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


    # 확인할 열 목록 (열 이름이 정확히 일치하는지 확인)
    cols = [
    "난자 채취 경과일",
    "난자 혼합 경과일",
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
    "대리모 여부",
    "배아 해동 경과일"
]

# 지정한 열들에서 모든 값이 결측치인 행을 찾는 조건 생성
condition = df[cols].isnull().all(axis=1)

# 해당 조건을 만족하는 행의 '배아 이식 경과일' 값을 0으로 설정
df.loc[condition, '배아 이식 경과일'] = 0

# 지정한 열들의 결측치를 0으로 채웁니다.
df[cols] = df[cols].fillna(0)

# "배아 이식 경과일"의 모든 결측치(NaN)를 5.0으로 대체
df['배아 이식 경과일'] = df['배아 이식 경과일'].fillna(5.0)


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

# "PGD 시술 여부"와 "PGS 시술 여부" 열의 결측값을 모두 문자열 '1.0'으로 대체
df[['PGD 시술 여부', 'PGS 시술 여부']] = df[['PGD 시술 여부', 'PGS 시술 여부']].fillna('1.0')



# 각 수치형 변수에 대해 요약 통계, 고유값 목록과 함께 각 고유값이 몇 번 나타나는지를 보여주는 코드 예시
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"\n'{col}' 변수 요약:")
    print(df[col].describe())
    print("Unique values:")
    print(df[col].unique())
    print("Value counts:")
    print(df[col].value_counts(dropna=False))
    
# 지정한 범주형 변수 목록에 대해, 각 변수의 고유값 목록과 해당 고유값의 빈도수를 출력하는 코드 예시
for col in categorical_columns:
    print(f"#### {col} ####")
    unique_vals = df[col].unique()
    print("Unique values:")
    print(unique_vals)
    print("Value counts:")
    print(df[col].value_counts(dropna=False))
    print("\n")


# Label Encoding할 변수 목록
label_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    # 불임 원인 관련 변수들
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
    # 시술 횟수/임신/출산 관련 변수들
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
    # 난자/정자 기증자 나이
    "난자 기증자 나이",
    "정자 기증자 나이",
    # 동결·신선·기증 배아 사용 여부 및 대리모 여부
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    # PGD, PGS 시술 여부 및 임신 성공 여부
    "PGD 시술 여부",
    "PGS 시술 여부",
    "임신 성공 여부"
]

label_encoders = {}

for col in label_columns:
    le = LabelEncoder()
    # 결측치나 기타 값 처리를 위해 먼저 문자열로 변환
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"'{col}' 변환 완료. 클래스: {le.classes_}")



# One-Hot Encoding할 열 목록
one_hot_columns = ["배란 유도 유형", "난자 출처", "정자 출처"]

# 해당 열들을 One-Hot Encoding하여 DataFrame에 추가 (기존 열은 제거)
df = pd.get_dummies(df, columns=one_hot_columns, prefix=one_hot_columns, drop_first=False)

# 결과 확인
print(df.head())



# "임신 시도 또는 마지막 임신 경과 연수" 변수를 Z‑score 표준화
# StandardScaler 객체 생성
scaler = StandardScaler()
# 해당 컬럼을 표준화하여 DataFrame에 다시 할당
df[['임신 시도 또는 마지막 임신 경과 연수']] = scaler.fit_transform(df[['임신 시도 또는 마지막 임신 경과 연수']])
# 결과 확인
print(df[['임신 시도 또는 마지막 임신 경과 연수']].describe())



# MinMax Scaling을 적용할 변수 목록
cols_to_scale = [
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",   # 범위 0~3
    "저장된 배아 수",         # 0~51, 매우 skewed
    "미세주입 후 저장된 배아 수",  # 0~51, 대부분 0
    "해동된 배아 수",         # 0~32, 대부분 0
    # "미세주입된 난자 수" 는 이미 포함되어 있음
    "혼합된 난자 수",         # 0~51
    "수집된 신선 난자 수",      # 0~51
    "저장된 신선 난자 수",      # 0~51, 거의 대부분 0
    "난자 혼합 경과일",        # 범위 0~7, 대부분 0
    "배아 이식 경과일",        # 범위 0~7
    "배아 해동 경과일",        # 범위 0~7, 대부분 0
    "파트너 정자와 혼합된 난자 수", # 0~51
    "기증자 정자와 혼합된 난자 수"  # 0~50, 대부분 0
]

# MinMaxScaler 객체 생성 후 지정한 열들에 대해 fit_transform 수행
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# 스케일링 결과 확인 (예: 각 변수의 요약 통계)
print(df[cols_to_scale].describe())