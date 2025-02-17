from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# 1. 데이터 파일 로드
data_path = 'data/train.csv'
df = pd.read_csv(data_path)

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


import pandas as pd

# One-Hot Encoding할 열 목록
one_hot_columns = ["배란 유도 유형", "난자 출처", "정자 출처"]

# 해당 열들을 One-Hot Encoding하여 DataFrame에 추가 (기존 열은 제거)
df = pd.get_dummies(df, columns=one_hot_columns, prefix=one_hot_columns, drop_first=False)

# 결과 확인
print(df.head())
