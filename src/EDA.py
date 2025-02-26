import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind


# 한글 표시를 위한 폰트 설정 (Mac의 경우 AppleGothic 사용)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 파일 로드
data_path = 'data/train.csv'
df = pd.read_csv(data_path)

# 2. 수치형 변수 처리 및 스케일링
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

# 각 수치형 변수를 숫자형으로 변환하고, 요약 통계 및 고유값, 빈도수 출력
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"\n'{col}' 변수 요약:")
    print(df[col].describe())
    print("Unique values:")
    print(df[col].unique())
    print("Value counts:")
    print(df[col].value_counts(dropna=False))
    
    # 결측치 처리 (DI 시술)
    cols = [
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
    condition = df[cols].isnull().all(axis=1)
    cols_to_update = [      # DI 시술과 무관한 항목
        '단일 배아 이식 여부',
        '착상 전 유전 검사 사용 여부',
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
        '대리모 여부',
        'PGD 시술 여부',
        'PGS 시술 여부',
        '난자 채취 경과일',
        '난자 해동 경과일',
        '난자 혼합 경과일',
        '배아 이식 경과일',
        '배아 해동 경과일'
    ]
    df.loc[condition, cols_to_update] = 0
    
    df[cols] = df[cols].fillna(0)
    df['배아 이식 경과일'] = df['배아 이식 경과일'].fillna(5.0)

# 3. 범주형 변수 처리 (모두 Label Encoding 사용)
# 범주형 변수 목록
categorical_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",   # 순서가 없는 명목형 변수지만, TabTransformer 등 임베딩 사용 시 Label Encoding으로 처리함
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
    "난자 출처",         # 원본 값 유지 (예: '본인 제공', '기증 제공', '알 수 없음')
    "정자 출처",         # 원본 값 유지 (예: '배우자 제공', '기증 제공', '미할당', '배우자 및 기증 제공')
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

# 원본 범주형 변수 복사본 저장 (분석용)
df_categorical_orig = df[categorical_columns].copy()

# 결측값을 '0.0'으로 대체 
# '총 생성 배아 수' 값이 0인 행을 찾습니다.
condition = df['총 생성 배아 수'] == 0

# 해당 조건을 만족하는 행의 'PGD 시술 여부'와 'PGS 시술 여부'를 0으로 설정합니다.
df.loc[condition, ['PGD 시술 여부', 'PGS 시술 여부']] = 0

# Label Encoding: 모든 범주형 변수를 LabelEncoder로 변환
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"'{col}' 변환 완료. 클래스: {le.classes_}")
    
# 각 범주형 변수에 대해 임신 성공 여부(0: 실패, 1: 성공)의 비율을 계산하는 코드
for col in categorical_columns:
    print(f"---- {col} ----")
    proportions = df.groupby(col)["임신 성공 여부"].apply(lambda x: x.value_counts(normalize=True)).unstack(fill_value=0)
    print(proportions)
    print("\n")

# "시술 유형"의 항목 비율 계산
type_counts = df["시술 유형"].value_counts(normalize=True) * 100

# 항목 비율 출력
print(type_counts)





########## 4. 수치형 변수 스케일링 ##########

# "임신 시도 또는 마지막 임신 경과 연수"는 연속형 변수로 Z‑score 표준화
scaler_std = StandardScaler()
df[['임신 시도 또는 마지막 임신 경과 연수']] = scaler_std.fit_transform(df[['임신 시도 또는 마지막 임신 경과 연수']])
print(df[['임신 시도 또는 마지막 임신 경과 연수']].describe())

# 나머지 수치형 변수는 MinMax Scaling (0~1 범위)
cols_to_scale = [
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",   # 범위 0~3
    "저장된 배아 수",         # 0~51, 매우 skewed
    "미세주입 후 저장된 배아 수",  # 0~51, 대부분 0
    "해동된 배아 수",         # 0~32, 대부분 0
    "혼합된 난자 수",         # 0~51
    "수집된 신선 난자 수",      # 0~51
    "저장된 신선 난자 수",      # 0~51, 거의 대부분 0
    "난자 혼합 경과일",        # 범위 0~7, 대부분 0
    "배아 이식 경과일",        # 범위 0~7
    "배아 해동 경과일",        # 범위 0~7, 대부분 0
    "파트너 정자와 혼합된 난자 수", # 0~51
    "기증자 정자와 혼합된 난자 수"  # 0~50, 대부분 0
]

scaler_mm = MinMaxScaler()
df[cols_to_scale] = scaler_mm.fit_transform(df[cols_to_scale])
print(df[cols_to_scale].describe())





# 연속형 변수 목록 (이미 전처리 및 스케일링된 변수들을 사용한다고 가정)
continuous_columns = [
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

"""
# 각 연속형 변수에 대해 그룹별 통계량과 분포 시각화 수행

for col in continuous_columns:
    print(f"==== {col} ====")
    # 그룹별 통계량 출력 (예: 임신 성공 여부별)
    stats = df.groupby("임신 성공 여부")[col].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
    print(stats)
    print("\n")
    
    # 박스 플롯 시각화
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="임신 성공 여부", y=col, data=df)
    plt.title(f"{col} - Boxplot by 임신 성공 여부")
    plt.xlabel("임신 성공 여부 (0: 실패, 1: 성공)")
    plt.ylabel(col)
    plt.show()
    
    # 변수의 표준편차를 계산해서, 너무 낮으면 kde 적용 안함.
    std_val = df[col].std()
    kde_flag = True if std_val > 0.01 else False  # 기준 값은 데이터에 따라 조정 가능
    
    try:
        plt.figure(figsize=(8, 6))
        sns.histplot(
            data=df,
            x=col,
            hue="임신 성공 여부",
            element="step",
            stat="density",
            common_norm=False,
            bins=30,
            kde=kde_flag
        )
        plt.title(f"{col} - Distribution by 임신 성공 여부")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.show()
    except np.linalg.LinAlgError as e:
        print(f"KDE error for {col}: {e}")
        # KDE에서 에러가 발생하면 kde 옵션 없이 히스토그램만 그립니다.
        plt.figure(figsize=(8, 6))
        sns.histplot(
            data=df,
            x=col,
            hue="임신 성공 여부",
            element="step",
            stat="density",
            common_norm=False,
            bins=30,
            kde=False
        )
        plt.title(f"{col} - Distribution by 임신 성공 여부 (KDE disabled)")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.show()
"""     
        
        
"""
for col in categorical_columns:
    if col not in df_categorical_orig.columns:
        print(f"Column '{col}' not found in the original categorical DataFrame.")
        continue

    # 해당 변수의 값 빈도수 계산
    counts = df_categorical_orig[col].value_counts(dropna=False)
    
    # 막대그래프 시각화
    plt.figure(figsize=(8,6))
    counts.plot(kind='bar', color='skyblue')
    plt.title(f"{col} - Bar Chart")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 파이 차트 시각화 (범례를 차트 아래쪽에 배치)
    plt.figure(figsize=(8,6))
    # pctdistance를 1.2로 설정하면 파이 조각 외부에 비율 텍스트가 표시됩니다.
    patches, texts, autotexts = plt.pie(counts, autopct='%1.1f%%', pctdistance=1.2, startangle=90, counterclock=False)
    plt.title(f"{col} - Pie Chart with Outside Labels")
    # 범례는 차트 아래쪽에 배치
    plt.legend(patches, counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.ylabel('')
    plt.tight_layout()
    plt.show()
    
    
    
    import matplotlib.pyplot as plt
    
"""


"""
# 임신 성공 집단만 필터링 (임신 성공 여부가 1인 경우)
success_df = df_categorical_orig[df_categorical_orig["임신 성공 여부"] == 1]

for col in categorical_columns:
    if col not in success_df.columns:
        print(f"Column '{col}' not found in the original categorical DataFrame.")
        continue

    # 해당 변수의 값 빈도수 계산 (성공 집단)
    counts = success_df[col].value_counts(dropna=False)
    
    # 막대그래프 시각화
    plt.figure(figsize=(8,6))
    counts.plot(kind='bar', color='skyblue')
    plt.title(f"{col} - Bar Chart (임신 성공 집단)")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 파이 차트 시각화
    plt.figure(figsize=(8,6))
    patches, texts, autotexts = plt.pie(counts, autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title(f"{col} - Pie Chart (임신 성공 집단)")
    plt.legend(patches, counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.ylabel('')  # y축 라벨 제거
    plt.tight_layout()
    plt.show()


# 연속형 변수 상관관계 히트맵
# 1. 연속형 변수들만 선택하여 새로운 DataFrame 생성
df_continuous = df[continuous_columns]

# 2. 피어슨 상관계수 계산
corr_matrix = df_continuous.corr(method='pearson')

# 3. 히트맵 시각화
plt.figure(figsize=(12, 10))  # 그림 크기 조정
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("연속형 변수 간 피어슨 상관관계 히트맵")
plt.show()
"""




"""
# 범수형 변수 상관관계 히트맵
# 1. Cramér’s V 계산 함수 정의
def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    # 행/열 개수
    r, k = confusion_matrix.shape
    # Cramér’s V 공식 적용
    if min(r, k) == 1:
        return 0
    return np.sqrt((chi2 / n) / (min(r, k) - 1))

# 2. Cramér’s V 행렬 계산
n_cols = len(categorical_columns)
cv_matrix = np.zeros((n_cols, n_cols))

# 3. 범주형 변수만 추출
df_cat = df[categorical_columns]

# 4. 각 변수 쌍에 대해 Cramér’s V 계산
for i in range(n_cols):
    for j in range(n_cols):
        x = df_cat.iloc[:, i]  # i번째 범주형 변수
        y = df_cat.iloc[:, j]  # j번째 범주형 변수
        cv = cramers_v(x, y)
        cv_matrix[i, j] = cv

# 5. 데이터프레임 형태로 변환
cv_df = pd.DataFrame(cv_matrix, index=categorical_columns, columns=categorical_columns)

# 6. 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(cv_df, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Cramér’s V Heatmap (범주형 변수 간 연관성)")
plt.show()
"""




# 범주형 변수와 임신 성공 여부 간 연관성을 카이제곱 검정(Chi-square test)으로 확인
# 주어진 범주형 변수 목록
categorical_columns = [
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
    # 마지막에 "임신 성공 여부" 포함
    "임신 성공 여부"
]

# "임신 성공 여부"를 제외한 나머지 범주형 변수들과의 카이제곱 검정
for col in categorical_columns:
    if col == "임신 성공 여부":
        continue  # 임신 성공 여부 자체는 제외
    
    print(f"\n### {col} vs. 임신 성공 여부 ###")
    
    # 교차표 생성
    contingency_table = pd.crosstab(df[col], df["임신 성공 여부"])
    
    # 카이제곱 검정 수행
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print("교차표:")
    print(contingency_table)
    print("\nChi-square statistic:", round(chi2, 4))
    print("p-value:", p_value)
    print("degrees of freedom:", dof)
    # 필요 시 expected frequencies도 확인 가능
    # print("Expected frequencies:\n", expected)
    
    # 유의수준 0.05 예시
    if p_value < 0.05:
        print("=> 통계적으로 유의한 연관성이 있음 (p < 0.05)")
    else:
        print("=> 통계적으로 유의하지 않음 (p >= 0.05)")






# 연속형 변수에 대해, 임신 성공 여부와 각 변수의 평균 차이가 유의한지 독립표본 t-검정 확인
# 임신 성공 여부 컬럼명 (0 or 1)
success_col = "임신 성공 여부"

# 연속형 변수 목록
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

alpha = 0.05  # 유의수준

# 임신 성공 그룹과 실패 그룹을 나눌 열 이름 (0 or 1)
success_group_df = df[df[success_col] == 1]
fail_group_df = df[df[success_col] == 0]

for col in numerical_columns:
    # 각 그룹에서 연속형 변수 추출, NaN 제거
    success_data = success_group_df[col].dropna()
    fail_data = fail_group_df[col].dropna()
    
    # Welch's t-test (등분산 가정 x)
    t_stat, p_value = ttest_ind(success_data, fail_data, equal_var=False)
    
    # 그룹별 평균도 함께 확인
    mean_success = success_data.mean()
    mean_fail = fail_data.mean()
    
    print(f"\n### {col} ###")
    print(f"성공 그룹 평균: {mean_success:.4f},  실패 그룹 평균: {mean_fail:.4f}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4e}")
    
    if p_value < alpha:
        print(f"=> 유의수준 {alpha}에서 두 그룹 간 평균 차이가 유의 (p < {alpha}).")
    else:
        print(f"=> 유의수준 {alpha}에서 유의하지 않음 (p >= {alpha}).")
        
        
        
    
        
    # 5. 결측치 비율 확인 및 특정 임계값 이상의 열 삭제

    # 5-1) 결측치 개수 및 비율 계산
    missing_counts = df.isnull().sum()                  # 각 열별 결측치 개수
    missing_ratio = (missing_counts / len(df)) * 100    # 결측치 비율(%)
    
    # 5-2) 결측치 비율 기준치 설정 (예: 90%)
    threshold = 80.0

    # 5-3) 기준치 이상인 열 찾기
    high_missing_cols = missing_ratio[missing_ratio >= threshold].index
    print(f"\n결측치 비율이 {threshold}% 이상인 열들:\n{high_missing_cols}")

    # 5-4) 열 삭제 전/후 shape 비교
    print(f"\n열 삭제 전 데이터프레임 shape: {df.shape}")
    df = df.drop(columns=high_missing_cols)
    print(f"열 삭제 후 데이터프레임 shape: {df.shape}")

