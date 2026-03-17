# 시험 성적 향상 정책 분석 보고서
## Kaggle Playground Series S6E1 — Data-Driven Policy Analysis

[![Kaggle](https://img.shields.io/badge/Kaggle-PlaygroundS6E1-blue)](https://www.kaggle.com/competitions/playground-series-s6e1)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![Score](https://img.shields.io/badge/Public%20Score-0.70504-green)]()
[![Tool](https://img.shields.io/badge/Tools-Pandas%20|%20Seaborn%20|%20XGBoost-orange)]()

---

## 프로젝트 개요

> 데이터로 성적 격차의 원인을 진단하고,  
> 경험이 아닌 증거 기반의 교육 정책을 제언합니다.

- **분석 목표** : 학생 성적에 영향을 미치는 핵심 행동 변수 도출 및 정책 제언
- **데이터셋** : Kaggle Playground Series S6E1 (exam_score 예측)
- **분석 기간** : 2025.12.23 ~ 2025.12.30

---

## Problem Definition

### 데이터 특성
- **관측치 수** : 약 25만 건의 학생 행동·생활 습관 설문 데이터
- **변수 구성** : 출석률, 학습 시간, 수면 패턴, 학습 방법 등 행동 변수 중심
- **타겟 변수** : `exam_score` (시험 점수, 연속형)
- **클래스 불균형** : 상·하위 성적 그룹 간 뚜렷한 분리 현상 존재

### 핵심 문제 인식
기존 성적 관리 정책은 **경험적 판단**이나 **일괄적 규제**에 기반하여,  
학생 개별 특성을 반영하지 못하고 사후 검증이 제한적이었습니다.

```
문제 인식 → 데이터 기반 검증 필요
경험 중심 정책의 한계  →  LMS·온라인 학습 데이터로 실증 분석
```

### 분석 질문 (Research Questions)
1. 학생 성적에 **가장 큰 영향**을 미치는 학습 행동 요인은 무엇인가?
2. 출석률과 학습 시간은 **통계적으로 유의미한** 관계를 가지는가?
3. 해당 요인을 기반으로 한 **정책 개입은 실효성**이 있는가?

---

## Data Preprocessing

### 분석 파이프라인
```
EDA (데이터 탐색)  →  변수 가공  →  통계 검정  →  정책 도출
```

### 주요 전처리 과정

| 단계 | 처리 내용 | 적용 방법 |
|------|-----------|-----------|
| 출석 그룹화 | 출석률 기반 3단계 집단 정의 | Low(~60%) / Mid(60~80%) / High(80%~) |
| 이상치 처리 | 학습 시간 극단값 제거 | IQR 기반 Outlier 탐지 |
| 변수 스케일링 | 선형 모델 적용을 위한 정규화 | StandardScaler (표준화) |
| 범주형 인코딩 | 순서형 변수 처리 | Ordinal Encoding |

### 데이터 특이사항
- 학습 시간이 증가할수록 성적도 오르지만, **일정 수준 이후 성과 증가 폭 감소** (한계 효용)
- 같은 학습 시간이라도 성적 편차 존재 → **'질(Quality)'의 문제** 개입 확인
- 예측 모델을 통해 `High Effort / Low Output` 학생군 식별 가능

---

## 핵심 분석 결과

### 성적을 결정하는 변수 (Feature Importance)
![feature_importance_chart](output/feature_importance_chart.html)

| 순위 | 변수 | 중요도 | 설명 |
|------|------|--------|------|
| 1 | 학습 시간 (Study Hours) | 0.62 | 가장 강력한 예측 변수 |
| 2 | 수업 출석률 (Attendance) | 0.15 | 정책 개입 가능 영역 |
| 3 | 수면의 질 (Sleep Quality) | 0.07 | 생활 습관 변수 |
| 4 | 학습 방법 (Study Method) | 0.06 | 질적 개선 영역 |

> 개인적 특성(수면 시간 등)보다 **능동적 행동(학습·출석)**이 성적을 더 잘 설명함

### 출석 그룹별 성적 차이 (ANOVA + Tukey 사후검정)

- High vs Low : 평균 차이 **-15.45** (p < 0.001)
- Low vs Mid  : 평균 차이 **+9.25**  (p < 0.001)
- **핵심 구간 (Key Zone) : Low → Mid** — 가장 시급하고 효과적인 개입 포인트

---

## 모델링

### 머신러닝 모델 비교

| 모델 | AUC-ROC | Accuracy | Recall | F1-Score |
|------|---------|----------|--------|----------|
| Logistic Regression | 0.693 | 0.626 | 0.593 | 0.664 |
| Decision Tree | 0.694 | 0.630 | 0.606 | 0.672 |
| **XGBoost (최종)** | **0.721** | **0.681** | **0.845** | **0.768** |
| LightGBM | 0.725 | 0.654 | 0.630 | 0.694 |

- **최종 선택 모델 : XGBoost**
- 선정 이유 : 가장 높은 Recall 및 F1-score, 조기 위험군 선별 목적에 최적
- Kaggle Public Score : **0.70504**
- 검증 방식 : 5-Fold Cross Validation / RMSE : 8.78 / 표준편차 : 0.012

---

## 3대 정책 제언

### 1. 조기 개입형 출석 관리 전략
- **대상** : 출석 위험군 (Low Attendance Group)
- **액션** : Early Warning System 가동
- **기대 효과** : Low → Mid 전환 시 평균 성적 +16% 향상

### 2. 학습 총량 관리제
- **정책** : 성적대별 '최적 학습 시간 구간' 도출 및 제시
- **액션** : High Effort / Low Output 학생 대상 학습법 멘토링 제공

### 3. 데이터 기반 학습 피드백 시스템
- 출석률 · 학습 시간 · 예측 성적을 포함한 **개인별 데이터 리포트** 제공
- Measure → Feedback → Adjust 사이클 구축

---

## Conclusion

### 분석 요약

> 설문 기반 행동 데이터만으로도 학생 성적 예측 및 위험군 선별이 가능함을 확인하였습니다.

| 구분 | 내용 |
|------|------|
| 핵심 발견 | 학습 시간(0.62)·출석률(0.15)이 성적 결정의 77%를 설명 |
| 통계 검증 | 출석 그룹 간 성적 차이 모두 p < 0.001로 유의미 |
| 모델 성능 | XGBoost Recall 0.845 — 위험군 선별 정확도 높음 |
| 정책 실효성 | Low → Mid 출석 개선 시 평균 성적 +16% 기대 |

### 한계점 및 향후 과제
- 설문 기반 데이터 특성상 **자기보고 편향(Self-report Bias)** 가능성 존재
- 실제 의료·교육 현장 적용을 위해서는 **종단 데이터(Longitudinal Data)** 필요
- 향후 **LMS 실시간 데이터**와 연동 시 조기 경보 시스템 정확도 향상 가능
- 개인 맞춤형 학습 경로 추천 시스템으로의 **확장 가능성** 확인

### 인사이트
```
성적 향상은 개인의 노력을 넘어, 시스템으로 관리 가능한 영역입니다.
통제 불가능한 변수(지능, 환경)가 아닌, 개입 가능한 행동 변수에 집중하는 것이
데이터 기반 교육 정책의 핵심입니다.
```

---

## 기술 스택

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from scipy import stats
```

---

## 참고

- Kaggle 대회 : [Playground Series S6E1](https://www.kaggle.com/competitions/playground-series-s6e1)
- 시험 성적 향상 정책 분석 보고서 : [시험 성적 예측 모델링 : 통계분석 및 머신러닝 접근](report/시험성적%20향상%20정책%20제언.pdf)
- 분석코드 : [분석코드](분석코드.ipynb)