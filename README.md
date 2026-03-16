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
- **분석 기간** : 2025.12.29 ~ 2025.12.30

---

## 핵심 분석 결과

### 성적을 결정하는 변수 (Feature Importance)

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

## 파일 구조

```
📂 exam-score-policy-analysis
├── 📄 README.md
├── 📓 analysis.ipynb          # 전체 분석 코드
├── 📊 report_slides.pdf       # 정책 분석 보고서 슬라이드
└── 📁 data/
    └── train.csv              # Kaggle 데이터셋
```

---

## 참고

- Kaggle 대회 : [Playground Series S6E1](https://www.kaggle.com/competitions/playground-series-s6e1)
- 분석 보고서 슬라이드 : `report_slides.pdf` 참고
