# 데이터 다운로드 안내

## 다운로드 방법

### 방법 1: Kaggle 웹사이트
1. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 접속
2. `Download` 버튼 클릭
3. `creditcard.csv` 를 이 폴더(`data/`)에 저장

### 방법 2: Kaggle CLI
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

## 데이터 설명

| 항목 | 내용 |
|------|------|
| 출처 | 2013년 유럽 카드사 실거래 데이터 |
| 건수 | 284,807건 |
| 사기 | 492건 (0.172%) |
| 피처 | V1~V28 (PCA 익명화), Time, Amount |
| 타겟 | Class (0=정상, 1=사기) |

## 데이터 없이 실행하기

`data/creditcard.csv` 가 없으면 자동으로 **동일한 통계 구조의 데모 데이터**가 생성됩니다.  
파이프라인 구조와 코드 품질 확인에는 데모 데이터로도 충분합니다.
