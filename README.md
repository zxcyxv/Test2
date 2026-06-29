# Financial Denoising

Kaggle S&P 500 excess returns forecasting 대회를 위해 만든 금융 시계열 feature denoising 파이프라인입니다.

**대회 성과:** 3,677팀 중 574위, 상위 15.6% (약 상위 16%).

대회 목표는 S&P 500의 일별 초과수익을 예측하고, 매 거래일 종가 기준으로 S&P 500 보유 비중을 `0`에서 `2` 사이로 결정하는 것입니다. 평가 지표는 시장 대비 초과 성과를 보면서도 120% volatility constraint를 넘는 전략을 강하게 penalize합니다.

이 프로젝트는 최종 allocation model 자체라기보다, 대회 데이터의 noisy tabular financial features를 더 안정적인 입력 신호로 바꾸기 위한 전처리/표현학습 실험입니다. feature clustering, Mamba 기반 denoiser 학습, causal denoising, denoising 결과 검증까지 하나의 흐름으로 묶어, denoised feature가 downstream return prediction과 trading signal에 도움이 되는지 검증하는 것을 목표로 합니다.

## 핵심 기능

- 통계적 특성에 따라 feature를 cluster로 묶습니다.
- cluster별 Causal Mamba denoiser를 학습합니다.
- VP-SDE 기반 diffusion denoising을 적용합니다.
- production 환경을 고려한 causal denoising 경로를 제공합니다.
- non-causal denoising과 시각화 도구를 실험용으로 제공합니다.
- 원본 데이터와 denoised 데이터를 비교하는 검증 스크립트를 포함합니다.
- Kaggle GPU 환경에서 전체 cluster 학습을 실행하는 보조 스크립트를 제공합니다.

## 대회 데이터에 맞춘 문제 설정

Kaggle 데이터는 단일 가격 시계열이 아니라 여러 계열의 tabular feature로 구성됩니다.

- `M*`: market dynamics / technical features
- `E*`: macro economic features
- `I*`: interest rate features
- `P*`: price / valuation features
- `V*`: volatility features
- `S*`: sentiment features
- `D*`: dummy / binary features
- `forward_returns`, `risk_free_rate`, `market_forward_excess_returns`: train set에서만 제공되는 target/보조 target

현재 포함된 `data/train.csv`는 `date_id`와 target columns를 포함해 98개 column을 갖습니다. 이 구조에서는 논문처럼 하나의 가격 시계열을 denoise하는 방식만으로는 충분하지 않습니다. 서로 다른 스케일과 통계적 성질을 가진 feature를 한 모델에 그대로 넣으면 모델 용량과 학습 시간이 커지고, 관계가 약한 feature끼리 서로를 오염시킬 수 있습니다. 그래서 먼저 feature를 통계적 특성으로 clustering하고, cluster별 denoiser를 학습한 뒤, denoised feature를 다시 합치는 구조로 설계했습니다.

## 프로젝트 구조

```text
FinancialDenoising/
├── data/
│   └── train.csv                         # 예제 학습 데이터
├── artifacts/
│   └── clustering_results/               # feature clustering 결과
├── models/
│   ├── clustering/                       # feature 분석 및 clustering
│   └── diffusion_mamba/                  # Mamba denoiser, VP-SDE, guidance losses
├── training/
│   ├── cluster_features.py               # feature clustering 실행
│   └── train_denoiser.py                 # cluster별 denoiser 학습
├── inference/
│   ├── denoise_causal.py                 # leak-free causal denoising
│   ├── denoise_dataset.py                # 실험용 batch/non-causal denoising
│   └── visualize_denoising.py            # denoising 결과 시각화
├── evaluation/
│   └── validate_denoising.py             # denoising 품질 검증
├── common/
│   └── evaluation/                       # trading signal 검증 공통 도구
├── analysis/
│   └── analyze_denoised_data.py          # denoised 데이터 분석
├── scripts/
│   ├── kaggle_train_all_clusters.py      # Kaggle 전체 학습 스크립트
│   └── calculate_target_loss.py          # target loss 산출 보조 스크립트
└── utils/                                # 데이터 분할 및 유틸리티
```

## 전체 흐름

1. `data/train.csv`에서 feature와 target을 로드합니다.
2. `training/cluster_features.py`로 feature 통계량을 분석하고 cluster를 만듭니다.
3. `training/train_denoiser.py`로 cluster별 denoiser를 학습합니다.
4. `inference/denoise_causal.py`로 각 row를 과거 window만 사용해 denoise합니다.
5. `evaluation/validate_denoising.py` 또는 `common/evaluation/validate_trading_signals.py`로 원본 대비 성능 변화를 검증합니다.

## 설치

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`uv`를 사용하는 경우:

```bash
uv sync
```

GPU 학습을 하려면 환경에 맞는 PyTorch CUDA 빌드가 설치되어 있어야 합니다.

## 빠른 실행

### 1. Feature clustering

```bash
python training/cluster_features.py \
  --data_path data/train.csv \
  --output_dir artifacts/clustering_results \
  --n_clusters 5
```

### 2. Cluster별 denoiser 학습

```bash
python training/train_denoiser.py \
  --cluster_id 0 \
  --data_path data/train.csv \
  --cluster_config artifacts/clustering_results/cluster_assignments.json \
  --output_dir trained_models \
  --epochs 100 \
  --device cuda
```

여러 cluster를 한 번에 학습하려면:

```bash
python scripts/kaggle_train_all_clusters.py
```

### 3. Causal denoising

```bash
python inference/denoise_causal.py \
  --input_csv data/train.csv \
  --output_csv artifacts/denoised/train_denoised_causal.csv \
  --cluster_config artifacts/clustering_results/cluster_assignments.json \
  --models_dir trained_models \
  --device cuda
```

causal denoising은 row `t`를 denoise할 때 `[t-59, t]` 구간만 사용합니다. 미래 데이터를 보지 않기 때문에 production 검증에 더 적합합니다.

### 4. 실험용 non-causal denoising

```bash
python inference/denoise_dataset.py \
  --input_csv data/train.csv \
  --output_csv artifacts/denoised/train_denoised.csv \
  --cluster_config artifacts/clustering_results/cluster_assignments.json \
  --models_dir trained_models \
  --stride 60 \
  --device cuda
```

이 경로는 빠른 실험용입니다. window 전체를 덮어쓰는 방식이라 production 평가에는 causal denoising을 우선 사용합니다.

## 검증

소스 컴파일:

```bash
python -m compileall models training inference evaluation common analysis utils scripts
```

원본 데이터와 denoised 데이터 비교:

```bash
python common/evaluation/validate_trading_signals.py \
  --train_original data/train.csv \
  --train_denoised artifacts/denoised/train_denoised_causal.csv \
  --val_original artifacts/splits/val_only.csv \
  --val_denoised artifacts/denoised/val_denoised_causal.csv
```

## 모델 구성

- `CausalMambaDenoiser`: causal sequence denoising을 위한 Mamba 기반 모델
- `VPSDE`: variance-preserving SDE diffusion process
- `DenoisingLoss`: noise prediction 중심 학습 objective
- `guidance.py`, `losses.py`: TV/Fourier 기반 guidance loss
- `iterative_denoiser.py`: inference 단계 iterative denoising 루프

## 논문 대비 구현 차이

이 프로젝트는 `A Financial Time Series Denoiser Based on Diffusion Model`의 핵심 아이디어인 diffusion 기반 financial time series denoising을 Kaggle 대회 환경에 맞게 변형한 구현입니다. 논문을 그대로 재현하는 것이 아니라, 대회 데이터 구조와 제출 제약에 맞춰 다음과 같이 설계했습니다.

| 항목 | 논문 | 이 프로젝트 | 변경 이유 |
| --- | --- | --- | --- |
| 문제 설정 | 금융 시계열 denoising으로 downstream return classification과 MACD/Bollinger trading signal 개선을 확인 | S&P 500 excess returns 예측 대회의 feature preprocessing/representation learning 모듈 | 대회 평가는 일별 allocation 전략의 Sharpe 변형 지표이므로, denoising 자체보다 downstream prediction에 유용한 feature 정제가 목적입니다. |
| 입력 데이터 | 단일 또는 소수의 가격 중심 시계열을 denoise하는 설정 | `M*`, `E*`, `I*`, `P*`, `V*`, `S*` 등 다수 feature를 가진 tabular time series | 대회 데이터는 feature family가 다양하고 스케일/분포/결측 패턴이 다르기 때문에, 단일 시계열 denoiser보다 multivariate feature denoising이 필요합니다. |
| Feature 처리 | 전체 시계열을 하나의 denoising 대상으로 다룸 | ADF p-value, Hurst exponent, autocorrelation, volatility, skewness 등으로 feature를 clustering | 모든 feature를 한 모델에 넣으면 차원이 커지고 서로 다른 동학이 섞입니다. 유사한 통계적 성질의 feature끼리 묶어 cluster별 모델을 학습하면 모델 부담을 줄이고 cluster 내부 구조를 더 잘 보존할 수 있습니다. |
| 모델 구조 | CSDI 계열 conditional transformer score network를 사용 | Mamba 기반 `CausalMambaDenoiser`와 cluster별 multivariate denoiser 사용 | Kaggle Notebook은 GPU 8시간 제한이 있고, forecasting phase도 시간 제한이 있습니다. Mamba는 긴 sequence를 효율적으로 처리하기 좋고, causal block으로 바꾸기 쉬워 evaluation API의 forward-looking 방지 요구와 잘 맞습니다. |
| Causality | 논문 실험은 denoised series의 downstream 성능 검증에 초점 | `denoise_causal.py`에서 row `t`를 `[t-59, t]` window만 사용해 denoise | Kaggle evaluation API는 미래 정보를 볼 수 없는 순차 예측 환경입니다. 따라서 실험용 non-causal path와 별도로, 제출/실전 검증에는 causal path를 우선하도록 분리했습니다. |
| Conditioning | conditional diffusion과 classifier-free guidance를 논문 핵심 구성으로 사용 | 현재 학습은 noise prediction 중심이며, condition interface는 확장 가능하게 유지 | 대회 feature에는 “깨끗한 정답 시계열”이나 명확한 condition label이 없습니다. 먼저 self-supervised denoising을 안정화하고, downstream target은 denoising 후 검증 단계에서 평가하도록 분리했습니다. |
| Guidance | inference 단계에서 TV loss와 Fourier loss를 guidance로 사용 | training에서는 `guidance_weight=0.0`, inference에서 TV/Fourier guidance 적용 | 학습 중 guidance를 강하게 넣으면 target 없는 self-supervised denoising에서 과도한 smoothing이 발생할 수 있습니다. 학습은 noise prediction으로 안정화하고, inference에서 smoothness와 frequency filtering을 조절합니다. |
| Sampling 안정화 | 여러 random seed의 reverse process를 평균해 randomness를 줄임 | 기본 경로는 단일/제한된 iterative denoising을 사용 | Kaggle runtime 안에서 여러 cluster와 여러 row를 처리해야 하므로, 여러 reverse sample 평균은 비용이 큽니다. 대신 cluster 분리, instance normalization, causal window 처리로 안정성을 확보하는 방향을 택했습니다. |
| Normalization | 논문은 시계열 denoising 실험 설정 중심 | window별 instance normalization 사용 | 대회 feature들은 장기 시계열에서 regime change와 scale shift가 큽니다. global statistics를 쓰면 train/validation leakage나 scale drift 문제가 생길 수 있어, 각 window 내부 통계로 정규화합니다. |
| 평가 | return classification, MACD/Bollinger 전략, classifier 기반 trading 실험 | 원본 vs denoised feature의 predictive signal, trading signal validation, downstream 모델 입력 품질 검증 | 대회 metric은 allocation 전략의 risk-adjusted performance입니다. 따라서 denoising 품질만 보지 않고, denoised feature가 예측력과 전략 품질을 개선하는지 확인하는 쪽으로 평가를 맞췄습니다. |

요약하면, 이 저장소는 논문의 diffusion denoising 아이디어를 Kaggle의 다중 feature 금융 예측 문제에 맞게 옮긴 실험입니다. 핵심 변경은 “단일 시계열 denoiser”에서 “cluster별 causal multivariate denoiser”로 확장한 점이며, 이는 대회 데이터의 feature 수, heterogeneity, runtime limit, forward-looking 방지 조건 때문에 필요한 설계였습니다.

## 데이터와 산출물 관리

저장소에는 구조 확인과 재현을 위한 `data/train.csv`와 `artifacts/clustering_results/`가 포함되어 있습니다.

다음 파일은 생성물로 보고 Git에서 제외합니다.

- `trained_models/`
- `artifacts/denoised/`
- `artifacts/splits/`
- `artifacts/visualizations/`
- `artifacts/evaluation_results/`
- 로그 파일과 체크포인트 파일

## 참고 문헌

```bibtex
@article{wang2024financial,
  title={A Financial Time Series Denoiser Based on Diffusion Model},
  author={Wang, Zhuohan and Ventre, Carmine},
  journal={arXiv preprint arXiv:2409.02138},
  year={2024}
}
```

Paper: https://arxiv.org/pdf/2409.02138

## 현재 상태

이 저장소는 연구/실험용 프로토타입입니다. 모델 학습과 denoising 경로는 포함되어 있지만, 데이터셋별 feature schema와 검증 기준은 실험 목적에 맞게 조정해야 합니다. 실제 투자 또는 production pipeline에 사용하기 전에는 leakage 검증, out-of-sample 검증, transaction cost 반영, 실험 추적 체계를 별도로 갖추는 것이 필요합니다.
