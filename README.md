# Financial Denoising

금융 시계열 feature를 diffusion 기반 denoising으로 정제하는 연구용 파이프라인입니다.

이 프로젝트는 feature clustering, Mamba 기반 denoiser 학습, causal denoising, denoising 결과 검증까지 하나의 실험 흐름으로 묶습니다. 핵심 목표는 noisy financial feature를 더 안정적인 신호로 변환하고, 변환된 데이터가 downstream trading signal에 도움이 되는지 검증하는 것입니다.

## 핵심 기능

- 통계적 특성에 따라 feature를 cluster로 묶습니다.
- cluster별 Causal Mamba denoiser를 학습합니다.
- VP-SDE 기반 diffusion denoising을 적용합니다.
- production 환경을 고려한 causal denoising 경로를 제공합니다.
- non-causal denoising과 시각화 도구를 실험용으로 제공합니다.
- 원본 데이터와 denoised 데이터를 비교하는 검증 스크립트를 포함합니다.
- Kaggle GPU 환경에서 전체 cluster 학습을 실행하는 보조 스크립트를 제공합니다.

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
