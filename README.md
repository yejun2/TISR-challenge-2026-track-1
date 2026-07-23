# TISR-challenge-2026-track-1
TISR challenge 2026 track 1 9th place

## 2. 주요 실험 정리

| 실험 | 관련 파일 | 설명 |
| --- | --- | --- |
| 기본 x8 finetune | `options/train/train_coslr_multiloss_multiscale_DRCT-L_SRx8_finetune_from_merged.yml` | mixed pretrain 가중치에서 시작해 공식 train/val 데이터로 x8 복원 성능을 finetune |
| mixed pretrain 데이터 생성 | `utils/merge_image.py` | 서로 다른 샘플 4개를 crop 후 합성해 `LR_x8_mix_pretrain`, `GT_mix_pretrain` 생성 |
| masking 실험 | `options/train/train_finetune_use_masking.yml`, `options/train/train_finetune_use_masking_paired.yml` | LQ 이미지에만 patch masking을 적용하는 augmentation 실험 |
| paired dataset 정리 | `drct/data/thermal_paired_dataset.py` | thermal 전용 paired loader, masking, HP feature 결합 로직 추가 |
| HP feature 5채널 입력 | `options/train/finetune_use_hp_feat.yml` | thermal 3채널 + HP 2채널을 합쳐 5채널 입력으로 사용 |
| pseudo x2 intermediate 학습 | `options/train/finetune_x2_from_opencv_pretrained.yml` | `LR_x8`을 입력으로 `LR_x2`를 맞추는 중간 단계 학습 |
| depth 관련 실험 흔적 | `options/train/finetune_use_depth_anything.yml` | depth 조건부 입력 실험용 설정이 있으나 현재 체크인된 yml에서는 `false`로 비활성화 상태 |
| 커스텀 loss | `drct/losses/multi_loss.py` | MSE와 SSIM을 결합한 loss 사용 |

## 3. 폴더 구조

아래 폴더들이 이 저장소에서 핵심입니다.

```text
PBVS_TSR/
├── drct/
│   ├── archs/                  # DRCT 아키텍처 및 커스텀 변형
│   ├── data/                   # thermal paired dataset, masking, HP feature loader
│   ├── losses/                 # MultiLoss, SSIMLoss
│   └── train.py / test.py
├── options/
│   ├── train/                  # 실험별 학습 설정
│   └── test/                   # 추론/평가 설정
├── utils/
│   ├── merge_image.py          # mixed pretrain 데이터 생성
│   ├── downsample2x.py         # LR_x2 생성용 다운샘플링
│   ├── post_processing.py      # 후처리 실험 스크립트
│   └── fuzzy_images.py         # 추가 이미지 조작/노이즈 실험
├── datasets/track1/            # 대회 데이터셋 및 파생 데이터
├── results/                    # 추론 결과
├── tb_logger/                  # TensorBoard 로그
├── Visualization/              # 시각화 노트북
└── BasicSR/                    # 기반 프레임워크 소스
```

## 4. 환경 설정

기본 README 기준 환경은 아래와 같습니다.

- Python 3.8
- PyTorch 1.12.1
- CUDA 11.6

예시 설치 순서:

```bash
conda create -n pbvs_tsr python=3.8 -y
conda activate pbvs_tsr
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd /SSD4/vipnu/PBVS_TSR
pip install -r requirements.txt
python setup.py develop
```

## 5. 데이터 준비

기본적으로는 공식 Track 1 thermal 데이터가 아래 구조로 들어가야 합니다.

```text
datasets/
└── track1/
    ├── thermal/
    │   ├── train/
    │   │   ├── GT
    │   │   └── LR_x8
    │   ├── val/
    │   │   ├── GT
    │   │   └── LR_x8
    │   └── test/
    │       └── sisr_x8
    └── depth/
        ├── train
        └── val
```

실험에 따라 아래 파생 폴더들도 사용합니다.

- `datasets/track1/thermal/train/LR_x8_mix_pretrain`
- `datasets/track1/thermal/train/GT_mix_pretrain`
- `datasets/track1/thermal/train/LR_x2`
- `datasets/track1/thermal/val/LR_x2`
- `datasets/track1/thermal/train/HP_feats`
- `datasets/track1/thermal/val/HP_feats`

### mixed pretrain 데이터 생성

```bash
python utils/merge_image.py
```

이 스크립트는 `train/LR_x8`, `train/GT`를 바탕으로 합성 샘플을 만들어 pretrain용 mixed dataset을 생성합니다.

### LR_x2 생성

```bash
python utils/downsample2x.py <input_dir> <output_dir>
```

예를 들어 `GT`를 2배 다운샘플해서 `LR_x2`를 만들 때 사용할 수 있습니다.

## 6. 학습

대표 학습 명령은 아래와 같습니다.

### 기본 finetune

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29521 \
  drct/train.py \
  -opt options/train/train_coslr_multiloss_multiscale_DRCT-L_SRx8_finetune_from_merged.yml \
  --launcher pytorch
```

### 저장된 실험용 쉘 스크립트

```bash
bash train_masking.sh
bash train_paired.sh
bash train_hp.sh
bash train_x2_opencv.sh
bash train_depth_anything.sh
```

각 스크립트의 의미는 다음과 같습니다.

- `train_masking.sh`: masking + paired loader 기반 실험
- `train_paired.sh`: thermal paired dataset 기반 finetune
- `train_hp.sh`: HP feature 5채널 입력 실험
- `train_x2_opencv.sh`: `LR_x8 -> LR_x2` intermediate 학습
- `train_depth_anything.sh`: depth 관련 실험용 실행 스크립트

학습 결과물은 보통 `experiments/`와 `tb_logger/` 아래에 저장됩니다.

## 7. 추론과 테스트

기본 테스트 엔트리는 아래와 같습니다.

```bash
python drct/test.py -opt options/test/test_DRCT-L_SRx8_finetune_from_merged.yml
# 또는
bash test.sh
```

현재 체크인된 `options/test/test_DRCT-L_SRx8_finetune_from_merged.yml`은 다음 특징이 있습니다.

- 입력 경로가 절대 경로로 고정되어 있음
- `results/test_x8tox2_archived_20260227_165001/visualization/thermal`를 입력으로 사용
- `experiments/pretrained_models/x4_pretrained_opencv.pth`를 읽어 최종 복원 수행

즉, 현재 설정은 단일 x8 모델보다는 `x8 -> x2 -> GT` 2-stage 흐름에 가깝습니다.  
다른 환경에서 재현할 때는 입력 경로와 체크포인트 경로를 반드시 다시 맞춰야 합니다.

추론 결과는 `results/` 아래에 저장됩니다.

## 8. 코드 수정 포인트

이 저장소에서 대회용으로 손댄 핵심 포인트는 아래 파일들입니다.

- `drct/data/thermal_paired_dataset.py`
  thermal 전용 paired loader, masking, HP feature 결합
- `drct/losses/multi_loss.py`
  MSE + SSIM 결합 loss
- `options/train/*.yml`
  실험별 하이퍼파라미터와 데이터 경로
- `utils/merge_image.py`
  mixed pretrain 데이터 생성
- `utils/downsample2x.py`
  intermediate supervision용 `LR_x2` 생성

## 9. 남아 있는 산출물

이 폴더에는 코드만 있는 것이 아니라 실제 실험 산출물도 함께 남아 있습니다.

- `datasets/track1/`: 대회 데이터 일부 및 파생 데이터
- `tb_logger/`: 학습 로그
- `results/`: 추론 결과 이미지
- `Visualization/`: 분석용 노트북

정리용 저장소로 다시 다듬을 계획이라면, 나중에 아래처럼 분리하는 것도 좋습니다.

- `code/`: 재현 가능한 학습/추론 코드
- `configs/`: 최종 사용한 설정만 정리
- `assets/`: README용 이미지
- `logs/` 또는 별도 스토리지: 대용량 실험 산출물

## 10. 주의할 점

- 일부 옵션 파일은 절대 경로를 사용합니다.
- 체크포인트 파일 이름이 `pretrained_model.pth`, `pretrain_model.pth`, `x4_pretrained_opencv.pth`처럼 실험별로 다릅니다.
- depth 관련 실험 코드는 남아 있지만, 현재 yml만 보면 실제 활성화 여부는 다시 확인이 필요합니다.
- 저장소 안에 데이터와 로그가 같이 포함되어 있어, 외부 공유용 README와 개인 실험 폴더 README의 목적이 조금 다를 수 있습니다.

## 11. Acknowledgements

이 코드베이스는 아래 프로젝트들의 도움을 크게 받았습니다.

- [DRCT](https://github.com/ming053l/DRCT)
- [TISR](https://github.com/upczww/TISR/tree/master)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)

---

이 README는 현재 폴더에 남아 있는 코드와 설정 파일을 기준으로 정리한 1차 문서입니다.  
추가로 아래 정보까지 넣으면 포트폴리오용 README로 훨씬 좋아집니다.

- 최종 제출 모델이 무엇이었는지
- 성능 비교 표 또는 리더보드 결과
- 어떤 실험이 실제로 유효했고 무엇을 버렸는지
- 대회에서 맡았던 역할과 구현 기여도
