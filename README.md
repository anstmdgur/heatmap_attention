# HGA-UNet++

X-ray 관상동맥 조영술 영상에서 협착 영역을 분할하기 위한 heatmap guided Attention-UNet++ 구현임.

ARCADE MICCAI 2023 stenosis subset만 사용함.  
혈관 segmentation, syntax task, bounding box detection은 사용하지 않음.

## 개요

협착 영역은 전체 영상에서 차지하는 비율이 작고, 조영 상태와 혈관 중첩에 따라 경계가 불분명한 경우가 많음.  
단순히 최종 segmentation mask만 예측하는 구조보다, 협착이 있을 가능성이 높은 위치를 coarse heatmap으로 먼저 학습시키고 이를 segmentation head에 전달하는 방식을 사용함.

구현 모델은 두 가지임.

- `baseline`: ResNet34 + UNet++ decoder + stenosis segmentation head
- `heatmap`: ResNet34 + UNet++ decoder + heatmap guided attention segmentation head

`heatmap` 모델이 HGA-UNet++임.

## 설계

공통 backbone은 ImageNet pretrained ResNet34를 사용함.  
입력 영상은 grayscale `1 x 512 x 512`이므로, ResNet34의 첫 convolution weight를 RGB 채널 평균으로 변환해 1채널 입력에 맞춤.

UNet++ decoder는 ResNet34 encoder feature를 dense skip connection 방식으로 결합함.  
최종적으로 `dec2: 64 x 128 x 128` feature를 얻고, 이 feature가 segmentation head의 기본 입력으로 사용됨.

### baseline

```text
image
 -> ResNet34 encoder
 -> UNet++ decoder
 -> dec2
 -> segmentation head
 -> stenosis mask
```

### heatmap / HGA-UNet++

```text
image
 -> ResNet34 encoder
 -> UNet++ decoder
 -> dec4, dec3, dec2
 -> FPN-style heatmap branch
 -> 128 x 128 stenosis heatmap
 -> heatmap feature + heatmap guide
 -> attention segmentation head
 -> stenosis mask
```

heatmap은 최종 mask가 아니라 협착 후보 위치를 알려주는 coarse localization guide임.  
segmentation head에는 `dec2`, `heatmap feature`, `sigmoid(heatmap logits)`가 함께 들어감.

heatmap branch가 segmentation loss에 의해 직접 흔들리지 않도록, segmentation head로 전달되는 heatmap feature와 heatmap probability는 detach해서 사용함.  
즉 heatmap branch는 heatmap loss로 학습되고, segmentation head는 heatmap을 공간적 guide로만 사용함.

attention gate는 feature를 지우는 방식이 아니라 residual하게 증폭하는 방식으로 구성함.

```text
A = sigmoid(Conv([F, C]))
F_att = F * (1 + gamma * A)
```

`gamma`는 학습 가능한 값이고 초기값은 `0.1`임.

## Loss

### baseline

```text
total loss = segmentation loss
```

### heatmap

```text
total loss = segmentation loss + 0.3 * heatmap loss
```

segmentation loss는 Focal Tversky loss를 사용함.

```text
alpha = 0.3
beta  = 0.7
gamma = 0.75
```

협착 영역을 놓치는 FN을 더 크게 반영하기 위해 `beta`를 크게 둠.

heatmap target은 stenosis mask에서 직접 생성함.

```text
512 x 512 binary mask
 -> 128 x 128 area resize
 -> max-pool dilation
 -> Gaussian blur
 -> image-wise max normalization
```

heatmap loss는 Dice loss와 weighted BCE를 같이 사용함.

```text
heatmap loss = 0.5 * Dice loss + 0.5 * Weighted BCE
positive weight = 2.0
```

Dice만 사용할 때 heatmap이 과도하게 넓게 활성화되는 문제를 줄이기 위해 weighted BCE를 함께 사용함.

## 결과

3개 seed 평균 기준 결과임.

| Model | Pixel-wise F1 | Precision | Recall | Stenosis IoU | ARCADE F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UNet++ baseline | 0.556 ± 0.006 | 0.515 ± 0.008 | 0.719 ± 0.004 | 0.412 ± 0.006 | 0.374 |
| HGA-UNet++ | 0.599 ± 0.007 | 0.567 ± 0.004 | 0.734 ± 0.015 | 0.454 ± 0.008 | 0.447 |

HGA-UNet++ 평균 추론 시간은 약 `0.037 s/frame`임.

Pixel-wise F1과 ARCADE F1은 서로 다른 metric임.  
Pixel-wise F1은 binary mask 전체를 기준으로 계산하고, ARCADE F1은 COCO polygon instance 단위로 계산함.

## 데이터 구조

기본 경로:

```text
C:\Code\workspace\data\ARCADE
```

필요한 폴더:

```text
stenosis/train/images
stenosis/train/masks
stenosis/train/annotations/train.json
stenosis/val/images
stenosis/val/masks
stenosis/val/annotations/val.json
stenosis/test/images
stenosis/test/masks
stenosis/test/annotations/test.json
```

`masks`는 학습용 binary mask로 사용함.  
`annotations/*.json`은 ARCADE polygon-instance F1 계산에 사용함.

## 실행 방법

### baseline 학습

```powershell
cd C:\Code\workspace\stenosis_heatmap
C:\Code\workspace\venv\Scripts\python.exe main.py --model baseline --output-dir C:\Code\workspace\stenosis_heatmap\outputs\baseline_seed40 --seed 40
```

### HGA-UNet++ 학습

```powershell
cd C:\Code\workspace\stenosis_heatmap
C:\Code\workspace\venv\Scripts\python.exe main.py --model heatmap --output-dir C:\Code\workspace\stenosis_heatmap\outputs\heatmap_seed40 --seed 40
```

### test만 실행

```powershell
C:\Code\workspace\venv\Scripts\python.exe main.py --model heatmap --output-dir C:\Code\workspace\stenosis_heatmap\outputs\heatmap_seed40 --test-only
```

## 주요 옵션

| Option | 의미 |
| --- | --- |
| `--model` | 실행할 모델 선택. `baseline` 또는 `heatmap` |
| `--output-dir` | checkpoint, csv, result, test image가 저장될 폴더 |
| `--data-dir` | ARCADE 데이터셋 경로 |
| `--epochs` | 최대 학습 epoch 수 |
| `--batch-size` | batch size |
| `--num-workers` | DataLoader worker 수 |
| `--lr` | AdamW learning rate |
| `--weight-decay` | AdamW weight decay |
| `--seed` | random seed |
| `--checkpoint-path` | test-only 시 사용할 checkpoint 경로를 직접 지정 |
| `--test-only` | 학습 없이 checkpoint를 불러와 test만 수행 |

기본 설정:

```text
epochs = 250
batch_size = 8
num_workers = 4
lr = 0.001
weight_decay = 0.0001
seed = 40
```

학습에는 AdamW optimizer를 사용함.  
Validation loss가 plateau 상태일 때 `ReduceLROnPlateau`로 learning rate에 `0.2`를 곱함.  
Early stopping은 validation ARCADE polygon-instance F1 기준으로 수행하며 patience는 `30`임.

## 평가

두 종류의 성능을 저장함.

### Pixel-wise metric

```text
prediction = sigmoid(stenosis logits) > 0.5
```

저장 지표:

```text
IoU
Precision
Recall
F1
Image-wise F1
```

### ARCADE polygon-instance F1

ARCADE 공식 stenosis evaluation 방식에 맞춰 polygon instance 단위로 계산함.

```text
prediction mask
 -> connected component contour
 -> COCO polygon prediction
 -> GT polygon annotation과 instance F1 계산
 -> image-wise 평균
 -> 전체 image 평균
```

결과 파일:

```text
test_arcade_predictions.json
test_arcade_metrics.json
```

## 저장 결과

`--output-dir` 아래에 다음 파일이 저장됨.

```text
output_dir
├── checkpoint.pt
├── training_history.csv
├── result.txt
├── test_arcade_predictions.json
├── test_arcade_metrics.json
└── test_images
    ├── stenosis_seg
    └── stenosis_heatmap   # heatmap 모델에서만 생성
```

`result.txt`에는 실행 옵션, checkpoint 정보, inference time, pixel-wise metric, ARCADE F1이 저장됨.

Test image는 400 dpi로 저장함.

```text
Original / Original + GT overlay / Original + prediction overlay
```

heatmap 모델은 heatmap 결과도 추가로 저장함.

```text
Original / Original + GT heatmap / Original + predicted heatmap
```

## Requirements

```text
torch
torchvision
albumentations
opencv-python
numpy
matplotlib
shapely
tqdm
natsort
```

실험 환경:

```text
PyTorch 2.9.1+cu130
NVIDIA GeForce RTX 4070 Ti SUPER 16GB
```
