# Stenosis Heatmap

ARCADE stenosis 데이터셋만 사용해서 관상동맥 협착 segmentation 모델을 학습하고 평가하는 프로젝트입니다.

비교 모델은 두 가지입니다.

1. `baseline`: ResNet34 + UNet++ + stenosis segmentation head
2. `heatmap`: ResNet34 + UNet++ + stenosis heatmap FPN + heatmap-guided attention segmentation head

syntax/vessel task와 bbox detection 코드는 사용하지 않습니다.

## 실행

### UNet++ baseline

```powershell
cd C:\Code\workspace\stenosis_heatmap
C:\Code\workspace\venv\Scripts\python.exe main.py --model baseline --output-dir C:\Code\workspace\stenosis_heatmap\outputs\baseline_seed40 --seed 40
```

### UNet++ + stenosis heatmap attention

```powershell
cd C:\Code\workspace\stenosis_heatmap
C:\Code\workspace\venv\Scripts\python.exe main.py --model heatmap --output-dir C:\Code\workspace\stenosis_heatmap\outputs\heatmap_seed40 --seed 40
```

### Test only

```powershell
C:\Code\workspace\venv\Scripts\python.exe main.py --model heatmap --output-dir C:\Code\workspace\stenosis_heatmap\outputs\heatmap_seed40 --test-only
```

기본값:

```text
epochs: 250
batch_size: 8
num_workers: 4
lr: 0.001
weight_decay: 0.0001
seed: 40
```

`--seed`로 seed를 바꿀 수 있습니다. 학습한 checkpoint에는 seed가 함께 저장되고, `result.txt`에는 checkpoint에 저장된 seed가 우선 기록됩니다.

## 데이터

기본 데이터 경로:

```text
C:\Code\workspace\data\ARCADE
```

사용 폴더:

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

`masks`는 학습용 binary semantic mask로 사용하고, `annotations/*.json`은 ARCADE 공식 COCO polygon-instance metric 계산에 사용합니다.

## Augmentation

Train split에만 적용합니다. 작은 stenosis target이 augmentation으로 과하게 왜곡되지 않도록 기존 설정보다 완화했습니다.

```text
vertical flip       p=0.5
horizontal flip     p=0.5
scale               0.8 ~ 1.2
translate           +/- 0.15
rotation            +/- 15 degrees
shear               +/- 3 degrees
perspective         scale 0.0 ~ 0.0005, p=0.3
hue                 +/- 3, p=0.5
saturation          +/- 30, p=0.5
value               +/- 40, p=0.5
```

원본은 grayscale이지만 HSV augmentation을 위해 transform 내부에서 일시적으로 RGB로 변환한 뒤 다시 grayscale로 되돌립니다.

## 모델 구조

### 공통 ResNet34 encoder

입력은 `1 x 512 x 512` grayscale angiography image입니다.

ResNet34는 ImageNet pretrained weight를 사용합니다. 원래 ResNet34의 첫 convolution은 RGB 3채널 입력용이므로, pretrained `conv1` weight를 RGB channel 방향으로 평균내어 1채널 convolution으로 초기화합니다.

```text
pretrained conv1: 64 x 3 x 7 x 7
grayscale conv1: 64 x 1 x 7 x 7 = mean(pretrained conv1, dim=RGB)
```

이렇게 하면 첫 layer를 랜덤 초기화하지 않아 seed variance를 줄일 수 있습니다.

Encoder 출력:

```text
e0:  64 x 256 x 256
e1:  64 x 128 x 128
e2: 128 x  64 x  64
e3: 256 x  32 x  32
e4: 512 x  16 x  16
```

### UNet++ dense decoder

공통 decoder는 UNet++ 방식의 dense skip connection을 사용합니다.

```text
e4 -> upsample + e3 -> dec4: 256 x 32 x 32
e3 -> 1x1 conv -> upsample + e2 -> x2_1: 128 x 64 x 64
dec4 -> upsample + x2_1 + e2 -> dec3: 128 x 64 x 64
e2 -> 1x1 conv -> upsample + e1 -> x1_1: 64 x 128 x 128
x2_1 -> 1x1 conv -> upsample + e1 + x1_1 -> x1_2: 64 x 128 x 128
dec3 -> upsample + e1 + x1_1 + x1_2 -> dec2: 64 x 128 x 128
```

`dec2`는 baseline과 heatmap 모델이 공유하는 최종 decoder feature입니다.

### UNet++ baseline

```text
dec2: 64 x 128 x 128
 -> segmentation head
 -> upsample x2
 -> upsample x2
 -> 1x1 conv
 -> stenosis logits: 1 x 512 x 512
```

Baseline은 heatmap branch와 attention gate를 사용하지 않습니다.

### UNet++ heatmap attention

Heatmap 모델은 `dec4`, `dec3`, `dec2`를 사용해 stenosis heatmap branch를 만듭니다.

```text
dec4: 256 x  32 x  32 -> 1x1 conv -> 64 x  32 x  32
dec3: 128 x  64 x  64 -> 1x1 conv -> 64 x  64 x  64
dec2:  64 x 128 x 128 -> 1x1 conv -> 64 x 128 x 128
```

FPN-like top-down fusion:

```text
64 x 32 x 32
 -> upsample to 64 x 64
 -> concat with 64-scale lateral feature
 -> conv x2
 -> f64: 64 x 64 x 64

f64
 -> upsample to 128 x 128
 -> concat with 128-scale lateral feature
 -> conv x2
 -> f128: 64 x 128 x 128
```

Heatmap branch attention:

```text
context = concat(f64_up, f32_to_128)  # 128 x 128 x 128
A = sigmoid(Conv([f128, context]))
heat_feature = f128 * (1 + gamma * A)
```

`gamma`는 learnable scalar이며 초기값은 `0.1`입니다. 예전처럼 feature를 강하게 증폭하지 않고, 학습 초기에 attention 영향이 작게 시작되도록 했습니다.

Heatmap output:

```text
heat_feature: 64 x 128 x 128
 -> ConvBNAct
 -> 1x1 conv
 -> heatmap logits: 1 x 128 x 128
```

Segmentation head input:

```text
dec2:          64 x 128 x 128
heat_feature: 64 x 128 x 128  # detach
heatmap_prob:  1 x 128 x 128  # sigmoid(heatmap_logits), detach
concat:       129 x 128 x 128
```

`heat_feature`와 `heatmap_prob`는 segmentation head에 들어가기 전에 detach합니다. 이 detach는 heatmap loss를 막기 위한 것이 아닙니다. 반대로 segmentation loss가 heatmap branch로 역전파되어 localization branch를 흔드는 것을 막기 위한 장치입니다.

즉 gradient 흐름은 다음과 같습니다.

```text
heatmap loss -> heatmap branch 학습 가능
segmentation loss -> heatmap branch로 직접 역전파되지 않음
segmentation loss -> dec2 및 segmentation head 학습
```

Segmentation head 내부 attention:

```text
fused = Conv(concat(dec2, detached heat_feature, detached heatmap_prob))
context = concat(detached heat_feature, detached heatmap_prob)
A = sigmoid(Conv([fused, context]))
fused_att = fused * (1 + gamma * A)
```

마지막으로 두 번 upsample하여 최종 mask logit을 만듭니다.

```text
fused_att
 -> refine conv x2
 -> upsample x2
 -> upsample x2
 -> 1x1 conv
 -> stenosis logits: 1 x 512 x 512
```

## Heatmap target

Heatmap target은 bbox center가 아니라 stenosis mask 자체에서 생성합니다.

```text
stenosis mask 512 x 512
 -> area resize to 128 x 128
 -> max-pool dilation kernel 5
 -> Gaussian blur kernel 9, sigma 2.0
 -> image-wise max normalize
 -> soft heatmap target
```

목적은 boundary만 맞추는 hard segmentation이 아니라, stenosis 주변의 coarse localization prior를 학습시키는 것입니다.

## Loss

### Baseline

```text
L_total = L_seg
L_seg = Focal Tversky Loss
alpha = 0.3
beta  = 0.7
gamma = 0.75
```

### Heatmap attention

```text
L_total = L_seg + 0.3 * L_heatmap
```

Heatmap loss는 Dice loss와 weighted BCE를 섞습니다.

```text
L_heatmap = 0.5 * DiceLoss + 0.5 * WeightedBCEWithLogits
positive weight = 2.0
```

Dice-only heatmap loss는 heatmap prediction이 넓은 영역에서 높게 켜지는 현상을 만들 수 있어, background false positive를 더 직접적으로 벌하기 위해 weighted BCE를 함께 사용합니다.

## Metrics

두 종류의 metric을 기록합니다.

### 1. Pixel-wise image F1

Semantic segmentation 보조 지표입니다.

```text
prediction = sigmoid(stenosis_logits) > 0.5
target = binary stenosis mask
image-wise IoU / Precision / Recall / F1
```

이 값은 이미지별 binary mask F1을 평균낸 것입니다. ARCADE leaderboard official score와 직접 비교하지 않습니다.

### 2. ARCADE official polygon-instance F1

ARCADE 공식 stenosis evaluator와 같은 방식의 주 평가 지표입니다.

과정:

```text
sigmoid(stenosis_logits) > threshold
 -> connected component contour extraction
 -> COCO polygon prediction 생성
 -> GT annotations/*.json의 polygon instance와 비교
 -> predicted polygon마다 같은 이미지의 GT polygon과 area-based F1 계산
 -> predicted instance별 max F1 사용
 -> missing GT instance 수만큼 0 추가
 -> image-wise instance mean
 -> total mean
```

결과는 다음 파일에 저장됩니다.

```text
test_arcade_predictions.json
test_arcade_metrics.json
result.txt
```

현재 checkpoint 저장 기준도 validation set의 ARCADE polygon-instance F1입니다.

```text
checkpoint monitor = val_arcade_polygon_instance_f1
```

Scheduler는 안정성을 위해 validation loss 기준을 유지합니다.

```text
scheduler monitor = val_loss
checkpoint monitor = val_arcade_polygon_instance_f1
early stopping monitor = val_arcade_polygon_instance_f1
```

## Post-processing

현재 segmentation visualization에는 closing이 적용됩니다.

```text
visualization mask = binary closing(raw prediction mask, kernel=3)
```

공식 metric용 polygon 변환은 raw threshold mask를 기반으로 수행합니다. 향후 validation set에서 다음 hyperparameter sweep을 통해 official F1 기준으로 후처리를 고정할 수 있습니다.

```text
threshold: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
min_area: 1, 5, 10, 20, 30, 50
closing kernel: none, 3, 5
opening kernel: none, 3
```

가장 큰 connected component만 남기는 방식은 사용하지 않습니다. 한 이미지에 stenosis instance가 여러 개 있을 수 있기 때문입니다. 대신 모든 component를 instance 후보로 유지하고, 너무 작은 component만 `min_area`로 제거하는 방식이 적절합니다.

## 저장 결과

```text
output_dir
├── checkpoint.pt
├── training_history.csv
├── result.txt
├── test_arcade_predictions.json
├── test_arcade_metrics.json
└── test_images
    ├── stenosis_seg
    └── stenosis_heatmap  # heatmap model only
```

`stenosis_seg` 이미지는 다음 3패널 형식입니다.

```text
Original / Original + GT mask overlay / Original + predicted mask overlay
```

`stenosis_heatmap` 이미지는 heatmap 모델에서만 저장됩니다.

```text
Original / Original + GT soft heatmap overlay / Original + predicted heatmap overlay
```

Predicted heatmap visualization에는 보기 좋게 Gaussian blur가 적용됩니다. 이 blur는 시각화용이며 네트워크 출력과 metric 계산에는 영향을 주지 않습니다.
