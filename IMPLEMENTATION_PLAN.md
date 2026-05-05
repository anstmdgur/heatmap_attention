# Stenosis Heatmap 구현 계획 및 현재 설계

이 문서는 현재 코드 기준의 구조, loss, metric, 학습 정책을 정리한 문서입니다.

## 목표

ARCADE stenosis 데이터만 사용해 두 모델을 같은 조건에서 비교합니다.

1. **UNet++ baseline**
   - 입력: `1 x 512 x 512`
   - 출력: `1 x 512 x 512` stenosis segmentation logit
   - heatmap branch 없음

2. **UNet++ heatmap attention**
   - 입력: `1 x 512 x 512`
   - ResNet34 encoder + UNet++ dense decoder
   - `dec4`, `dec3`, `dec2`에서 heatmap FPN 생성
   - `1 x 128 x 128` stenosis heatmap 예측
   - heatmap feature와 heatmap probability를 detach한 뒤 segmentation head의 spatial prior로 사용
   - 출력:
     - `stenosis`: `1 x 512 x 512`
     - `heatmap`: `1 x 128 x 128`

## 파일 구성

```text
C:\Code\workspace\stenosis_heatmap
├── README.md
├── IMPLEMENTATION_PLAN.md
├── dataset.py
├── model.py
├── loss.py
├── metrics.py
├── arcade_metric.py
├── train.py
└── main.py
```

역할:

```text
dataset.py       ARCADE stenosis image/mask loader, augmentation
model.py         ResNet34-UNet++ baseline 및 heatmap attention model
loss.py          focal tversky, weighted BCE, heatmap target/loss
metrics.py       pixel-wise image metrics
arcade_metric.py ARCADE official polygon-instance F1 계산
train.py         train/eval/test loop
main.py          CLI, optimizer, scheduler, checkpoint, result 저장
```

## Dataset

사용 경로:

```text
C:\Code\workspace\data\ARCADE
```

사용 파일:

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

학습 loss는 binary mask를 사용합니다. 공식 ARCADE metric은 COCO polygon annotation json을 사용합니다.

## Augmentation

Train split에만 적용합니다.

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

기존 강한 augmentation은 작은 stenosis target을 과하게 왜곡할 수 있어 완화했습니다.

## ResNet34 입력 처리

입력 이미지는 grayscale `1 x 512 x 512`입니다.

ImageNet pretrained ResNet34는 RGB 3채널 입력을 기대하므로, 첫 convolution weight를 다음 방식으로 초기화합니다.

```text
pretrained conv1 weight: 64 x 3 x 7 x 7
new grayscale conv1:    64 x 1 x 7 x 7
new weight = mean(pretrained weight over RGB channel)
```

이 방식은 첫 convolution을 랜덤 초기화하지 않으므로 seed variance를 줄이는 데 도움이 됩니다.

## UNet++ Baseline

구조:

```text
Input 1 x 512 x 512
 -> ResNet34 encoder
 -> UNet++ dense decoder
 -> dec2: 64 x 128 x 128
 -> segmentation head
 -> stenosis logits: 1 x 512 x 512
```

Segmentation head:

```text
ConvBNAct 64 -> 64
ConvBNAct 64 -> 48
Upsample x2
ConvBNAct 48 -> 48
Upsample x2
ConvBNAct 48 -> 32
1x1 Conv 32 -> 1
```

## UNet++ Heatmap Attention

Decoder features:

```text
dec4: 256 x  32 x  32
dec3: 128 x  64 x  64
dec2:  64 x 128 x 128
```

Heatmap FPN:

```text
dec4 -> 1x1 conv -> 64 x 32 x 32
dec3 -> 1x1 conv -> 64 x 64 x 64
dec2 -> 1x1 conv -> 64 x 128 x 128

32-scale feature -> upsample -> 64-scale fusion
64-scale feature -> upsample -> 128-scale fusion
```

Heatmap branch attention:

```text
context = concat(f64_up, f32_to_128)
A = sigmoid(Conv([f128, context]))
heat_feature = f128 * (1 + gamma * A)
gamma: learnable scalar, initial value 0.1
```

Heatmap output:

```text
heat_feature -> ConvBNAct -> 1x1 Conv -> heatmap logits: 1 x 128 x 128
```

Segmentation head input:

```text
dec2:          64 x 128 x 128
heat_feature: 64 x 128 x 128  # detach
heatmap_prob:  1 x 128 x 128  # sigmoid(heatmap logits), detach
concat:       129 x 128 x 128
```

Detach의 목적:

```text
heatmap loss는 heatmap branch로 정상 전파됨
segmentation loss는 heatmap branch로 직접 역전파되지 않음
```

즉 detach는 heatmap loss를 막는 장치가 아닙니다. Segmentation loss가 heatmap branch를 localization 목적과 다르게 흔드는 것을 막기 위한 장치입니다.

Segmentation attention:

```text
fused = Conv(concat(dec2, detached heat_feature, detached heatmap_prob))
context = concat(detached heat_feature, detached heatmap_prob)
A = sigmoid(Conv([fused, context]))
fused_att = fused * (1 + gamma * A)
```

Output:

```text
fused_att -> refine conv -> upsample x2 -> upsample x2 -> 1x1 conv
 -> stenosis logits: 1 x 512 x 512
```

## Heatmap Target

```text
GT stenosis mask 512 x 512
 -> area resize to 128 x 128
 -> max-pool dilation kernel 5
 -> Gaussian blur kernel 9, sigma 2.0
 -> image-wise max normalize
 -> soft heatmap target
```

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

```text
L_heatmap = 0.5 * DiceLoss + 0.5 * WeightedBCEWithLogits
positive weight = 2.0
```

Weighted BCE를 섞는 이유는 Dice-only heatmap loss가 넓은 false positive heatmap을 충분히 벌하지 못할 수 있기 때문입니다.

## Metrics

### Pixel-wise image F1

Semantic segmentation 보조 지표입니다.

```text
prediction = sigmoid(logit) > 0.5
target = binary mask
image-wise IoU / Precision / Recall / F1
```

### ARCADE official polygon-instance F1

공식 ARCADE stenosis evaluator 방식과 맞춘 주 지표입니다.

```text
binary prediction mask
 -> connected component contour extraction
 -> COCO polygon prediction
 -> polygon area-based F1 against GT polygon instances
 -> predicted instance별 max GT F1
 -> missing GT instances receive 0
 -> image-wise mean
 -> total mean
```

Checkpoint 저장 기준:

```text
monitor = val_arcade_polygon_instance_f1
mode = max
```

Scheduler 기준:

```text
monitor = val_loss
```

## Post-processing 계획

현재 visualization에는 closing이 적용됩니다.

```text
visualization prediction = closing(raw binary prediction, kernel=3)
```

공식 metric용 prediction은 raw threshold mask에서 contour를 추출합니다. 향후 validation set에서 다음 sweep으로 submission conversion hyperparameter를 고정할 수 있습니다.

```text
threshold: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
min_area: 1, 5, 10, 20, 30, 50
closing: none, 3, 5
opening: none, 3
```

가장 큰 connected component만 남기는 방식은 사용하지 않습니다. 한 이미지에 stenosis instance가 여러 개 있을 수 있기 때문입니다.

## 실행 예시

```powershell
C:\Code\workspace\venv\Scripts\python.exe main.py --model baseline --output-dir C:\Code\workspace\stenosis_heatmap\outputs\baseline_seed40 --seed 40
```

```powershell
C:\Code\workspace\venv\Scripts\python.exe main.py --model heatmap --output-dir C:\Code\workspace\stenosis_heatmap\outputs\heatmap_seed40 --seed 40
```

Test only:

```powershell
C:\Code\workspace\venv\Scripts\python.exe main.py --model heatmap --output-dir C:\Code\workspace\stenosis_heatmap\outputs\heatmap_seed40 --test-only
```

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
    └── stenosis_heatmap
```

`result.txt`에는 experiment config, checkpoint monitor, ARCADE official polygon-instance F1, pixel-wise image F1, heatmap metric, inference time이 저장됩니다.
