# Phase 2 ablation config family

이 계열은 학습 코드를 바꾸지 않고 config 이름만 선택해 LiDAR encoder,
camera view transform, fuser, object head를 비교한다. 모든 leaf는 nuScenes
6-camera, single sweep, GT-Aug 미사용, CBGS, 256×704 입력, 동일 augmentation,
AdamW/cyclic schedule, 20 epochs를 공유한다.

## 전체 16개 조합

파일명의 네 bit는 차례로 DSVT, WidthFormer, GFusion, DAL 사용 여부다.

| canonical config | LiDAR | camera vtransform | fuser | head |
|---|---|---|---|---|
| `dsvt0_wf0_gf0_dal0.yaml` | SparseEncoder | DepthLSS | ConvFuser | TransFusion |
| `dsvt0_wf0_gf0_dal1.yaml` | SparseEncoder | DepthLSS | ConvFuser | DAL |
| `dsvt0_wf0_gf1_dal0.yaml` | SparseEncoder | DepthLSS | DepthGFusion | TransFusion |
| `dsvt0_wf0_gf1_dal1.yaml` | SparseEncoder | DepthLSS | DepthGFusion | DAL |
| `dsvt0_wf1_gf0_dal0.yaml` | SparseEncoder | WidthFormer | ConvFuser | TransFusion |
| `dsvt0_wf1_gf0_dal1.yaml` | SparseEncoder | WidthFormer | ConvFuser | DAL |
| `dsvt0_wf1_gf1_dal0.yaml` | SparseEncoder | WidthFormer | DepthGFusion | TransFusion |
| `dsvt0_wf1_gf1_dal1.yaml` | SparseEncoder | WidthFormer | DepthGFusion | DAL |
| `dsvt1_wf0_gf0_dal0.yaml` | DSVT | DepthLSS | ConvFuser | TransFusion |
| `dsvt1_wf0_gf0_dal1.yaml` | DSVT | DepthLSS | ConvFuser | DAL |
| `dsvt1_wf0_gf1_dal0.yaml` | DSVT | DepthLSS | DepthGFusion | TransFusion |
| `dsvt1_wf0_gf1_dal1.yaml` | DSVT | DepthLSS | DepthGFusion | DAL |
| `dsvt1_wf1_gf0_dal0.yaml` | DSVT | WidthFormer | ConvFuser | TransFusion |
| `dsvt1_wf1_gf0_dal1.yaml` | DSVT | WidthFormer | ConvFuser | DAL |
| `dsvt1_wf1_gf1_dal0.yaml` | DSVT | WidthFormer | DepthGFusion | TransFusion |
| `dsvt1_wf1_gf1_dal1.yaml` | DSVT | WidthFormer | DepthGFusion | DAL |

## 기존 이름 별칭

별칭은 대응 canonical과 YAML 내용이 같으며, 생성된 헤더에 canonical 이름을
기록한다.

| alias | canonical |
|---|---|
| `b0_legacy.yaml` | `dsvt0_wf0_gf0_dal0.yaml` |
| `e1_dsvt.yaml` | `dsvt1_wf0_gf0_dal0.yaml` |
| `e2_widthformer.yaml` | `dsvt0_wf1_gf0_dal0.yaml` |
| `e3_gfusion.yaml` | `dsvt0_wf0_gf1_dal0.yaml` |
| `e4_dal.yaml` | `dsvt0_wf0_gf0_dal1.yaml` |
| `c2_dsvt_widthformer.yaml` | `dsvt1_wf1_gf0_dal0.yaml` |
| `c3_dsvt_widthformer_gfusion.yaml` | `dsvt1_wf1_gf1_dal0.yaml` |
| `c4_full.yaml` | `dsvt1_wf1_gf1_dal1.yaml` |

공통 B0는 ResNet-34 + LSSFPN(`in_indices: [2,1]`, stride 16),
DepthLSSTransform, voxel 0.075의 SparseEncoder, ConvFuser,
TransFusionHead(`out_size_factor: 8`)다. 원 BEVFusion camera+LiDAR 기준은
ResNet-50 + GeneralizedLSSFPN(stride 8)이므로 B0는 그 config의 단순 별칭이
아니다. camera backbone과 실험 조건을 신규 edge target에 맞춰 고정한 비교
기준이다.

DSVT leaf는 voxel 0.3, `out_size_factor: 2`, grid 360×360×1을 함께 선택한다.
모든 encoder/fuser 출력은 180×180이고 decoder 입력은 256 channels다.

## 학습

일반 형식은 다음과 같다.

```bash
BEVFUSION_CONFIG=configs/nuscenes/det/ablation/dsvt0_wf0_gf0_dal0.yaml
torchpack dist-run -np 1 python tools/train.py "$BEVFUSION_CONFIG" \
  --run-dir runs/ablation/dsvt0_wf0_gf0_dal0
```

`dsvt0_*` 조합은 config와 run directory를 바꿔 같은 명령을 사용한다.
`dsvt1_*` 조합은 변환된 공식 checkpoint를 명시적으로 로드한다. leaf에는
`load_from` 기본값이 없다.

```bash
BEVFUSION_CONFIG=configs/nuscenes/det/ablation/dsvt1_wf0_gf0_dal0.yaml
torchpack dist-run -np 1 python tools/train.py \
  "$BEVFUSION_CONFIG" \
  --run-dir runs/ablation/dsvt1_wf0_gf0_dal0 \
  --load_from pretrained/dsvt_nuscenes_official_lidar.pth
```

공식 checkpoint 생성과 적용 범위는
[`tools/dsvt_pretrained/README.md`](../dsvt_pretrained/README.md)를 따른다.

## 생성·검사와 server smoke

leaf는 직접 편집하지 않고 생성기로 갱신한다.

```bash
python tools/ablation/gen_configs.py
python tools/ablation/gen_configs.py --check
pytest -q tools/ablation/tests
bash -n tools/ablation/smoke_all.sh
```

실제 1-step forward/backward는 CUDA, mmcv, torchpack, nuScenes data가 있는
서버에서 실행한다. canonical config 16개를 순차 실행하고 결과를
`tools/ablation/smoke_results.md`에 append한다.

```bash
DATAROOT=/data/nuscenes DEVICE=0 bash tools/ablation/smoke_all.sh
```

## 단일 센서 dispatch 계약

`ModularBEVFusion`은 feature를 `camera`, `lidar` key로 보관하며 두 센서가
있을 때만 선택된 fuser를 호출한다. 단일 센서는 fuser를 우회한다. DAL은
LiDAR가 있으면 encoder의 LiDAR BEV를 회귀 입력으로 사용하고, camera-only면
decoder 출력을 회귀 입력 대체값으로 사용한다. 표준 TransFusion head는 항상
decoder 출력과 metadata만 받는다.
