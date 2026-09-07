# BEVFusion 신규 아키텍처

## 1. 목적과 선택 방법

이 브랜치는 별도 학습 프로젝트를 만들지 않고 기존 `configs/`, `mmdet3d/`, `tools/` 구조에 신규 모델을 추가한다. 기존 BEVFusion 코드는 그대로 선택할 수 있으며 모델 종류와 카메라 수는 일반 config 파일로 결정한다.

| 용도 | 실행 config | 모델 | 카메라 |
|---|---|---|---|
| 팀 기존 구성 | `configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml` | `BEVFusion` | 전방 3대 |
| NVIDIA BEVFusion 구성 | `configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser_6cam.yaml` | `BEVFusion` | nuScenes 6대 |
| 신규 구성 | `configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml` | `DynamicBEVFusion` | nuScenes 6대 |

`configs/nuscenes/default.yaml`의 기본값은 팀의 기존 3-camera 동작을 보존한다. 6-camera config는 `camera_names`만 명시적으로 재정의한다. `NuScenesDataset`이 더 이상 카메라 세 대를 소스에 하드코딩하지 않으므로 config를 복사해 카메라 조합을 변경할 수 있다.

## 2. 아키텍처

```mermaid
flowchart LR
    CFG["선택한 YAML config"] --> BF["기존 BEVFusion"]
    CFG --> DBF["신규 DynamicBEVFusion"]

    subgraph LEGACY["기존 경로: BEVFusion"]
      I0["3 또는 6 camera images"] --> C0["ResNet-50 + GeneralizedLSSFPN"]
      C0 --> V0["DepthLSS view transform"]
      P0["LiDAR points"] --> L0["Voxelization + SparseEncoder"]
      V0 --> F0["ConvFuser"]
      L0 --> F0
      F0 --> D0["SECOND + SECONDFPN"]
      D0 --> H0["TransFusionHead"]
    end

    subgraph NEW["신규 경로: DynamicBEVFusion"]
      I1["6 camera images<br/>[B,6,3,256,704]"] --> C1["ResNet-34 + LSSFPN"]
      C1 --> W1["WidthFormerTransform"]
      W1 --> CB["camera BEV<br/>[B,80,180,180]"]

      P1["통합 LiDAR points<br/>List([Ni,5])"] --> DV["Dynamic Pillar VFE"]
      DV --> DS["DSVT set attention<br/>4 blocks"]
      DS --> LN["Dense scatter + DSVT BEV neck"]
      LN --> LB["LiDAR BEV<br/>[B,256,180,180]"]

      CB --> DG["DepthGFusion<br/>depth-guided deformable attention"]
      LB --> DG
      DG --> FB["fused BEV<br/>[B,256,180,180]"]
      FB --> DEC["SECOND + SECONDFPN"]
      DEC --> FH["fused feature: classification"]
      LB --> RH["LiDAR-only feature: box regression"]
      FH --> DAL["DALDecoupledHead"]
      RH --> DAL
      DAL --> OUT["train: loss dict<br/>eval: boxes / scores / labels"]
    end
```

신규 모델은 기존 `BEVFusion`을 상속한다. 카메라와 LiDAR branch 뒤의 결합 방식만 `DepthGFusion`으로 교체하고, DAL 아이디어에 따라 fused feature는 분류에, LiDAR-only feature는 box regression에 사용한다. 신규 구현은 PyTorch 모듈이며 별도 CUDA extension을 추가하지 않는다.

## 3. 추가·변경 파일

### 신규 모델 파일

| 파일 | registry 이름과 역할 |
|---|---|
| `mmdet3d/models/backbones/dsvt.py` | `DSVTLidarEncoder` registry adapter |
| `mmdet3d/models/backbones/dsvt_core.py` | raw point batch → dynamic pillar → DSVT → LiDAR BEV |
| `mmdet3d/models/vtransforms/widthformer.py` | `WidthFormerTransform`; calibration-aware camera BEV |
| `mmdet3d/models/fusers/depth_guided_attention.py` | depth-guided deformable cross-attention |
| `mmdet3d/models/fusers/depth_gfusion.py` | `DepthGFusion`; camera/LiDAR BEV 결합 |
| `mmdet3d/models/heads/bbox/dal_decoupled.py` | `DALDecoupledHead`; 분류/회귀 feature decoupling |
| `mmdet3d/models/fusion_models/dsvt_bevfusion.py` | 기존 BEVFusion에서 raw-point DSVT encoder만 교체 |
| `mmdet3d/models/fusion_models/dynamic_bevfusion.py` | `DynamicBEVFusion`; 전체 신규 경로 조립 |
| `mmdet3d/models/utils/bev_grid.py` | BEV 크기·축 순서 검증 |

각 패키지의 `__init__.py`에는 위 registry 모듈 import만 추가했다.

### Config·dataset·검증 파일

| 파일 | 변경 내용 |
|---|---|
| `configs/nuscenes/default.yaml` | 기본 3-camera 목록 및 train/val/test dataset 전달 |
| `configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser_6cam.yaml` | 기존 BEVFusion의 6-camera 선택 config |
| `configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml` | 신규 아키텍처 전체 config |
| `mmdet3d/datasets/nuscenes_dataset.py` | `camera_names` 지원, 3-camera 하드코딩 제거, intrinsic key 호환 |
| `mmdet3d/models/fusion_models/bevfusion.py` | non-depth transform도 호출 가능하도록 `depths=None` 기본값 추가 |
| `tools/smoke_train.py` | 표준 config로 실제 mini batch의 forward/backward/optimizer 검증 |

## 4. 신규 모델 입출력 계약

`DynamicBEVFusion.forward()`의 주요 입력은 다음과 같다.

| 이름 | 형식 | 의미 |
|---|---|---|
| `img` | `FloatTensor[B,N,3,256,704]` | 개별 카메라 영상. 현재 신규 config의 `N=6` |
| `points` | 길이 `B`의 list, 원소 `FloatTensor[Ni,5]` | 통합 LiDAR point cloud. 현재 nuScenes loader가 선택한 5개 column을 그대로 사용 |
| `camera2ego`, `lidar2camera`, `lidar2image`, `camera2lidar` | `FloatTensor[B,N,4,4]` | 카메라별 calibration transform |
| `lidar2ego` | `FloatTensor[B,4,4]` | LiDAR-to-ego transform |
| `camera_intrinsics` | `FloatTensor[B,N,4,4]` | camera intrinsic matrix |
| `img_aug_matrix`, `lidar_aug_matrix` | `FloatTensor[B,N,4,4]`, `FloatTensor[B,4,4]` 중심 | 학습 augmentation transform |
| `metas` | batch metadata | bbox decoding에 필요한 sample metadata |
| `gt_bboxes_3d`, `gt_labels_3d` | batch list | 학습 target |

학습 모드 출력은 `loss/object/*`와 `stats/object/*` key를 가진 dict다. 평가 모드 출력은 sample별 `boxes_3d`, `scores_3d`, `labels_3d` dict다.

신규 PyTorch DSVT 입력의 `Ni`는 고정 shape가 아니며 batch마다 달라도 된다. 다섯 대 solid-state LiDAR의 좌표계와 timestamp가 상위 모듈에서 하나의 ego/LiDAR 기준으로 정렬·병합되어 들어온다는 전제다. 현재 구현에는 10,000-point 같은 임의 절단은 없다. 다만 1,000,000 points의 학습·추론 메모리와 latency는 아직 실측하지 않았으므로 지원 완료로 간주하지 않는다. TensorRT의 고정 capacity와 35 ms 추론 목표도 이 PyTorch 학습 smoke의 검증 범위가 아니다.

## 5. 환경과 실행

```bash
cd /home/culee/workspace/bevfusion
source /home/culee/2608_bevfusion/bin/activate

# 최초 1회: 이 저장소의 기존 CUDA extension 설치
CPLUS_INCLUDE_PATH=/tmp/bevfusion-python-dev/extracted/usr/include/python3.8:/tmp/bevfusion-python-dev/extracted/usr/include \
  MAX_JOBS=8 pip install -e . --no-deps --no-build-isolation

# 로컬 mini 데이터
mkdir -p data
ln -s /home/culee/nuscenes_mini data/nuscenes

# 신규 아키텍처 실제 1-step smoke
python tools/smoke_train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml \
  --dataroot /home/culee/nuscenes_mini --device 0

# 전체 학습
torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3.yaml \
  --run-dir runs/new-arch
```

mini map-expansion 파일이 없는 환경에서 기존 detection config만 smoke할 때는 `tools/smoke_train.py ... --object-only`를 사용한다. 이는 detection 입력과 loss를 유지하고 map segmentation label loader만 제외한다.

## 6. 2026-09-07 검증 결과

- PASS: Python 문법 검사.
- PASS: venv의 `mmdet3d`가 `/home/culee/workspace/bevfusion/mmdet3d`를 가리킴.
- PASS: 기존 CUDA extension 12개 빌드 및 신규 registry 5종 등록.
- PASS: 3-camera BEVFusion, 6-camera BEVFusion, 6-camera DynamicBEVFusion config 모두 recursive resolve.
- PASS: 신규 DynamicBEVFusion의 실제 nuScenes-mini 1-step 학습.
  - 입력: 6 views, 32,250 points, 15 boxes
  - 모델 파라미터: 44,521,340
  - 실행: forward, backward, optimizer step
  - gradient parameter 수: camera 190, LiDAR 194, fusion 54, head 42
  - peak CUDA allocated: 4.612 GiB

## 7. 참고 구현

- DSVT: <https://github.com/Haiyang-W/DSVT>
- WidthFormer: <https://github.com/ChenhongyiYang/WidthFormer>
- DepthFusion DGF: <https://github.com/Mingqj/DepthFusion>
- DAL head: <https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/mmdet3d/models/dense_heads/dal_head.py>

전체 외부 저장소를 복제하지 않았으며, 현재 BEVFusion 인터페이스에 필요한 아이디어와 연산 경계만 옮겼다.
