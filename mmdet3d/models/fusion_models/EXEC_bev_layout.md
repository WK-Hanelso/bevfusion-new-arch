# EXEC — BEV 레이아웃 규약 구현 검증

실행일: 2026-09-20 (KST)  
작업 위치: 저장소 루트  
커밋 명령: 이 작업에서는 실행하지 않음

## 구현 결과

- 표준 상수는 `CANONICAL_BEV_LAYOUT = "yx"` (`[B,C,Y,X]`)이다.
- `SparseEncoder`, `BaseTransform` 계열, PointPillars 계열은 `xy`를 선언한다.
- `DSVTLidarEncoder`, `WidthFormerTransform`은 `yx`를 선언한다.
- `ModularBEVFusion`은 센서 특징 생성 직후 선언을 검사하고 `xy`만
  `transpose(-1, -2).contiguous()`로 바꾼다. 외부 `camera_bev`는 `yx`로
  간주한다.
- 모듈형 object head 설정에는 `yx`를 주입한다. `TransFusionHead` 기본값은
  레거시 호환을 위해 `xy`이고 `DALDecoupledHead` 기본값은 `yx`이다.
- DAL의 `yx` grid/heatmap 구현을 `TransFusionHead`로 이동하고 DAL이 상속해
  재사용한다. `DepthGFusion`의 레이아웃 기본값과 연산 코드는 수정하지 않았다.
- 레거시 `BEVFusion` 구현은 수정하지 않았다. `DynamicBEVFusion`의 상속 관계도
  수정하지 않았다.

## 호스트 검증 (`mmdet3d` import 없음)

실행:

```bash
pytest -q tests/test_bev_layout.py tests/test_spconv2_equiv.py
```

결과: **9 passed, 5 skipped**, 1.04초. skip은 Docker의 mmdet3d/mmcv 또는 GPU가
필요한 항목이다. BEV 레이아웃 테스트 단독 결과는 **5 passed, 4 skipped**였다.

확인 수치:

| 검사 | 결과 |
|---|---:|
| 직접 YAML 파싱한 ablation leaf | 16/16 |
| 각 leaf의 LiDAR·camera 클래스 레이아웃 선언 | 16/16 통과 |
| 각 leaf의 object head `yx` 주입 | 16/16 통과 |
| `yx` 타깃 최대 위치 (`x_idx=3`, `y_idx=2`) | `[row=2, col=3]` |
| `yx` grid의 flat index | `2*8+3 = 19`, 값 `(3.5, 2.5)` |
| 레거시 `xy` 타깃 최대 위치 | `[row=3, col=2]` |
| 레거시 `xy` grid의 flat index | `3*6+2 = 20`, 값 `(3.5, 2.5)` |
| `xy [1,1,2,3]` 정규화 출력 shape | `[1,1,3,2]`, contiguous |
| 기존 DAL vs 부모로 이동한 `yx` target 최대 절대차 | `0.0` |
| 기존 DAL vs 부모로 이동한 `yx` grid 최대 절대차 | `0.0` |

DAL 비교는 현재 파일과 `git show HEAD:.../dal_decoupled.py`의 두 좌표 메서드를
AST로 직접 실행한 결과다. DSVT, WidthFormer, DepthGFusion의 forward 연산에는
변경이 없으며 DSVT/WidthFormer에는 클래스 선언만 추가했다.

추가 정적 검증:

```bash
python -m compileall -q mmdet3d/models/fusion_models \
  mmdet3d/models/heads/bbox mmdet3d/models/backbones/sparse_encoder.py \
  mmdet3d/models/backbones/dsvt.py mmdet3d/models/backbones/dsvt_core.py \
  mmdet3d/models/backbones/pillar_encoder.py mmdet3d/models/vtransforms \
  tests/test_bev_layout.py
git diff --check
```

결과: 두 명령 모두 성공했다.

## Docker CPU 검증

테스트에는 다음 Docker 전용 검사가 들어 있다.

1. 실제 `TransFusionHead`의 `xy`/`yx` grid 및 heatmap 왕복
2. 가짜 `xy` LiDAR encoder를 넣은 `ModularBEVFusion.forward_single` 전치 경로
3. A1의 실제 `DSVTLidarEncoder -> SECOND -> SECONDFPN -> TransFusionHead` 합성
   배치 forward/loss (`BboxOverlaps3D`만 CPU 대체 assigner로 격리)

요구된 대로 `--gpus`를 쓰지 않았고, 호스트 `mmdet3d/ops`를 마운트하지
않았다. `mmdet3d/models`, `mmdet3d/core`, `mmdet3d/datasets`, `configs`, `tests`,
`tools`만 마운트하고 `/tmp/bev_layout_torchhome`을 `TORCH_HOME`으로 사용했다.

실행 결과: **환경 차단으로 미실행**. Docker daemon 접속 단계에서 아래 오류가
발생해 이미지 내부 pytest는 시작되지 않았다.

```text
docker: permission denied while trying to connect to the Docker daemon socket at
unix:///var/run/docker.sock: ... connect: operation not permitted.
```

따라서 Docker 3개 항목과 A1 합성 loss의 실제 수치는 이 실행 기록에서 PASS로
표시하지 않는다. 재실행 명령은 다음과 같다.

```bash
mkdir -p /tmp/bev_layout_torchhome
docker run --rm --shm-size=8g --user "$(id -u):$(id -g)" \
  -v "$PWD/mmdet3d/models:/workspace/bevfusion/mmdet3d/models:ro" \
  -v "$PWD/mmdet3d/core:/workspace/bevfusion/mmdet3d/core:ro" \
  -v "$PWD/mmdet3d/datasets:/workspace/bevfusion/mmdet3d/datasets:ro" \
  -v "$PWD/configs:/workspace/bevfusion/configs:ro" \
  -v "$PWD/tests:/workspace/bevfusion/tests:ro" \
  -v "$PWD/tools:/workspace/bevfusion/tools:ro" \
  -v "/tmp/bev_layout_torchhome:/workspace/cache" \
  -e HOME=/workspace/cache -e TORCH_HOME=/workspace/cache/torch \
  -e PYTHONPATH=/workspace/bevfusion -w /workspace/bevfusion \
  bevfusion-train:cu113 bash -lc 'pytest -q tests/test_bev_layout.py'
```

## 범위 확인

- `tools/` 파일은 수정하지 않았다.
- 시작 시 이미 존재하던 untracked `agent/`, `experiments/overfit/`,
  `tools/ablation/EXEC_wrapper_equiv.md`, `wrapper_equiv_*` 파일은 건드리지 않았다.
- 패키지를 설치하지 않았고 이 작업에서는 `git commit`을 실행하지 않았다.
- 작업 도중 공유 worktree의 HEAD가 외부에서 `a4d9cb0`에서 `091901a`로
  이동했다. 후자의 history-only 커밋에 이 작업이 추가한 §2 행이 함께
  포함됐으며, 이를 되돌리거나 새 커밋을 만들지는 않았다.
