# Phase 1c 실행 기록 — 공식 DSVT 수치 동치 검증

## 결론

**PASS.** 두 nuScenes LiDAR sample 모두 pillar 좌표 집합이 완전히 같았고,
VFE부터 공식 `spatial_features_2d`에 대응하는 BEV 2D backbone 출력까지 모든
단계의 max absolute difference가 허용 오차 `1e-3`보다 작았다. 관측된 전체
최대값은 `4.27141786e-05`였다. 따라서 처음 어긋나는 단계는 없다.

## 환경과 실행

저장소 루트에서 다음 명령을 실행했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python tools/dsvt_pretrained/equiv_check.py
```

- Python: `3.9.5`
- PyTorch: `1.12.0+cu116`, CPU (`torch.cuda.is_available() == False`)
- dtype/mode: float32, eval
- thread 수: 2
- 판정 기준: 각 단계 `max_abs_diff <= 1e-3`
- 상대 오차 정의: `L2(ours - official) / L2(official)`
- 입력: 파일명 정렬 결과의 첫 두 `.pcd.bin`; 두 경로 모두 range filter 후
  동일한 `[x, y, z, intensity, 0]`을 사용했다. 마지막 채널은 공식 경로의
  timestamp 및 우리 경로의 ring을 모두 0으로 만든 값이다.

공식 checkpoint에서는 VFE/DSVT/scatter/BEV backbone prefix에 속하는 336개
tensor를 이름 변경 없이 vendored 공식 모듈에 `strict=True`로 로드했다.
변환 checkpoint에서는 `encoders.lidar.backbone.` prefix만 제거해 339개
tensor를 로드했다. 공식 checkpoint에 존재하지 않는 최종 384→256 adapter의
state 6개만 초기화되지 않았다. PyTorch의 BatchNorm 하위 호환 규칙 때문에
`num_batches_tracked`를 제외한 5개만 loader의 missing 목록에 나타나며,
unexpected key는 0개다. 비교는 이 adapter 직전의 384채널 공식 출력에서
수행했다.

## Pillar 좌표 집합

| sample timestamp | raw points | range-filtered | official pillars | ours pillars | 집합 일치 | official-only | ours-only |
|---|---:|---:|---:|---:|:---:|---:|---:|
| `1533151603547590` | 34,752 | 33,623 | 3,687 | 3,687 | yes | 0 | 0 |
| `1533151604048025` | 34,688 | 33,530 | 3,774 | 3,774 | yes | 0 | 0 |

Pillar feature와 각 DSVT block 출력은 `(batch, z, y, x)` 좌표 키로 각각
정렬한 뒤 비교했다.

## 단계별 수치 결과

두 sample 중 가장 나쁜 값을 모은 표다. cosine은 두 sample 중 최솟값이다.

| 단계 | max abs diff | max relative L2 diff | min cosine | 판정 |
|---|---:|---:|---:|:---:|
| VFE pillar feature | `2.93646008e-06` | `1.16759475e-06` | `0.999999999999` | PASS |
| DSVT block 0 | `7.09295273e-06` | `1.99389342e-06` | `0.999999999998` | PASS |
| DSVT block 1 | `6.75953925e-06` | `1.85886881e-06` | `0.999999999998` | PASS |
| DSVT block 2 | `6.10947609e-06` | `2.35675111e-06` | `0.999999999997` | PASS |
| DSVT block 3 | `1.38357282e-05` | `2.91487981e-06` | `0.999999999996` | PASS |
| dense BEV (`spatial_features`) | `1.38357282e-05` | `2.91489323e-06` | `0.999999999996` | PASS |
| BEV backbone (`spatial_features_2d`) | `4.27141786e-05` | `3.17731906e-06` | `0.999999999995` | PASS |

Sample별 원 수치와 shape는 `equiv_report.json`에 기록했다.

## 공식 참조 vendoring

`official_ref/model.py`는 `/tmp/DSVT`의 DynamicPillarVFE, DSVT input layer,
DSVT blocks, PointPillarScatter3d, BaseBEVResBackbone을 nuScenes 단일-stage
설정으로 고정한 CPU 참조다. OpenPCDet registry/config 의존을 제거했고,
`ingroup_inds` CUDA op만 그룹 id별 입력 등장 순서를 반환하는 순수 PyTorch
구현으로 대체했다. 원본 파일 목록, 출처와 Apache-2.0 고지는
`official_ref/NOTICE.md`에 기록했다.

## 진단

허용 오차를 넘는 단계가 없으므로 구조 의미가 다른 위치에 가중치가
이식됐다는 가설은 이 두 실제 sample의 LiDAR 경로에서 지지되지 않는다.
남은 작은 차이는 CPU reduction/연산 순서에 따른 float32 반올림 범위이며,
VFE에서 시작해 깊이에 따라 완만히 누적되는 형태다. `mmdet3d/` 코드는
수정하지 않았다.
