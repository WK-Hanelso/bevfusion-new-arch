# BEV 레이아웃 규약

모듈형 융합 경로의 표준 BEV 텐서는 `[B, C, Y, X]`이며
`CANONICAL_BEV_LAYOUT = "yx"`이다. 마지막 두 축에서 행은 y, 열은 x에
대응한다. 레이아웃 선언은 텐서의 메모리 배치만 설명하며 좌표계나 물리적
범위를 바꾸지 않는다.

| 모듈 | `bev_layout` | 출력 텐서 | 비고 |
|---|---:|---|---|
| `SparseEncoder` | `xy` | `[B,C,X,Y]` | 레거시 sparse dense 출력 |
| `PointPillarsScatter` / `PointPillarsEncoder` | `xy` | `[B,C,X,Y]` | 레거시 scatter 출력 |
| `BaseTransform` 계열 (`LSS`, `DepthLSS`, `AwareBEVDepth` 등) | `xy` | `[B,C,X,Y]` | `bev_pool` 사용 계열, 선언 상속 |
| `DSVTLidarEncoder` | `yx` | `[B,C,Y,X]` | OpenPCDet 규약 |
| `WidthFormerTransform` | `yx` | `[B,C,Y,X]` | 직접 표준 출력 |
| `TransFusionHead` | 기본 `xy`, 설정 가능 | 입력과 타깃이 같은 규약 | 레거시 `BEVFusion` 기본 동작 유지 |
| `DALDecoupledHead` | 기본 `yx` | `[B,C,Y,X]` | 부모의 `yx` grid/타깃 구현 재사용 |

`ModularBEVFusion.forward_single`은 카메라 vtransform 또는 LiDAR encoder의
출력을 받은 직후 `bev_layout`을 검사한다. `xy`이면
`transpose(-1, -2).contiguous()`로 표준화하고, `yx`이면 그대로 전달한다.
선언이 없거나 `xy`/`yx` 이외 값이면 즉시 `ValueError`를 낸다. 외부에서
주입한 `camera_bev`는 이미 표준 `yx`인 것으로 간주한다.

따라서 fuser와 decoder는 언제나 `[B,C,Y,X]`만 받는다. 모듈형 모델은 object
head 생성 설정에도 `bev_layout="yx"`를 강제로 주입한다. `DepthGFusion`에는
카메라와 LiDAR가 모두 표준화된 뒤 들어가므로 기존 기본값
`camera_layout="yx"`, `lidar_layout="yx"`를 유지한다.

레거시 `BEVFusion` 경로에는 이 정규화나 head 설정 주입이 없다.
`TransFusionHead`의 기본값도 `xy`이므로 기존 `[B,C,X,Y]` 체인은 바뀌지
않는다. `DynamicBEVFusion`은 계속 `ModularBEVFusion` 호환 클래스다.
