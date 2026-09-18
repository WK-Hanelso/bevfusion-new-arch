# Orin 실측 — DSVT LiDAR 인코더 기존 구조 vs 공식 구조(pretrained) (2026-09-17)

목적: 공식 DSVT 체크포인트를 쓰기 위해 LiDAR 분기를 공식 구조(`official_layout=True`)로 바꿀 때의 추론 비용 실측 ("손해 1" 정량화).
환경: Jetson AGX Orin, JetPack 5.1.2 (L4T R35.4.1), **NV Power Mode 15W(기본, 변경 안 함)**, 컨테이너 `dustynv/l4t-pytorch:r35.4.1` (torch 2.0.0+nv23.05, TensorRT 8.5.2.2, torch_scatter 없음 → 순수 torch fallback).
입력: 합성 점군 34,688 points (Thor 벤치와 동일 수), 5 feature. 코드: `bevfusion` branch `dsvt-pretrained-init` 커밋 68d23d0의 `dsvt_core.py` 단독 실행. 벤치 스크립트·결과 JSON은 Orin `~/bench_dsvt/`.
주의: Thor 배포 엔진(CUDA 13/TRT 10.13, plugin 5개)은 Orin(TRT 8.5)에서 실행 불가. 아래는 **PyTorch eager**(전체) 및 **TensorRT FP16**(BEV backbone 부분만) 측정이며 Thor 26.4ms와 직접 비교하지 않는다.

## 1. PyTorch eager, 전체 LiDAR 인코더 (warmup 20, 100회, p50 ms)

| 점군 분포 | 점유 pillar | 구조 | 정밀도 | params | total | vfe | DSVT backbone | scatter+neck |
|---|---:|---|---|---:|---:|---:|---:|---:|
| std 18 m (과밀) | 24,826 | legacy | FP32 | 3.74M | 1081.5 | 20.9 | 921.2 | 138.9 |
| | | legacy | FP16 | 3.74M | 738.4 | 20.0 | 629.4 | 88.2 |
| | | official | FP32 | 6.32M | 1227.8 | 21.8 | 956.3 | 249.3 |
| | | official | FP16 | 6.32M | 829.3 | 20.7 | 665.8 | 142.5 |
| **std 9 m (실제 밀도 근사)** | **13,545** | legacy | FP32 | 3.74M | 668.6 | 20.0 | 509.5 | 138.5 |
| | | legacy | FP16 | 3.74M | 464.6 | 18.6 | 357.2 | 88.5 |
| | | official | FP32 | 6.32M | 798.2 | 21.1 | 527.6 | 249.2 |
| | | official | FP16 | 6.32M | **537.6** | 19.1 | 376.1 | **142.0** |

출력 finite 확인: 4/4 True, shape [1,256,180,180]. pretrained 로드: unexpected 0, missing = neck.adapter만.

## 2. 해석 (eager 기준)

- 구조 변경 비용은 거의 전부 **BEV backbone(scatter+neck)**에서 나온다: FP16 88.5 → 142.0 ms (+53.5 ms, +60%). 파라미터 2.2M → 4.8M, conv 수 약 2배와 일치.
- DSVT attention 블록에 추가된 LayerNorm 8개의 비용은 FP16 +19 ms(357→376, +5%)로 작다.
- DSVT backbone 자체가 eager에서 357~630 ms로 지배적인데, 이는 python 수준 set 분할·gather가 원인이며 Thor에서는 plugin으로 대체되는 부분. 따라서 **eager 절대치는 배포 latency가 아니다.**
- 배포 관점에서 의미 있는 숫자는 §3의 TensorRT BEV backbone 비교다.

## 3. TensorRT FP16, BEV backbone(+adapter)만, 입력 [1,128,360,360] (trtexec, 15W)

(빌드 진행 중 — 완료 후 기입)

trtexec `--fp16 --warmUp=500 --iterations=100 --avgRuns=10`, 저장 엔진 로드 후 재측정. "Latency" = H2D+compute+D2H (trtexec end-to-end), H2D ≈ 7.5 ms(입력 128×360×360 FP32 66MB).

| 구조 | ONNX | engine | Latency median | H2D median | compute 근사 (Latency−H2D) |
|---|---:|---:|---:|---:|---:|
| legacy `DSVTBEVNeck` | 10.1 MB | 5.3 MB | 54.87 ms | 7.51 | ≈ 47 ms |
| official `DSVTBEVResNeck` (pretrained) | 20.4 MB | 10.4 MB | 85.27 ms | 7.52 | ≈ 78 ms |

**Delta ≈ +30 ms (+65%) at Orin 15W FP16.** eager의 +60%와 일치. Thor(정격)에서는 절대치가 크게 줄지만 비율은 비슷할 것으로 예상 → Thor Engine B 12.6 ms 중 2D backbone 몫이 x라면 약 +0.65x. 정확한 Thor 수치는 S1 이후 Engine B 재빌드로 확정.

## 4. 결론

- 공식 pretrained 채택의 추론 비용은 BEV backbone에서 약 +65%(부분), 인코더 전체 기준 eager +16%. LayerNorm 추가 비용은 무시 가능.
- 사용자 결정(2026-09-17): **공식 구조를 기본 구조로 채택**. MAXN 재측정은 나중에 일괄.
- 발견된 결함: `deployment/onnx/dsvt_backbone.py`가 `layer_norms`를 순회하지 않음 → 공식 구조 export 시 PyTorch와 불일치. Phase 1b에서 수정.

## 5. Orin TensorRT 8.5 엔진 (진단 ckpt = c4_full + DSVT 공식 init, FP16, 2026-09-19)

| 엔진 | 빌드 | 크기 | trtexec Latency median (15W) | H2D | 비고 |
|---|---|---:|---:|---:|---|
| A camera_bev (6×3×256×704 + geometry) | 성공, 27분@15W | 62.3 MB | 15W: 105.2 ms(e2e) / **MAXN: GPU compute 16.7 ms**(H2D 0.8) | | Thor 정격 5.0 ms 대비 약 3.3배. MAXN=1.3GHz |
| B dsvt_lidar (plugin 5개, raw points 34,688) | **빌드 성공**(원인 연쇄 5단계 해결: add_cast → identity-cast bool → np.bool shim → Cast→BOOL 제거 → new_full segfault/full_like 1GB 상수 → 산술 마스킹) | 14.0 MB | MAXN: GPU compute **53.1 ms**, e2e 54.8 | 0.05 | Thor 정격 12.6 ms 대비 약 4.2배. trtexec 합성 입력 기준 |
| C fusion_dal | r3(squeeze→정적 인덱싱)로 If 제거 후 **빌드 성공** | 36.7 MB | MAXN: 22.4 ms(e2e), H2D 2.2 → compute ≈ 20 ms | | Thor 정격 8.8 ms 대비 약 2.3배 |

MAXN 전환: `nvpmodel -m 0`은 재부팅 필요 → **사용자 승인 후 2026-09-19 00:25 재부팅, MAXN(모드 0) + jetson_clocks 적용 확인**(GPU 1,300 MHz, CPU 12코어). 이후 측정은 MAXN 기준.

## 6. 요약 (2026-09-19 01:10, Orin MAXN, TensorRT 8.5.2 FP16, 진단 ckpt)

| 엔진 | Orin MAXN compute | Thor 정격(2026-09-03) | 비율 |
|---|---:|---:|---:|
| A camera | 16.7 ms | 5.0 ms | 3.3× |
| B DSVT LiDAR | 53.1 ms | 12.6 ms | 4.2× |
| C fusion+DAL | ≈20 ms | 8.8 ms | 2.3× |
| 직렬 합 | ≈90 ms | 26.4 ms | 3.4× |

**DSVT를 포함한 세 엔진이 Orin(TRT 8.5)에서 전부 빌드·실행됨.** 병렬 A/B/C E2E는 C++ 런타임 컴파일 후 측정. 정확도 비교(pth↔TRT)는 학습된 가중치 확보 후.

## 7. C++ 런타임 E2E (Orin MAXN, TRT 8.5, r6 export 엔진 3개, 34,688 points, warmup 20, 100회)

| 항목 | single-thread-submit | parallel-submit |
|---|---:|---:|
| engine_a p50 | 16.30 ms | 16.32 ms |
| engine_b p50 | 53.23 ms | 53.15 ms |
| engine_c p50 | 19.96 ms | 19.96 ms |
| **sequential p50** | 90.20 ms | 90.20 ms |
| **concurrent p50** | 88.31 ms | 87.93 ms |
| median speedup | 1.021× | 1.026× |

- 초기화 약 1.98 s, 출력 boxes[1,200,9]/scores/labels 정상(첫 검출 score 0.252). `PASS C++ TensorRT inference` 양쪽 모두.
- Thor(26.4 ms)와 동일하게 **병렬 제출의 이득이 거의 없음**(A/B 겹침이 총 latency를 줄이지 못함) — Engine B가 지배적(59%).
- Orin MAXN E2E ≈ 88 ms ↔ Thor 26.4 ms = 3.3배. 35 ms 목표는 Orin에서는 불가, Thor 전용 목표로 유지.
