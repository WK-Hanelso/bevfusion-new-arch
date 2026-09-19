# SPEC — spconv v2 교체 (레거시 mmdet3d/ops/spconv → pip spconv-cu12x)

## 동기 (2026-09-19)
레거시 spconv(v1 포크)가 GPU당 배치 32에서 `illegal memory access`. 유지보수되는 spconv v2(traveller59)는 제한 없고 빠름. 사용자 지시.
NVIDIA CUDA-BEVFusion의 libspconv는 추론 전용(TensorRT)이므로 학습 교체 대상은 spconv v2 pip 패키지.

## 요구
1. `mmdet3d/ops/sparse_block.py`·`SparseEncoder`(mmdet3d/models/backbones/sparse_encoder.py 등 사용처)가 **환경 변수 또는 config 플래그**(`BEVFUSION_SPCONV=v2`, 기본 legacy)로 spconv v2(`spconv.pytorch`)를 쓸 수 있게 한다. 레거시 경로는 그대로 유지(기본값 변경 없음).
2. API 매핑: SparseConvTensor(features, indices, spatial_shape, batch_size), SubMConv3d/SparseConv3d/SparseInverseConv3d(indice_key), SparseSequential, `.dense()`. BN/ReLU는 spconv.pytorch에서도 동일하게 SparseSequential에 넣을 수 있음.
3. 가중치 호환: 레거시 state_dict 키/shape가 v2 모듈에 그대로 로드되어야 함(conv weight 레이아웃 차이가 있으면 변환 함수 제공: legacy weight shape (k,k,k,in,out) vs spconv v2 (out,k,k,k,in) 확인).
4. 동치 테스트 `tests/test_spconv2_equiv.py`(GPU 있을 때만): 같은 가중치·같은 voxel 입력으로 legacy vs v2 SparseEncoder 출력 max abs diff ≤ 1e-3(fp32). CPU 호스트에서는 skip.
5. 서버 설치 절차 문서: `pip install spconv-cu126`(torch 2.7.1+cu128 호환 빌드 확인; cu128 휠이 없으면 cu126/cu124 호환 여부 명시), `docker/requirements-cu128.txt`에 주석 추가(선택 설치).
6. 산출물: 코드, 테스트, `mmdet3d/ops/EXEC_spconv2.md`. 커밋 금지.
