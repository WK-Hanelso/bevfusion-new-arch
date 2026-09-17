# SPEC — 공식 DSVT nuScenes 체크포인트를 DynamicBEVFusion LiDAR 분기 초기화로 이식

브랜치: `dsvt-pretrained-init` (main에서 분기, 2026-09-17). 원본 config·기존 학습 경로는 불변. 이 문서가 구현의 단일 기준이다.

## 0. 목적 · 범위

- **목적**: 공식 DSVT(Pillar) nuScenes 검출 모델(`DSVT_Nuscenes_val.pth`, val mAP 66.4 / NDS 71.1, Apache-2.0)의 가중치를 `DSVTLidarEncoder`(VFE + DSVT backbone + BEV backbone)에 로드할 수 있게 만들고, 그 상태에서 nuScenes full 학습을 시작할 수 있는 config를 제공한다.
- **범위(Phase 1)**: LiDAR 분기(`vfe`, `backbone_3d`, `backbone_2d`) 전부 = 공식 파라미터 7.13M 중 약 6.23M. 헤드는 Phase 2(§8).
- **비범위**: 학습 실행(서버 작업, 별도), Thor ONNX/TensorRT 재수출, 10-sweep 전환.

## 1. 검증 결과 (2026-09-17, 오케스트레이터 확인 완료)

| 항목 | 공식 (Haiyang-W/DSVT) | 우리 (`mmdet3d/models/backbones/dsvt_core.py`) | 판정 |
|---|---|---|---|
| 라이선스 | Apache-2.0 | — | 이식 가능. NOTICE에 출처 기재 |
| 체크포인트 | `DSVT_Nuscenes_val.pth`, `model_state` 449 tensors, 7,130,122 params, global_step 77,240 | — | 로컬 `pretrained/DSVT_Nuscenes_val.pth` (gitignore) |
| 하이퍼파라미터 | voxel 0.3, range ±54, sparse [360,360,1], d_model 128, nhead 8, ff 256, set 90, block 4, window [30,30,1], shifts [0,0,0]/[15,15,0], gelu, dropout 0, normalize_pos False | 동일 | ✓ |
| VFE | DynPillarVFE [128,128]: `pfn_layers.0.linear (64,11)`+BN, `pfn_layers.1.linear (128,128)`+BN. 입력 = [x,y,z,i,t] + cluster(3) + center(3) = 11 | `vfe.layers.0/1` 동일 shape·동일 concat 순서 | ✓ 1:1 |
| 5번째 point feature | `timestamp` (10-sweep, 현재 프레임 = 0) | `ring_index` (sweeps 0, 값 0~31) | ✗ 의미 불일치 → §2 D3 |
| Position embedding | `posembed_layers.0.{block}.{shift}.position_embedding_head.{0:Linear(2,128),1:BN,3:Linear(128,128)}`, 입력 (x−win/2, y−win/2) | `input_layer.position_layers.{block}.{shift}.layers.{0,1,3}`, 입력 (x−15, y−15) | ✓ 1:1 |
| Set attention | `stage_0.{b}.encoder_list.{a}.win_attn.{self_attn, linear1, linear2, norm1, norm2}`; norm1(x+attn) → norm2(x+ffn) | `blocks.{b}.layers.{a}.{attention, linear1, linear2, norm1, norm2}`; 동일 순서 | ✓ 1:1 |
| Encoder-layer 잔차 norm | `encoder_list.{a}.norm`: out = LayerNorm(win_attn(x) + x) | **없음** | ✗ 구조 누락 (8 LayerNorm) → §2 D2-a |
| Block 잔차 norm | `residual_norm_stage_0.{b}`: LayerNorm(block(x)+x) | `residual_norms.{b}` 동일 | ✓ 1:1 |
| Scatter | PointPillarScatter3d → [B,128,ny,nx] | DensePillarScatter → [B,128,Y,X] | ✓ 파라미터 없음 |
| BEV backbone | **BaseBEVResBackbone**: blocks[i] = BasicBlock(stride, downsample=True) + layer_nums[i]×BasicBlock; BasicBlock = conv1(3×3,pad1)-bn1-relu-conv2(3×3)-bn2 (+1×1 downsample) 잔차; layer_nums [1,2,2], strides [1,2,2], filters [128,128,256]; deblocks [Conv2d(128,128,2,s2), ConvT(128,128,1), ConvT(256,128,2,s2)] 각 +BN+ReLU; 출력 concat 384ch | `DSVTBEVNeck`: **비잔차** ZeroPad+conv-BN-ReLU 스택 (BaseBEVBackbone 형태). deblocks 동일. adapter 384→256 추가 | ✗ 구조 불일치 (4.80M + 0.21M params, 전체의 70%) → §2 D2-b |
| 헤드 | TransFusionHead(one-stage, +iou): `shared_conv (128,384,3,3)`, heatmap_head, class_encoding, decoder(단일), prediction_head{center,height,dim,rot,vel,iou,heatmap} | DALDecoupledHead: shared_conv(256→128, fused), lidar_shared_conv(256→128), heatmap_head, class_encoding, decoder[0], prediction_heads[0]{center,height,dim,rot,vel,heatmap} | 부분 가능 (Phase 2) |

**결론: 이식 가능. 단, LiDAR 분기를 공식 구조와 정확히 맞추려면 두 곳(encoder-layer norm, 잔차 BEV backbone)을 추가해야 하며 이는 기존 ep19 체크포인트와 호환되지 않는다. 따라서 옵션으로 추가하고 기존 경로는 그대로 둔다.**

## 2. 설계 결정

- **D1 config 분리**: 신규 `configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3_dsvtpre.yaml`. 기존 `..._0p3.yaml`은 한 글자도 바꾸지 않는다. 신규 config는 `_base_`/복사 중 저장소 관례를 따른다(기존 yaml 상속 방식 확인 후 결정, 없으면 복사).
- **D2 `DSVTLidarEncoder(official_layout: bool = False)`**:
  - (a) True면 `DSVTBlock`의 각 `SetAttention` 뒤에 identity 잔차 + `LayerNorm`을 적용한다. 모듈 속성명은 `blocks.{b}.layer_norms.{a}` (ModuleList of LayerNorm). forward: `x = layer_norms[a](layers[a](x, …) + x)`.
  - (b) True면 neck을 `DSVTBEVResNeck`(신규 클래스, §1의 BaseBEVResBackbone과 동일 구조, `blocks.{i}.{j}.{conv1,bn1,conv2,bn2,downsample_layer}` + `deblocks` + `adapter`)로 생성한다. 속성명은 공식과 동일하게 맞춰 매핑을 단순화한다. BN eps 1e-3, momentum 0.01.
  - False면 코드 경로·state_dict 키·수치가 현재와 100% 동일해야 한다(회귀 테스트 T1).
- **D3 5번째 feature 처리**: `DynamicPillarVFE(zero_feature_channels: list[int] = [])`. 신규 config는 `[4]`로 설정해 ring_index를 0으로 덮는다(공식의 단일 프레임 timestamp=0과 동일 의미). 다중 sweep 전환 시 `LoadPointsFromMultiSweeps`가 5번째를 timestamp로 채우므로 그때는 `[]`.
- **D4 변환 스크립트** `tools/dsvt_pretrained/convert_official_dsvt.py`:
  - 입력: 공식 pth. 출력: `pretrained/dsvt_nuscenes_official_lidar.pth` (mmcv `load_checkpoint` 호환: `{"state_dict": {...}, "meta": {"source": ..., "sha256": ..., "mapping_version": 1}}`) + `pretrained/dsvt_nuscenes_official_lidar.report.json`.
  - 키 접두사: `encoders.lidar.backbone.` (BEVFusion `self.encoders["lidar"]["backbone"]` 경로. 구현 시 실제 모델 `state_dict().keys()`로 확인).
  - 매핑 표(§3)대로 이름 변환. `num_batches_tracked` 포함. 헤드 키는 `--include-head` 옵션에서만(Phase 2, 기본 off).
  - report: matched/missing(우리 모듈 중 미초기화)/unexpected(공식 중 미사용) 키 목록과 param 수, 총 이식 비율.
- **D5 로딩 방식**: 학습은 `--load_from pretrained/dsvt_nuscenes_official_lidar.pth`(runner.load_checkpoint, strict=False). 신규 config에 `load_from` 기본값을 넣지 않는다(명시 실행). README에 명령 기재.
- **D6 LR**: 기본은 기존과 동일 스케줄. 선택 옵션으로 `paramwise_cfg` lidar 분기 `lr_mult 0.1` 예시를 config 주석으로만 둔다(실험 변수는 서버 단계에서 결정).

## 3. 키 매핑 표 (공식 → 우리, 접두사 `encoders.lidar.backbone.` 생략)

| 공식 | 우리 |
|---|---|
| `vfe.pfn_layers.{i}.linear.weight` | `vfe.layers.{i}.linear.weight` |
| `vfe.pfn_layers.{i}.norm.{weight,bias,running_mean,running_var,num_batches_tracked}` | `vfe.layers.{i}.norm.*` |
| `backbone_3d.input_layer.posembed_layers.0.{b}.{s}.position_embedding_head.{0,1,3}.*` | `backbone.input_layer.position_layers.{b}.{s}.layers.{0,1,3}.*` |
| `backbone_3d.stage_0.{b}.encoder_list.{a}.win_attn.self_attn.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}` | `backbone.blocks.{b}.layers.{a}.attention.*` |
| `backbone_3d.stage_0.{b}.encoder_list.{a}.win_attn.{linear1,linear2,norm1,norm2}.*` | `backbone.blocks.{b}.layers.{a}.{linear1,linear2,norm1,norm2}.*` |
| `backbone_3d.stage_0.{b}.encoder_list.{a}.norm.*` | `backbone.blocks.{b}.layer_norms.{a}.*` (official_layout=True에만 존재) |
| `backbone_3d.residual_norm_stage_0.{b}.*` | `backbone.residual_norms.{b}.*` |
| `backbone_2d.blocks.{i}.{j}.{conv1,bn1,conv2,bn2,downsample_layer.0,downsample_layer.1}.*` | `neck.blocks.{i}.{j}.*` (DSVTBEVResNeck, 동일 이름) |
| `backbone_2d.deblocks.{i}.{0,1}.*` | `neck.deblocks.{i}.{0,1}.*` |
| (없음) | `neck.adapter.*` — random init 유지 (missing 허용) |
| `dense_head.*`, `global_step` | Phase 2 / 무시 |

## 4. 파일

신규: `mmdet3d/models/backbones/dsvt_core.py`(수정: D2, D3), `tools/dsvt_pretrained/convert_official_dsvt.py`, `tools/dsvt_pretrained/README.md`, `tools/dsvt_pretrained/tests/test_official_layout.py`, `configs/.../dsvt_dgf_dal_widthformer_0p3_dsvtpre.yaml`, `NOTICE`(없으면 생성; DSVT Apache-2.0 출처), `.gitignore`에 `pretrained/*.pth`, `pretrained/*.json` 추가, `new-arch.md`에 절 추가(“공식 DSVT 초기화”), `tools/dsvt_pretrained/EXEC.md`(실행 로그).
참고 원문(커밋 금지): `/tmp/claude-1000/-home-hanelso-hanelso/8edcff05-d561-4bd4-9120-e7bcf8ad837d/scratchpad/dsvt_ckpt/official/` (dsvt.py, dsvt_input_layer.py, dynamic_pillar_vfe.py, base_bev_res_backbone.py, pointpillar_scatter.py, 두 yaml, LICENSE).

## 5. 검증 (DoD) — 전부 CPU에서 수행 가능해야 함

- **T1 회귀**: `official_layout=False`로 만든 `DSVTLidarEncoder`의 `state_dict().keys()`와 각 shape가 현재 main과 동일(테스트 안에 현재 키 목록을 고정값으로 박아 비교). 랜덤 입력 forward 출력이 변경 전 코드와 동일(허용 오차 0) — 변경 전 모듈은 `git show main:mmdet3d/models/backbones/dsvt_core.py`를 임시 모듈로 import해 비교.
- **T2 로딩**: 변환 pth를 `official_layout=True` 인코더에 `load_state_dict(strict=False)` → missing = `neck.adapter.*`만, unexpected = 0. report.json의 param 이식 수 = 6,232,xxx (VFE 17,858 + input_layer 139,272 + stage 1,061,888 + encoder norms 2,048 + residual norms 1,024 + backbone_2d 5,011,990 근사; 실제 값을 report에 기록).
- **T3 수치 동치**(순수 torch 부분): 공식 원문에서 `SetAttention`·`DSVT_EncoderLayer`(dsvt.py) 및 `BasicBlock`·`BaseBEVResBackbone`(base_bev_res_backbone.py)를 최소 stub으로 import하고 같은 가중치·같은 입력(동일 set index/mask/pos)으로 우리 모듈과 출력 max-abs-diff < 1e-5 확인. VFE는 torch_scatter 부재 시 생략 가능(EXEC.md에 사유 기록).
- **T4 smoke**: `tools/smoke_train.py` 또는 동등 경로로 신규 config 1-step forward/backward를 CPU(또는 로컬 2060 6GB)에서 시도. 데이터가 없으면 합성 입력으로 모델 build + forward만 확인하고 EXEC.md에 기록.
- **T5 문서·git**: README(사용 절차: 다운로드 → 변환 → 학습 명령), NOTICE, new-arch.md, .gitignore. 커밋 identity `WK-Hanelso <wk.hanelso@gmail.com>`, 메시지 `dsvt-pretrained: 공식 DSVT nuScenes 체크포인트 LiDAR 분기 이식 (official_layout, converter, tests)`. **push 금지**(오케스트레이터가 검토 후 push).

## 6. 서버 실험 (다음 단계, 본 SPEC 밖)

- E1 mini: scratch vs pretrained-init, 동일 5 epoch, mAP/loss 곡선 비교.
- E2 full: pretrained-init 20 epoch vs 기존 ep19 (mAP 0.5281 / NDS 0.5132).
- 변수 후보: lidar 분기 lr_mult, `zero_feature_channels` vs 10-sweep.

## 7. 금지

- 기존 `..._0p3.yaml`, `transfusion.py`, Thor deployment 코드 수정 금지.
- 공식 소스 파일을 저장소에 복사 금지(참조만). 체크포인트 커밋 금지.
- 확인되지 않은 수치를 README에 쓰지 않는다.

## 8. Phase 2 (별도 승인 후)

헤드 부분 이식: `dense_head.{heatmap_head, class_encoding, decoder, prediction_head.{center,height,dim,rot,vel,heatmap}}` → DALDecoupledHead 대응 키(shape 일치 시). `shared_conv`(384 vs 256)와 `iou`는 제외.
