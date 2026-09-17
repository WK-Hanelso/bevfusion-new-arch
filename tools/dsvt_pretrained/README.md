# 공식 DSVT nuScenes LiDAR 초기화

이 도구는 [Haiyang-W/DSVT](https://github.com/Haiyang-W/DSVT)의 공식
nuScenes checkpoint에서 VFE, DSVT backbone, residual BEV backbone 가중치만
DynamicBEVFusion LiDAR encoder 형식으로 변환한다. 공식 소스와 checkpoint는
Apache-2.0이며 저장소에 포함하지 않는다.

## 1. 다운로드

공식 DSVT 저장소의 nuScenes model zoo에서 `DSVT_Nuscenes_val.pth`를 받아
다음 위치에 둔다.

```bash
mkdir -p pretrained
# 공식 model zoo에서 받은 파일을 아래 경로로 이동
ls -l pretrained/DSVT_Nuscenes_val.pth
sha256sum pretrained/DSVT_Nuscenes_val.pth
```

검증에 사용한 파일의 SHA-256은
`a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb`다.
다른 hash라면 같은 공식 checkpoint인지 먼저 확인한다.

## 2. 변환

저장소 root에서 실행한다.

```bash
python tools/dsvt_pretrained/convert_official_dsvt.py
```

다음 두 파일이 생성된다.

- `pretrained/dsvt_nuscenes_official_lidar.pth`: MMCV `load_checkpoint` 호환 파일
- `pretrained/dsvt_nuscenes_official_lidar.report.json`: 모든 key 매핑과 누락·미사용 수치

검증한 공식 checkpoint에서는 336 tensors, 6,232,032 tensor elements가
이식된다. `neck.adapter.*`는 공식 모델에 없는 384→256 adapter이므로 random
init으로 남는다. `dense_head.*`와 `global_step`은 Phase 1에서 사용하지 않는다.

입출력 경로를 바꾸려면 다음처럼 지정할 수 있다.

```bash
python tools/dsvt_pretrained/convert_official_dsvt.py \
  /path/to/DSVT_Nuscenes_val.pth \
  --output /path/to/dsvt_nuscenes_official_lidar.pth \
  --report /path/to/dsvt_nuscenes_official_lidar.report.json
```

`--include-head`는 별도 승인 대상인 Phase 2 예약 옵션이며 현재 변환하지 않는다.

## 3. 테스트

CPU와 PyTorch 1.12에서 실행할 수 있다.

```bash
pytest -q -s tools/dsvt_pretrained/tests/test_official_layout.py
```

테스트는 legacy 경로의 state dict/forward 회귀, 변환 checkpoint 로딩,
공식 attention·encoder layer·BEV backbone과의 수치 동치, 합성 LiDAR forward를
검증한다. 공식 원문은 테스트 시 SPEC에 명시된 외부 읽기 전용 경로에서만
import하며 저장소로 복사하지 않는다.

## 4. 학습

기존 config가 아니라 pretrained 전용 config를 선택하고 `--load_from`을
명시한다. 기본 LR schedule은 기존 학습과 같다.

```bash
BEVFUSION_CONFIG=configs/nuscenes/det/transfusion/secfpn/lidar/dsvt_dgf_dal_widthformer_0p3_dsvtpre.yaml
python -m torch.distributed.run --nproc_per_node=1 tools/train_torchrun.py \
  "$BEVFUSION_CONFIG" --run-dir runs/dsvt-pretrained \
  --load_from pretrained/dsvt_nuscenes_official_lidar.pth \
  --data.samples_per_gpu 1 --data.workers_per_gpu 0
```

이 config는 `official_layout: true`와 `zero_feature_channels: [4]`를 사용한다.
현재 single-sweep loader의 다섯 번째 `ring_index`를 0으로 만들어 공식
single-frame timestamp=0 입력 의미에 맞춘다. 향후
`LoadPointsFromMultiSweeps`가 다섯 번째 channel에 timestamp를 넣는 구성에서는
`zero_feature_channels: []`로 바꿔야 한다.

