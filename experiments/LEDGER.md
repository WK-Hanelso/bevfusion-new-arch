# Ablation experiment ledger

수치는 실행 후 기록하고, 클래스별 AP 원본 표/파일은 링크로 남긴다. DSVT
공식 초기화를 쓰지 않은 재실험은 초기화 칸에 `scratch`를 명시한다.

| 실험ID | config | 우선 실행 | git commit | 초기화 (DSVT 공식/ImageNet/scratch) | 데이터 (full / 1/4 subset) | sweeps | epochs | best epoch | mAP | NDS | 클래스별 AP 링크 | Thor A P50 | Thor B P50 | Thor C P50 | Thor E2E P50 | Orin A P50 | Orin B P50 | Orin C P50 | Orin E2E P50 | 비고 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A-0000 | `configs/nuscenes/det/ablation/dsvt0_wf0_gf0_dal0.yaml` | 1차 (B0) | | ImageNet + scratch | | | | | | | | | | | | | | | | |
| A-0001 | `configs/nuscenes/det/ablation/dsvt0_wf0_gf0_dal1.yaml` | 1차 (E4) | | ImageNet + scratch | | | | | | | | | | | | | | | | |
| A-0010 | `configs/nuscenes/det/ablation/dsvt0_wf0_gf1_dal0.yaml` | 1차 (E3) | | ImageNet + scratch | | | | | | | | | | | | | | | | |
| A-0011 | `configs/nuscenes/det/ablation/dsvt0_wf0_gf1_dal1.yaml` | | | ImageNet + scratch | | | | | | | | | | | | | | | | |
| A-0100 | `configs/nuscenes/det/ablation/dsvt0_wf1_gf0_dal0.yaml` | 1차 (E2) | | ImageNet + scratch | | | | | | | | | | | | | | | | |
| A-0101 | `configs/nuscenes/det/ablation/dsvt0_wf1_gf0_dal1.yaml` | | | ImageNet + scratch | | | | | | | | | | | | | | | | |
| A-0110 | `configs/nuscenes/det/ablation/dsvt0_wf1_gf1_dal0.yaml` | | | ImageNet + scratch | | | | | | | | | | | | | | | | |
| A-0111 | `configs/nuscenes/det/ablation/dsvt0_wf1_gf1_dal1.yaml` | | | ImageNet + scratch | | | | | | | | | | | | | | | | |
| A-1000 | `configs/nuscenes/det/ablation/dsvt1_wf0_gf0_dal0.yaml` | 1차 (E1) | | DSVT 공식 + ImageNet | | | | | | | | | | | | | | | | |
| A-1001 | `configs/nuscenes/det/ablation/dsvt1_wf0_gf0_dal1.yaml` | | | DSVT 공식 + ImageNet | | | | | | | | | | | | | | | | |
| A-1010 | `configs/nuscenes/det/ablation/dsvt1_wf0_gf1_dal0.yaml` | | | DSVT 공식 + ImageNet | | | | | | | | | | | | | | | | |
| A-1011 | `configs/nuscenes/det/ablation/dsvt1_wf0_gf1_dal1.yaml` | | | DSVT 공식 + ImageNet | | | | | | | | | | | | | | | | |
| A-1100 | `configs/nuscenes/det/ablation/dsvt1_wf1_gf0_dal0.yaml` | 1차 (C2) | | DSVT 공식 + ImageNet | | | | | | | | | | | | | | | | |
| A-1101 | `configs/nuscenes/det/ablation/dsvt1_wf1_gf0_dal1.yaml` | | | DSVT 공식 + ImageNet | | | | | | | | | | | | | | | | |
| A-1110 | `configs/nuscenes/det/ablation/dsvt1_wf1_gf1_dal0.yaml` | 1차 (C3) | | DSVT 공식 + ImageNet | | | | | | | | | | | | | | | | |
| A-1111 | `configs/nuscenes/det/ablation/dsvt1_wf1_gf1_dal1.yaml` | 1차 (C4) | | DSVT 공식 + ImageNet | | | | | | | | | | | | | | | | |
