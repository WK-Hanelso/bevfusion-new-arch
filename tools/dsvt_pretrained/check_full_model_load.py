"""V3: 실제 전체 모델(config)에 변환된 공식 DSVT 체크포인트를 부분 로드하고 lidar 분기 매칭을 보고한다.

사용(컨테이너): PYTHONPATH=. python tools/dsvt_pretrained/check_full_model_load.py \
    configs/nuscenes/det/ablation/dsvt1_wf1_gf1_dal1.yaml pretrained/dsvt_nuscenes_official_lidar.pth
"""
import argparse, json, sys
import torch
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--out", default="tools/dsvt_pretrained/full_model_load_report.json")
    a = ap.parse_args()
    configs.load(a.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=a.config)
    model = build_model(cfg.model)
    sd = torch.load(a.checkpoint, map_location="cpu")["state_dict"]
    result = model.load_state_dict(sd, strict=False)
    prefix = "encoders.lidar.backbone."
    lidar_keys = [k for k in model.state_dict() if k.startswith(prefix)]
    lidar_missing = [k for k in result.missing_keys if k.startswith(prefix)]
    report = {
        "config": a.config, "checkpoint": a.checkpoint, "model_type": cfg.model["type"],
        "checkpoint_tensors": len(sd), "model_tensors": len(model.state_dict()),
        "lidar_branch_tensors": len(lidar_keys),
        "lidar_branch_missing": lidar_missing,
        "unexpected": list(result.unexpected_keys),
        "loaded_into_lidar_branch": len(lidar_keys) - len(lidar_missing),
    }
    json.dump(report, open(a.out, "w"), indent=2, ensure_ascii=False)
    ok = not result.unexpected_keys and all(k.startswith(prefix + "neck.adapter") for k in lidar_missing)
    print(json.dumps({k: v for k, v in report.items() if k not in ("lidar_branch_missing", "unexpected")}, ensure_ascii=False))
    print("lidar_branch_missing:", lidar_missing); print("unexpected:", result.unexpected_keys[:5], "..." if len(result.unexpected_keys) > 5 else "")
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
