"""진단용 전체 모델 체크포인트: config로 모델을 만들고 변환된 공식 DSVT ckpt를 부분 로드한 뒤 저장.
(정확도 검증용이 아니라 export/엔진 파이프라인 검증용. 카메라·fusion·head는 초기화 상태.)"""
import argparse, torch
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("checkpoint"); ap.add_argument("out")
a = ap.parse_args()
configs.load(a.config, recursive=True); cfg = Config(recursive_eval(configs), filename=a.config)
model = build_model(cfg.model)
r = model.load_state_dict(torch.load(a.checkpoint, map_location="cpu")["state_dict"], strict=False)
assert not r.unexpected_keys, r.unexpected_keys
torch.save({"state_dict": model.state_dict(), "meta": {"diagnostic": True, "config": a.config, "lidar_init": a.checkpoint}}, a.out)
print("saved", a.out, "tensors", len(model.state_dict()))
