# flake8: noqa
import os
import sys
import os.path as osp

# ✅ 로컬 BasicSR를 import 최우선으로
BASICSR_ROOT = "/SSD4/vipnu/TISR/PBVS_TSR/BasicSR"
if BASICSR_ROOT not in sys.path:
    sys.path.insert(0, BASICSR_ROOT)

# (선택) 프로젝트 루트도 필요하면 추가
PROJECT_ROOT = "/SSD4/vipnu/TISR/PBVS_TSR"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ✅ 디버그 확인
import basicsr, inspect
import basicsr.models.sr_model as m
print("[DEBUG] basicsr from:", basicsr.__file__)
print("[DEBUG] SRModel from:", inspect.getfile(m.SRModel))

from drct.archs import *
from drct.data import *
from drct.models import *

from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)