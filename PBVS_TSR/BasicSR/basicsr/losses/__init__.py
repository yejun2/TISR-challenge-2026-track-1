from copy import deepcopy

import sys
# ✅ 로컬 BasicSR를 import 최우선으로
BASICSR_ROOT = "/SSD4/vipnu/TISR/PBVS_TSR/BasicSR"
if BASICSR_ROOT not in sys.path:
    sys.path.insert(0, BASICSR_ROOT)

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, PerceptualLoss, WeightedTVLoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty, SSIMStructureOnlyLoss, MultiLoss)


__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss', 'GANLoss', 'gradient_penalty_loss',
    'r1_penalty', 'g_path_regularize', 'SSIMStructureOnlyLoss', 'MultiLoss'
]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
