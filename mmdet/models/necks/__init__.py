from .bfp import BFP
from .fpn import FPN
from .hrfpn import HRFPN
from .nas_fpn import NASFPN
from .nasfcos_fpn import NASFCOS_FPN
from .pafpn import PAFPN
from .rfp import RFP
from .yolo_neck import YOLOV3Neck
# modified by zy 20210115
from .fpn_carafe import FPN_CARAFE
from .fpn_carafe_2 import FPN_CARAFE_2
from .fpn_carafe_3 import FPN_CARAFE_3
from .fpn_carafe_cuda import FPN_CARAFE_cuda
# modified by zy 20210313
from .fpn_carafe_se import FPN_CARAFE_se
from .fpn_carafe_se_2 import FPN_CARAFE_se2

from .fpn_carafe_sa_3 import FPN_CARAFE_sa_3
from .fpn_carafe_se_3 import FPN_CARAFE_se_3
from .fpn_carafe_3_sa import FPN_CARAFE_3_sa
from .fpn_carafe_3_se import FPN_CARAFE_3_se
from .fpn_carafe_norm import FPN_CARAFE_norm

from .fpn_carafe_3_3_norm import FPN_CARAFE_3_3_norm
from .fpn_carafe_3_res3_norm import FPN_CARAFE_3_res3_norm

# from .icarafe_3_3_sa_se_norm import CARAFE_3_3_sa_se_norm
# from .icarafe_3_3_se_sa_norm import CARAFE_3_3_se_sa_norm
# from .icarafe_3_se_sa_3_norm import CARAFE_3_se_sa_3_norm
# from .icarafe_3_sa_se_3_norm import CARAFE_3_sa_se_3_norm
from .fpn_carafe_3_3_sa_se_norm import FPN_CARAFE_3_3_sa_se_norm
from .fpn_carafe_3_se_sa_3_norm import FPN_CARAFE_3_se_sa_3_norm
from .fpn_carafe_3_3_se_sa_norm import FPN_CARAFE_3_3_se_sa_norm
from .fpn_carafe_3_sa_se_3_norm import FPN_CARAFE_3_sa_se_3_norm

from .fpn_carafe_3_pow import FPN_CARAFE_3_pow
from .fpn_carafe_3_pow_norm import FPN_CARAFE_3_pow_norm
from .fpn_carafe_3_3_pow import FPN_CARAFE_3_3_pow
from .fpn_carafe_3_3_pow_norm import FPN_CARAFE_3_3_pow_norm

from .fpn_carafe_3_kernelexp import FPN_CARAFE_3_kernelexp
from .fpn_carafe_3_3_kernelexp import FPN_CARAFE_3_3_kernelexp

from .fpn_carafe_3_exp import FPN_CARAFE_3_exp

__all__ = [
    'FPN_CARAFE_3_exp',
    'FPN_CARAFE_3_kernelexp',
    'FPN_CARAFE_3_3_kernelexp',

    'FPN', 'BFP', 'HRFPN', 'NASFPN', 'PAFPN', 'NASFCOS_FPN',
    'RFP', 'YOLOV3Neck',
    # modify by zy 20210111
    'FPN_CARAFE',       # python 实现的CARAFE
    'FPN_CARAFE_2',     # python 实现的CARAFE + 3*3
    'FPN_CARAFE_cuda',  # cuda 实现的CARAFE

    'FPN_CARAFE_se',   # python 实现的CARAFE, content encoder增加了senet的结构，增加全局信息
    'FPN_CARAFE_se2',  # python 实现的CARAFE + 3*3, content encoder增加了senet的结构，增加全局信息

    'FPN_CARAFE_sa_3',
    'FPN_CARAFE_se_3',
    'FPN_CARAFE_3_sa',
    'FPN_CARAFE_3_se',
    'FPN_CARAFE_norm',
    'FPN_CARAFE_3_3_norm',
    'FPN_CARAFE_3_res3_norm',

    'FPN_CARAFE_3_3_sa_se_norm',
    'FPN_CARAFE_3_3_se_sa_norm',
    'FPN_CARAFE_3_sa_se_3_norm',
    'FPN_CARAFE_3_se_sa_3_norm',

    'FPN_CARAFE_3_pow',
    'FPN_CARAFE_3_pow_norm',
    'FPN_CARAFE_3_3_pow',
    'FPN_CARAFE_3_3_pow_norm',
]
