from ..builder import DETECTORS
from .single_stage_kpt import SingleStageDetector_kpt


@DETECTORS.register_module
class SMPR(SingleStageDetector_kpt):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SMPR, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

