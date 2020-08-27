import torch

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, kpts, gt_kpts, **kwargs):
        # positive的oks对应的index
        # torch.unique()表示gt的index只出现一次
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        # negative的oks对应的index
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = kpts.new_zeros(kpts.shape[0], dtype=torch.uint8)
        gt_kpts = gt_kpts.reshape(-1, 51)
        kpt_index = []
        for i in range(17):    
            kpt_index.append(3 * i)
            kpt_index.append(3 * i + 1)
        gt_kpts = gt_kpts[:, kpt_index]
        
        # max_overlaps.shape = [13343]
        max_overlaps = assign_result.max_overlaps
        # kpts.shape = [13343, 34]
        sampling_result = SamplingResult(pos_inds, neg_inds, kpts, gt_kpts,
                                         assign_result, gt_flags, max_overlaps)
        return sampling_result

    # def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
    #     """Directly returns the positive and negative indices  of samples.

    #     Args:
    #         assign_result (:obj:`AssignResult`): Assigned results
    #         bboxes (torch.Tensor): Bounding boxes
    #         gt_bboxes (torch.Tensor): Ground truth boxes

    #     Returns:
    #         :obj:`SamplingResult`: sampler results
    #     """
    #     pos_inds = torch.nonzero(
    #         assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    #     neg_inds = torch.nonzero(
    #         assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
    #     gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
    #     sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
    #                                      assign_result, gt_flags)
    #     return sampling_result
