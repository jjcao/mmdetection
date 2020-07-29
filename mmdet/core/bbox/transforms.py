import numpy as np
import torch


def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal" and
            "vertical". Default: "horizontal"


    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical']
    flipped = bboxes.clone()
    if direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4]
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4]
    return flipped


def bbox_mapping(bboxes,
                 img_shape,
                 scale_factor,
                 flip,
                 flip_direction='horizontal'):
    """Map bboxes from the original image scale to testing scale."""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape, flip_direction)
    return new_bboxes


def bbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    """Convert rois to bounding box format.

    Args:
        rois (torch.Tensor): RoIs with the shape (n, 5) where the first
            column indicates batch id of each RoI.

    Returns:
        list[torch.Tensor]: Converted boxes of corresponding rois.
    """
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)

def kpt2result(bboxes, kpts, labels, num_classes):
    assert bboxes.shape[0] == kpts.shape[0]
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)], \
               [np.zeros((0, 35), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        kpts = kpts.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)], \
            [kpts[labels == i, :] for i in range(num_classes - 1)]
            
def offset2kpt(points, offset, max_shape=None):
    assert points.shape[0] == offset.shape[0]
    assert offset.shape[1] == 34

    x1 = points[:, 0] + offset[:, 0]
    y1 = points[:, 1] + offset[:, 1]
    x2 = points[:, 0] + offset[:, 2]
    y2 = points[:, 1] + offset[:, 3]
    x3 = points[:, 0] + offset[:, 4]
    y3 = points[:, 1] + offset[:, 5]
    x4 = points[:, 0] + offset[:, 6]
    y4 = points[:, 1] + offset[:, 7]
    x5 = points[:, 0] + offset[:, 8]
    y5 = points[:, 1] + offset[:, 9]
    x6 = points[:, 0] + offset[:, 10]
    y6 = points[:, 1] + offset[:, 11]
    x7 = points[:, 0] + offset[:, 12]
    y7 = points[:, 1] + offset[:, 13]
    x8 = points[:, 0] + offset[:, 14]
    y8 = points[:, 1] + offset[:, 15]
    x9 = points[:, 0] + offset[:, 16]
    y9 = points[:, 1] + offset[:, 17]
    x10 = points[:, 0] + offset[:, 18]
    y10 = points[:, 1] + offset[:, 19]
    x11 = points[:, 0] + offset[:, 20]
    y11 = points[:, 1] + offset[:, 21]
    x12 = points[:, 0] + offset[:, 22]
    y12 = points[:, 1] + offset[:, 23]
    x13 = points[:, 0] + offset[:, 24]
    y13 = points[:, 1] + offset[:, 25]
    x14 = points[:, 0] + offset[:, 26]
    y14 = points[:, 1] + offset[:, 27]
    x15 = points[:, 0] + offset[:, 28]
    y15 = points[:, 1] + offset[:, 29]
    x16 = points[:, 0] + offset[:, 30]
    y16 = points[:, 1] + offset[:, 31]
    x17 = points[:, 0] + offset[:, 32]
    y17 = points[:, 1] + offset[:, 33]

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
        x3 = x3.clamp(min=0, max=max_shape[1] - 1)
        y3 = y3.clamp(min=0, max=max_shape[0] - 1)
        x4 = x4.clamp(min=0, max=max_shape[1] - 1)
        y4 = y4.clamp(min=0, max=max_shape[0] - 1)
        x5 = x5.clamp(min=0, max=max_shape[1] - 1)
        y5 = y5.clamp(min=0, max=max_shape[0] - 1)
        x6 = x6.clamp(min=0, max=max_shape[1] - 1)
        y6 = y6.clamp(min=0, max=max_shape[0] - 1)
        x7 = x7.clamp(min=0, max=max_shape[1] - 1)
        y7 = y7.clamp(min=0, max=max_shape[0] - 1)
        x8 = x8.clamp(min=0, max=max_shape[1] - 1)
        y8 = y8.clamp(min=0, max=max_shape[0] - 1)
        x9 = x9.clamp(min=0, max=max_shape[1] - 1)
        y9 = y9.clamp(min=0, max=max_shape[0] - 1)
        x10 = x10.clamp(min=0, max=max_shape[1] - 1)
        y10 = y10.clamp(min=0, max=max_shape[0] - 1)
        x11 = x11.clamp(min=0, max=max_shape[1] - 1)
        y11 = y11.clamp(min=0, max=max_shape[0] - 1)
        x12 = x12.clamp(min=0, max=max_shape[1] - 1)
        y12 = y12.clamp(min=0, max=max_shape[0] - 1)
        x13 = x13.clamp(min=0, max=max_shape[1] - 1)
        y13 = y13.clamp(min=0, max=max_shape[0] - 1)
        x14 = x14.clamp(min=0, max=max_shape[1] - 1)
        y14 = y14.clamp(min=0, max=max_shape[0] - 1)
        x15 = x15.clamp(min=0, max=max_shape[1] - 1)
        y15 = y15.clamp(min=0, max=max_shape[0] - 1)
        x16 = x16.clamp(min=0, max=max_shape[1] - 1)
        y16 = y16.clamp(min=0, max=max_shape[0] - 1)
        x17 = x17.clamp(min=0, max=max_shape[1] - 1)
        y17 = y17.clamp(min=0, max=max_shape[0] - 1)
    
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, \
        x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13,\
        x14, y14, x15, y15, x16, y16, x17, y17], -1)

def struct_offset2kpt(points, offset, max_shape=None):
    assert points.shape[0] == offset.shape[0]
    assert offset.shape[1] == 34

    x1 = points[:, 0] + offset[:, 0]
    y1 = points[:, 1] + offset[:, 1]
    x2 = x1 + offset[:, 2]
    y2 = y1 + offset[:, 3]
    x3 = x1 + offset[:, 4]
    y3 = y1 + offset[:, 5]
    x4 = x2 + offset[:, 6]
    y4 = y2 + offset[:, 7]
    x5 = x3 + offset[:, 8]
    y5 = y3 + offset[:, 9]
    x6 = points[:, 0] + offset[:, 10]
    y6 = points[:, 1] + offset[:, 11]
    x7 = points[:, 0] + offset[:, 12]
    y7 = points[:, 1] + offset[:, 13]
    x8 = x6 + offset[:, 14]
    y8 = y6 + offset[:, 15]
    x9 = x7 + offset[:, 16]
    y9 = y7 + offset[:, 17]
    x10 = x8 + offset[:, 18]
    y10 = y8 + offset[:, 19]
    x11 = x9 + offset[:, 20]
    y11 = y9 + offset[:, 21]
    x12 = points[:, 0] + offset[:, 22]
    y12 = points[:, 1] + offset[:, 23]
    x13 = points[:, 0] + offset[:, 24]
    y13 = points[:, 1] + offset[:, 25]
    x14 = x12 + offset[:, 26]
    y14 = y12 + offset[:, 27]
    x15 = x13 + offset[:, 28]
    y15 = y13 + offset[:, 29]
    x16 = x14 + offset[:, 30]
    y16 = y14 + offset[:, 31]
    x17 = x15 + offset[:, 32]
    y17 = y15 + offset[:, 33]

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
        x3 = x3.clamp(min=0, max=max_shape[1] - 1)
        y3 = y3.clamp(min=0, max=max_shape[0] - 1)
        x4 = x4.clamp(min=0, max=max_shape[1] - 1)
        y4 = y4.clamp(min=0, max=max_shape[0] - 1)
        x5 = x5.clamp(min=0, max=max_shape[1] - 1)
        y5 = y5.clamp(min=0, max=max_shape[0] - 1)
        x6 = x6.clamp(min=0, max=max_shape[1] - 1)
        y6 = y6.clamp(min=0, max=max_shape[0] - 1)
        x7 = x7.clamp(min=0, max=max_shape[1] - 1)
        y7 = y7.clamp(min=0, max=max_shape[0] - 1)
        x8 = x8.clamp(min=0, max=max_shape[1] - 1)
        y8 = y8.clamp(min=0, max=max_shape[0] - 1)
        x9 = x9.clamp(min=0, max=max_shape[1] - 1)
        y9 = y9.clamp(min=0, max=max_shape[0] - 1)
        x10 = x10.clamp(min=0, max=max_shape[1] - 1)
        y10 = y10.clamp(min=0, max=max_shape[0] - 1)
        x11 = x11.clamp(min=0, max=max_shape[1] - 1)
        y11 = y11.clamp(min=0, max=max_shape[0] - 1)
        x12 = x12.clamp(min=0, max=max_shape[1] - 1)
        y12 = y12.clamp(min=0, max=max_shape[0] - 1)
        x13 = x13.clamp(min=0, max=max_shape[1] - 1)
        y13 = y13.clamp(min=0, max=max_shape[0] - 1)
        x14 = x14.clamp(min=0, max=max_shape[1] - 1)
        y14 = y14.clamp(min=0, max=max_shape[0] - 1)
        x15 = x15.clamp(min=0, max=max_shape[1] - 1)
        y15 = y15.clamp(min=0, max=max_shape[0] - 1)
        x16 = x16.clamp(min=0, max=max_shape[1] - 1)
        y16 = y16.clamp(min=0, max=max_shape[0] - 1)
        x17 = x17.clamp(min=0, max=max_shape[1] - 1)
        y17 = y17.clamp(min=0, max=max_shape[0] - 1)
    
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, \
        x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13,\
        x14, y14, x15, y15, x16, y16, x17, y17], -1)
