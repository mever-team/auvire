import torch


def iou_with_anchors_torch(anchors_min, anchors_max, box_min, box_max, device):
    """Compute jaccard score between a box and the anchors."""
    len_anchors = anchors_max - anchors_min
    int_xmin = torch.maximum(anchors_min, box_min)
    int_xmax = torch.minimum(anchors_max, box_max)
    inter_len = torch.maximum(int_xmax - int_xmin, torch.tensor(0.0).to(device))
    union_len = len_anchors - inter_len + box_max - box_min
    iou = inter_len / union_len
    return iou


def soft_nms_torch_parallel(proposal, sigma, t1, t2, fps, device="cpu"):
    proposal = proposal.to(device)
    t_score = proposal[:, :, 0]
    t_start = proposal[:, :, 1] / fps
    t_end = proposal[:, :, 2] / fps

    r_start = torch.empty((0)).to(device)
    r_end = torch.empty((0,)).to(device)
    r_score = torch.empty((0)).to(device)
    nproposal = 0

    while torch.sum(t_score[0, :] > -torch.inf) > 1 and nproposal < 101:
        _, max_index = torch.max(t_score, -1)
        max_index_2d = (torch.arange(t_score.shape[0]).to(device), max_index)
        width = t_end[max_index_2d] - t_start[max_index_2d]
        ious = iou_with_anchors_torch(
            t_start,
            t_end,
            t_start[max_index_2d].unsqueeze(1),
            t_end[max_index_2d].unsqueeze(1),
            device,
        )
        idx = ious > t1 + (t2 - t1) * width.unsqueeze(1)
        idx[max_index_2d] = False
        t_score[idx] *= torch.exp(-torch.square(ious[idx]) / sigma)

        r_start = torch.cat((r_start, t_start[max_index_2d].unsqueeze(1)), dim=1)
        r_end = torch.cat((r_end, t_end[max_index_2d].unsqueeze(1)), dim=1)
        r_score = torch.cat((r_score, t_score[max_index_2d].unsqueeze(1)), dim=1)

        t_start[max_index_2d] = torch.nan
        t_end[max_index_2d] = torch.nan
        t_score[max_index_2d] = -torch.inf

        nproposal = r_score.shape[1]

    return torch.stack((r_score, r_start * fps, r_end * fps), dim=-1)
