import torchvision
import torch.nn as nn
import torch


def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return torchvision.ops.sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_obj = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        return self.loss_obj(inputs, targets)


class CombinedLoss(nn.Module):

    def __init__(self, alpha, gamma, composition, factor):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l_focal = 1.0 if "focal" in composition else 0.0
        self.l_diou = 1.0 if "diou" in composition else 0.0
        self.l_sl1 = 1.0 if "sl1" in composition else 0.0
        self.l_rec = 1.0 if "rec" in composition else 0.0
        self.l_det = 1.0 if "det" in composition else 0.0
        self.factor = factor
        self.bce = BCELoss()

    def smooth_l1(self, inputs, targets):
        loss = torch.nn.functional.smooth_l1_loss(inputs[:, :, 1:] / 25, targets[:, :, 1:] / 25, reduction="none")
        loss = loss.mean(dim=-1)
        loss = targets[:, :, 0] * loss
        return loss

    def focal(self, inputs, targets):
        loss = torchvision.ops.sigmoid_focal_loss(
            inputs[:, :, 0],
            targets[:, :, 0],
            self.alpha,
            self.gamma,
            "none",
        )
        return loss

    def diou(self, inputs, targets):
        loss = ctr_diou_loss_1d(
            input_offsets=inputs[:, :, 1:].reshape((-1, 2)),
            target_offsets=targets[:, :, 1:].view(-1, 2),
            reduction="none",
        ).view(inputs.shape[0], inputs.shape[1])
        loss = targets[:, :, 0] * loss
        return loss

    def num_positives(self, targets):
        den = targets[:, :, 0].sum(dim=-1)
        den[den == 0] = 1
        return den

    def localization_loss(self, inputs, targets):
        return torch.stack(
            [
                (
                    self.l_focal * self.focal(input_, targets[:, :: self.factor[i], :])
                    + self.l_diou * self.diou(input_, targets[:, :: self.factor[i], :])
                    + self.l_sl1 * self.smooth_l1(input_, targets[:, :: self.factor[i], :])
                ).sum(dim=-1)
                / self.num_positives(targets[:, :: self.factor[i], :])
                for i, input_ in enumerate(inputs)
            ],
            dim=-1,
        ).mean(dim=-1)

    def detection_loss(self, inputs, targets):
        alpha = 4
        video_level_inputs = [
            torch.sum(input_[:, :, 0] * torch.softmax(alpha * input_[:, :, 0], dim=-1), dim=-1) for input_ in inputs
        ]
        video_level_targets = [torch.any(targets[:, :, 0], dim=-1).float() for _ in inputs]
        return torch.stack(
            [self.bce(input_, target_) for input_, target_ in zip(video_level_inputs, video_level_targets)], dim=-1
        ).mean(dim=-1)

    def reconstruction_loss(self, dissimilarity, targets):
        fakes = torch.any(targets[:, :, 0], dim=-1).view(-1, 1)
        masked_dissimilarity = dissimilarity.masked_fill(fakes, 0.0)
        return masked_dissimilarity.mean(dim=-1)

    def forward(self, inputs, targets, errors):
        loss_loc = self.localization_loss(inputs, targets)
        loss_det = self.detection_loss(inputs, targets)
        loss_rec = self.reconstruction_loss(errors, targets)
        loss = (loss_loc + self.l_det * loss_det + self.l_rec * loss_rec) / (1 + self.l_det + self.l_rec)
        return loss.mean()
