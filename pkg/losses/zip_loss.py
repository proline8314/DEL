import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ZIPLoss(nn.Module):
    def __init__(
        self,
        label_size: int,
        target_size: int,
        matrix_size: int,
        requires_grad: bool = True,
        **kwargs
    ):
        super(ZIPLoss, self).__init__()
        self.label_size = label_size
        self.target_size = target_size
        self.matrix_size = matrix_size

        self.eps = 1e-8 if "eps" not in kwargs else kwargs["eps"]
        self.poisson_loss = nn.PoissonNLLLoss(reduction="none", full=True, **kwargs)

        # parameters
        self.total_count = Parameter(torch.Tensor([1.0]), requires_grad=requires_grad)
        self.matrix_inflat_prob = Parameter(torch.Tensor([0.0]), requires_grad=requires_grad)
        self.target_inflat_prob = Parameter(torch.Tensor([0.0]), requires_grad=requires_grad)
    
    def ZIP_nll(self, lamda, count, inflat_prob):
        zero_nll = -torch.log(
            inflat_prob + (1 - inflat_prob) * torch.exp(-lamda) + self.eps
        )
        non_zero_nll = self.poisson_loss(lamda, count) - torch.log(1 - inflat_prob + self.eps)

        count_is_zero = (count == 0).type(torch.float)
        nll = count_is_zero * zero_nll + (1 - count_is_zero) * non_zero_nll
        # average over batches and labels
        nll = torch.mean(nll)
        return nll

    def forward(self, input, real_count):
        r"""input and real_count must be non-negative"""
        # clip inflat probabilities
        self.matrix_inflat_prob.data.clamp_(self.eps, 1 - self.eps)
        self.target_inflat_prob.data.clamp_(self.eps, 1 - self.eps)

        # split input into matrix and target
        y_tgt, y_mat = torch.split(input, [1, 1], dim=1)
        c_tgt, c_mat = torch.split(real_count, [self.target_size, self.matrix_size], dim=1)
        
        c_pred_tgt = self.total_count * (y_tgt + y_mat)
        c_pred_mat = self.total_count * y_mat

        # calculate nll
        nll_tgt = self.ZIP_nll(c_pred_tgt, c_tgt, self.target_inflat_prob)
        nll_mat = self.ZIP_nll(c_pred_mat, c_mat, self.matrix_inflat_prob)
        nll = nll_tgt + nll_mat
        return nll





