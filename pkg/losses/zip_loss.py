import torch
import torch.nn as nn
import torch.nn.functional as F
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
        nll[nll.isnan()] = 0.
        nll[nll.isinf()] = 0.
        nll = torch.mean(nll)
        return nll

    def forward(self, input, real_count, return_loss=False):
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
        if return_loss:
            return nll, nll_tgt, nll_mat
        else:
            return nll


class CorrectedZIPLoss(nn.Module):
    def __init__(
        self,
        label_size: int,
        target_size: int,
        matrix_size: int,
        requires_grad: bool = True,
        count_loss_weight: float = 0.01,
        **kwargs
    ):
        super(CorrectedZIPLoss, self).__init__()
        self.label_size = label_size
        self.target_size = target_size
        self.matrix_size = matrix_size
        self.count_loss_weight = count_loss_weight

        self.eps = 1e-8 if "eps" not in kwargs else kwargs["eps"]
        self.poisson_loss = nn.PoissonNLLLoss(reduction="none", full=True, **kwargs)

        # parameters
        self.target_count = Parameter(torch.Tensor([1.0]), requires_grad=requires_grad)
        self.matrix_count = Parameter(torch.Tensor([1.0]), requires_grad=requires_grad)


        self.matrix_inflat_prob = Parameter(torch.Tensor([0.0]), requires_grad=requires_grad)
        self.target_inflat_prob = Parameter(torch.Tensor([0.0]), requires_grad=requires_grad)

        self.sigma_correction_factor = Parameter(torch.Tensor([0.0]), requires_grad=requires_grad)
    
    def ZIP_nll(self, lamda, count, inflat_prob):
        zero_nll = -torch.log(
            inflat_prob + (1 - inflat_prob) * torch.exp(-lamda) + self.eps
        )
        non_zero_nll = self.poisson_loss(lamda, count) - torch.log(1 - inflat_prob + self.eps)

        count_is_zero = (count == 0).type(torch.float)
        nll = count_is_zero * zero_nll + (1 - count_is_zero) * non_zero_nll
        # average over batches and labels
        nll[nll.isnan()] = 0.
        nll[nll.isinf()] = 0.
        nll = torch.mean(nll)
        return nll

    def forward(self, input, real_count, return_loss=False):
        r"""input and real_count must be non-negative"""
        # clip inflat probabilities
        self.matrix_inflat_prob.data.clamp_(self.eps, 1 - self.eps)
        self.target_inflat_prob.data.clamp_(self.eps, 1 - self.eps)
        self.sigma_correction_factor.data.clamp_(0, 1)

        # split input into matrix and target
        y_tgt, y_mat = torch.split(input, [1, 1], dim=1)
        c_tgt, c_mat = torch.split(real_count, [self.target_size, self.matrix_size], dim=1)
        
        c_pred_tgt = self.target_count * (y_tgt + y_mat * (1 - self.sigma_correction_factor))
        c_pred_mat = self.matrix_count * y_mat

        # calculate nll
        nll_tgt = self.ZIP_nll(c_pred_tgt, c_tgt, self.target_inflat_prob)
        nll_mat = self.ZIP_nll(c_pred_mat, c_mat, self.matrix_inflat_prob)
        mse_tgt = F.mse_loss(self.target_count, c_tgt.float())
        mse_mat = F.mse_loss(self.matrix_count, c_mat.float())

        loss = nll_tgt + nll_mat + self.count_loss_weight * (mse_tgt + mse_mat)
        if return_loss:
            return loss, nll_tgt, nll_mat, mse_tgt, mse_mat
        else:
            return loss




