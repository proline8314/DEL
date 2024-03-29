import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import test_graph_dataset
from models import basic_gnn
from scipy.stats import pearsonr
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class ZIPLoss(nn.Module):
    def __init__(self, label_size: int = 6, target_size: int = 4, matrix_size: int = 2, eps: float = 1e-8):
        super(ZIPLoss, self).__init__()
        self.label_size = label_size
        self.target_size = target_size
        self.matrix_size = matrix_size
        self.loss_func = nn.PoissonNLLLoss(
            log_input=False, full=False, reduction='none')
        self.eps = eps
        # parameters
        self.total_count = nn.parameter.Parameter(
            torch.tensor(1.0), requires_grad=True)
        self.matrix_inflat_prob = nn.parameter.Parameter(
            torch.tensor(0.0), requires_grad=True)
        self.target_inflat_prob = nn.parameter.Parameter(
            torch.tensor(0.0), requires_grad=True)

    def ZIP_nll(self, lamda, count, inflat_prob):
        # use sigmoid to scale prob
        inflat_prob = F.sigmoid(inflat_prob)

        zero_nll = -torch.log(inflat_prob +
                              (1 - inflat_prob) * torch.exp(-lamda) + self.eps)
        non_zero_nll = self.loss_func(
            lamda, count) - torch.log(1 - inflat_prob + self.eps)
        count_is_zero = (count == 0).type(torch.float)
        nll = count_is_zero * zero_nll + (1 - count_is_zero) * non_zero_nll
        # average over batches and labels
        nll = torch.mean(nll)
        return nll

    def forward(self, y_true, y_pred):
        y_true = y_true.view(-1, self.label_size)
        y_true_target, y_true_matrix = y_true[:, :4], y_true[:, 4:]

        y_pred_target, y_pred_matrix = y_pred[:, 0], y_pred[:, 1]

        target_lambda = self.total_count * \
            torch.exp(y_pred_matrix + y_pred_target)
        matrix_lambda = self.total_count * torch.exp(y_pred_matrix)

        target_lambda = target_lambda.view(-1, 1).expand(-1, self.target_size)
        matrix_lambda = matrix_lambda.view(-1, 1).expand(-1, self.matrix_size)

        target_nll = self.ZIP_nll(
            target_lambda, y_true_target, self.target_inflat_prob)
        matrix_nll = self.ZIP_nll(
            matrix_lambda, y_true_matrix, self.matrix_inflat_prob)

        loss = target_nll + matrix_nll
        return loss


batch_size = 16
dataset = test_graph_dataset.GraphDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

# predict target and matrix mean
model = basic_gnn.GNN(7, 2)

# load model
"""
model.load_state_dict(torch.load(
    "/data02/gtguo/model.pth"))
"""

# test on test set
print("testing...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"number of parameters: {sum(p.numel() for p in model.parameters())}")
accordance_tensor = torch.tensor([[], []], dtype=torch.float)
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        y = data.y.view(-1 ,6)
        y_tar, y_mat = y[:, :4].float(), y[:, 4:].float()
        y_eff = (torch.mean(y_tar, dim=-1) - torch.mean(y_mat, dim=-1)).view(1, -1)

        data = data.to(device)
        out = model(data)
        out = out.detach().cpu()[:, 0].view(1, -1)
        
        accordance_tensor = torch.cat((accordance_tensor, torch.cat((y_eff, out), dim=0)), dim=1)

        if i >= 10:
            break

print("calculating pearson r...")
print(pearsonr(accordance_tensor[0], accordance_tensor[1]))
print("ploting...")
plt.plot(accordance_tensor[0], accordance_tensor[1], ".")
plt.savefig("corr_r.png")
        