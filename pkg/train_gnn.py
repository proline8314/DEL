# %% imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import test_graph_dataset
from models import basic_gnn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm


# %% learnable ZIP(Zero-Inflated Poisson) loss
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


# %% load data
batch_size = 4096
dataset = test_graph_dataset.GraphDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_graph_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_graph_dataset, batch_size=batch_size, shuffle=False)

# %% train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# predict target and matrix mean
model = basic_gnn.GNN(7, 2).to(device)


# loss function
criterion = ZIPLoss()
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()), lr=0.01)

# train
epoch = 5
for e in range(epoch):
    model.train()
    train_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(data.y, out)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(loss.item())
        print(criterion.total_count, criterion.matrix_inflat_prob,
              criterion.target_inflat_prob)
    print(f"Epoch {e+1}, Train Loss: {train_loss / len(train_loader)}")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(data.y, out)
            test_loss += loss.item()
    print(f"Epoch {e+1}, Test Loss: {test_loss / len(test_loader)}")

# print loss parameters
print(criterion.total_count, criterion.matrix_inflat_prob,
      criterion.target_inflat_prob)


# %% save model
torch.save(model.state_dict(), "model.pth")
