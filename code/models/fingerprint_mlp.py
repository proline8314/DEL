import torch
import torch.nn as nn
import torch.nn.functional as F


class FingerprintMLP(nn.Module):
    def __init__(self, fpsize: int = 2048, hidden_size: int = 64, target_size: int = 2):
        super(FingerprintMLP, self).__init__()
        self.fp_to_hidden = self._get_multilayer_perceptron(
            fpsize, hidden_size, hidden_size)
        self.singleton_to_doubleton = self._get_multilayer_perceptron(
            hidden_size * 2, hidden_size, hidden_size)
        self.sing_and_doub_to_tripleton = self._get_multilayer_perceptron(
            hidden_size * 6, hidden_size, hidden_size)
        self.sing_and_doub_and_trip_to_target = self._get_multilayer_perceptron(
            hidden_size * 7, hidden_size, target_size)

    def forward(self, **net_input):
        # get fingerprint
        fp1, fp2, fp3 = net_input["BB1_fp"], net_input["BB2_fp"], net_input["BB3_fp"]
        hid_1, hid_2, hid_3 = self.fp_to_hidden(fp1), self.fp_to_hidden(
            fp2), self.fp_to_hidden(fp3)
        hid_12, hid_23, hid_13 = torch.cat(
            [hid_1, hid_2], dim=1), torch.cat([hid_2, hid_3], dim=1), torch.cat([hid_1, hid_3], dim=1)
        hid_12, hid_23, hid_13 = self.singleton_to_doubleton(
            hid_12), self.singleton_to_doubleton(hid_23), self.singleton_to_doubleton(hid_13)
        hid_123 = torch.cat(
            [hid_1, hid_2, hid_3, hid_12, hid_23, hid_13], dim=1)
        hid_123 = self.sing_and_doub_to_tripleton(hid_123)
        output = self.sing_and_doub_and_trip_to_target(
            torch.cat([hid_1, hid_2, hid_3, hid_12, hid_23, hid_13, hid_123], dim=1)
        )
        return output

    def _get_multilayer_perceptron(self, input_size, hidden_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
