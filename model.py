import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # 26个字母所以26, 50是参考常见的嵌入维度, word的情况一般300
        self.embedding = nn.Embedding(26, 50)
        # Input:(*) Output:(*, H) H是嵌入维度
        # input:(b, len(word), len(dict)) output(b, 1, mul(len(word), len(dict), H))

        self.linear1 = nn.Linear(5 * 26 * 50, 100)
        self.linear2 = nn.Linear(100, 7)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        b, _, _ = input.shape

        out = self.embedding(input).view(b, -1)

        out = self.linear1(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.softmax(out)
        return out
