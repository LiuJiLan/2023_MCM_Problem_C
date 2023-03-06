import torch
from torch import nn

# 注意, torchinfo是第三方库
from torchinfo import summary


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # 26个字母所以26, 50是参考常见的嵌入维度, word的情况一般300
        self.embedding = nn.Embedding(26, 50)
        # Input:(*) Output:(*, H) H是嵌入维度
        # input:(b, len(word), len(dict)) output(b, 1, mul(len(word), len(dict), H))

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(5 * 26 * 50, 100)
        self.linear2 = nn.Linear(100, 7)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        # b, _, _ = input.shape

        # out = self.embedding(input).view(b, -1)
        # 相当于out = self.flatten(out)
        # , flatten没有权重和偏置

        out = self.embedding(input)
        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    # input (b, 5, 26)
    dummy_input = torch.zeros((1, 5, 26)).long()

    summary(
        Model(),
        input_size=dummy_input.shape,  # 要求携带batch
        # batch_dim=0,  # 不要用这个参数, 暂时不知道是bug还是什么, 会多一维
        # device=
        dtypes=[dummy_input.dtype]
    )

    generate_ONNX_model = True
    if generate_ONNX_model:
        torch.onnx.export(
            Model(),
            dummy_input,
            "./model.onnx",
            # export_params=False,
            # 虽然不需要存储权重, 但这样会导致weight和bias在外部, 非常的难看
            input_names=["input"],
            output_names=["output"],
            # 写了会显式的标出batch, 但是会导致早期的数据变得复杂
            # 因为这个参数标注了那些维度是可变的
            # 改为flatten就没有这种情况
            dynamic_axes={
                "input": {0: "b"},
                "output": {0: "b"}
            }
        )
