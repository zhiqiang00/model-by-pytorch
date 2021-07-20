import math
import torch

from torch.nn.parameter import Parameter
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """

        :param in_features: 数值大小 不是feature 用in_num_feature 更合适
        :param out_features:
        :param bias:
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            # Parameters与register_parameter都会向parameters写入参数，
            # 但是后者可以支持字符串命名
            self.register_parameter('bias', None)
        self.reset_parameters()


    # 初始权重
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    """
    前馈运算 即计算A~ X W(0)
    input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
    直接输入与权重之间进行torch.mm操作，得到support，即XW
    support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias
    """
    def forward(self, input, adj):
        # torch.mm(a, b)是矩阵a和b矩阵相乘
        # torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + '(' \
                + str(self.in_features) + '->' \
                + str(self.out_features) + ')'

