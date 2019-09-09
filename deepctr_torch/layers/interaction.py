import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term
    

class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """
    def __init__(self, input_feature_num,layer_num=2, seed=1024,**kwargs):
        super(CrossNet, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(input_feature_num,1))) for i in range(self.layer_num)])
        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(torch.empty(input_feature_num,1))) for i in range(self.layer_num)])
        

    def forward(self,inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l