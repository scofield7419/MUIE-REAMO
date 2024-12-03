
# import paddle
# from paddlemix.models.blip2.Qformer import BertLMHeadModel
# from paddlenlp.transformers.bert.configuration import BertConfig
# from paddle.nn import Transformer

import torch 
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features=None, out_features=None, num_layers=1):
        super().__init__()
        modules = [nn.Linear(in_features=in_features, out_features=out_features)]

        for _ in range(1, num_layers):
            modules.append(nn.GELU())
            modules.append(nn.Linear(in_features=out_features, out_features=out_features))

        self.layer =  nn.Sequential(*modules)
    
    def forward(self, x):
        return self.layer(x)
    
    @property
    def config(self):
        return {"mm_projector_type": "mlp"}
    
    @property
    def device(self):
        return self.layer[0].weight.device
    
    @property
    def dtype(self):
        return self.layer[0].weight.dtype
