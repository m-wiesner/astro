import torch
import torch.nn.functional as F
import torch.nn as nn


class FactoredLinear(nn.Module):
    '''
        A model to replace linear layers with a factorized verion based on 
        specific features that should be shared.
    '''
    def __init__(self, hdim, extra_units, featurized_units, num_classes):
        '''
            Inputs:
                :param hdim: the dimension of the input features
                :param extra_units: the number of extra (non-featurized) units
                :param featurized_units: the dictionary of units to features
                :param: the number of classes for each feature
        '''
        super(FactoredLinear, self).__init__() 
        self.extra_syms_output = nn.Linear(hdim, len(extra_units))
        code_lists = [[] for i in num_classes]
        for u, u_feats in featurized_units.items():
            if u not in extra_units:
                for lst, unit, nc in zip(code_lists, u_feats, num_classes):
                    lst.append(
                        F.one_hot(torch.LongTensor([unit]), num_classes=nc)
                    )
        feat_mats = [torch.cat(code_list, dim=0) for code_list in code_lists]  
        feat_mat = torch.cat(feat_mats, dim=-1).to(torch.float32)
        self.register_buffer('feat_mat', feat_mat)
        self.linear_out = nn.Linear(hdim, sum(num_classes))
        
    def forward(self, x):
        z = self.linear_out(x) @ self.feat_mat.transpose(0, 1)
        extra = self.extra_syms_output(x)
        y = torch.cat([extra, z], dim=-1)
        return y


class FactoredLinear2(nn.Module):
    '''
        A model to replace linear layers with a factorized verion based on 
        specific features that should be shared.
    '''
    def __init__(self, hdim, extra_units, featurized_units, num_classes):
        '''
            Inputs:
                :param hdim: the dimension of the input features
                :param extra_units: the number of extra (non-featurized) units
                :param featurized_units: the dictionary of units to features
                :param: the number of classes for each feature
        '''
        super(FactoredLinear2, self).__init__() 
        self.extra_syms_output = nn.Linear(hdim, len(extra_units), bias=False)
        code_lists = [[] for i in num_classes]
        for u, u_feats in featurized_units.items():
            if u not in extra_units:
                for lst, unit, nc in zip(code_lists, u_feats, num_classes):
                    lst.append(
                        F.one_hot(torch.LongTensor([unit]), num_classes=nc)
                    )
        feat_mats = [torch.cat(code_list, dim=0) for code_list in code_lists]  
        feat_mat = torch.cat(feat_mats, dim=-1).to(torch.float32)
        self.register_buffer('feat_mat', feat_mat)
        self.linear_out = nn.Linear(hdim, sum(num_classes), bias=False)
        num_outputs = len(extra_units) + len(featurized_units)
        self.bias_out = nn.Parameter(torch.randn(num_outputs,))
        
    def forward(self, x):
        z = self.linear_out(x) @ self.feat_mat.transpose(0, 1)
        extra = self.extra_syms_output(x)
        y = torch.cat([extra, z], dim=-1) + self.bias_out
        return y   
