import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

'''
Loss function for Matryoshka Representation Learning 
'''

class Matryoshka_CE_Loss(nn.Module):
	def __init__(self, relative_importance=None, **kwargs):
		super(Matryoshka_CE_Loss, self).__init__()
		self.criterion = nn.CrossEntropyLoss(**kwargs)
		self.relative_importance= relative_importance

	def forward(self, output, target):
		loss=0
		N= len(output)
		for i in range(N):
			rel = 1. if self.relative_importance is None else self.relative_importance[i] 
			loss+= rel*self.criterion(output[i], target)
		return loss

class BinarizeWeight(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        # flat_out and out access the same memory.
        out[out <= 0] = -1.0
        out[out >= 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g,None

class BinaryLinearLayer(nn.Linear):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def get_weight(self):
		binarizedWeight = BinarizeWeight.apply(self.weight)
		return binarizedWeight
	
	def forward(self, x):
		w = self.get_weight()
		x = F.linear(x, w, self.bias)
		return x


class MRL_Linear_Layer(nn.Module):
	def __init__(self, nesting_list: List, binary_nesting_list=List, num_classes=1000, efficient=False, **kwargs):
		super(MRL_Linear_Layer, self).__init__()
		self.nesting_list=nesting_list
		self.binary_nesting_list=binary_nesting_list
		self.num_classes=num_classes # Number of classes for classification
		self.efficient = efficient
		if self.efficient:
			setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))		
		else:	
			for i, num_feat in enumerate(self.nesting_list):
				setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))	
			
			for i, num_feat in enumerate(self.binary_nesting_list):
				setattr(self, f"binarized_nesting_classifier_{i}", BinaryLinearLayer(num_feat, self.num_classes, **kwargs))


	def reset_parameters(self):
		if self.efficient:
			self.nesting_classifier_0.reset_parameters()
		else:
			for i in range(len(self.nesting_list)):
				getattr(self, f"nesting_classifier_{i}").reset_parameters()
			
			for i in range(len(self.binary_nesting_list)):
				getattr(self, f"binarized_nesting_classifier_{i}").reset_parameters()


	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			if self.efficient:
				if self.nesting_classifier_0.bias is None:
					nesting_logits+= (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
				else:
					nesting_logits+= (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
			else:
				nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)
		
		for i, num_feat in enumerate(self.binary_nesting_list):
			nesting_logits += (getattr(self, f"binarized_nesting_classifier_{i}")(x[:, :num_feat]),)

		return nesting_logits


class FixedFeatureLayer(nn.Linear):
    '''
    For our fixed feature baseline, we just replace the classification layer with the following. 
    It effectively just look at the first "in_features" for the classification. 
    '''

    def __init__(self, in_features, out_features, **kwargs):
        super(FixedFeatureLayer, self).__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        if not (self.bias is None):
            out = torch.matmul(x[:, :self.in_features], self.weight.t()) + self.bias
        else:
            out = torch.matmul(x[:, :self.in_features], self.weight.t())
        return out
        
