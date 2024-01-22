from torch import nn
import torch

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class GroupRMSELoss(nn.Module):
    def __init__(self, n, eps=1e-8,cff=1.):
        super().__init__()
        self.n = n
        self.cff = cff
        self.cts1 = []
        self.cts2 = []
        for _ in range(n):
            self.cts1.append(RMSELoss(eps))
            self.cts2.append(RMSELoss(eps))
    def forward(self,preds,targets):
        loss = 0.
        for i in range(self.n):
            p1,p2 = preds[i][:,:-5,...],preds[i][:,-5:,...]
            t1,t2 = targets[i][:,:-5,...],targets[i][:,-5:,...]
            l1 = self.cts1[i](p1,t1)
            l2 = self.cff*self.cts2[i](p2,t2)
            loss += (l1+l2)
        return loss