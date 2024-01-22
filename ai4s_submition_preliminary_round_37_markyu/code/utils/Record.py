import os,torch
from datetime import datetime

class HisRecorder:
    def __init__(self,log_name='test'):
        self.name = log_name
        self.history = {}
        t = datetime.now().strftime('%Y-%m-%d-%H-%M')
        fname = self.name + '_' + t
        self.fname = fname
    def record_info(self,epoch,train_loss,val_loss):
        self.history[epoch] = (train_loss,val_loss)
    def save_hist(self):
        f = open('./logs/'+self.fname,'w',encoding='utf-8')
        f.write(str(self.history))
        f.close()
        print(self.fname)
    def clean_hist(self):
        self.history = {}

class CheckSaver:
    def __init__(self,name,sfx='best'):
        self.best_train_loss = 100
        self.best_val_loss = 100
        self.name = name
        self.suffix = sfx
        
    def save_check(self,model,optim,sche,trn_loss,val_loss,epoch):
        flag = self.best_val_loss == val_loss and self.best_train_loss > trn_loss
        flag = self.best_val_loss > val_loss or flag
        if flag:
            self.best_val_loss = val_loss
            self.best_train_loss = trn_loss
            stat = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_val_loss': val_loss,
                    'best_trn_loss': trn_loss,
                    'optimizer' : optim.state_dict(),
                    'scheduler' : sche.state_dict()
                }
            torch.save(stat,f'./model/{self.name}_{self.suffix}.pth.tar')
            
    def save_last(self,model,optim,sche,trn_loss,val_loss,epoch):
        stat = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_val_loss': val_loss,
                'best_trn_loss': trn_loss,
                'optimizer' : optim.state_dict(),
                'scheduler' : sche.state_dict()
            }
        torch.save(stat,f'./model/{self.name}_{self.suffix}.pth.tar')