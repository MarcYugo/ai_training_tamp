import numpy as np
import pandas as pd
import xarray as xr
import argparse,os,torch

from utils.TSDataSet import TSDataSet
from torch.utils.data import DataLoader
from utils.TSCNN import TSCNN
from utils.Loss import GroupRMSELoss
from utils.Record import HisRecorder,CheckSaver

parser = argparse.ArgumentParser()
parser.add_argument("--assume",type=str,default='',help='checkpoint')
parser.add_argument("--epochs",type=int,default=30,help='training epochs')
parser.add_argument("--log",type=str,default='test',help='log name')
parser.add_argument("--train_root",type=str,default='data',help='log name')
args = parser.parse_args()

path = args.train_root
# 如果不是 dask array，按时刻进行分块转换为 dask array
def chunk_time(ds):
    dims = {k:v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds
# 数据载入
def load_dataset():
    ds = []
    for y in range(2007, 2012):
        data_name = os.path.join(path, f'weather_round1_train_{y}')
        x = xr.open_zarr(data_name, consolidated=True)
        print(f'{data_name}, {x.time.values[0]} ~ {x.time.values[-1]}')
        ds.append(x)
    ds = xr.concat(ds, 'time')
    ds = chunk_time(ds)
    return ds

print('Load data ...')
ds = load_dataset().x # 载入的数据集中含有其他信息，x是 array 形式的数据

from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
scaler = GradScaler()

def train_one_epoch(model,train_loader,criterion,optimizer,scaler):
    torch.cuda.empty_cache()
    model.train()
    train_loss = 0.0
    le = len(train_loader)
    
    for i, (inputs, target) in enumerate(train_loader): # (bs,t,c,x,y)
        inputs = inputs.permute([0,2,1,3,4]) # (bs,c,t,x,y)
        target = target.permute([0,2,1,3,4])
        inputs = inputs.cuda()
        target = target.cuda()
        targets = [target[:,:,:4,...],
                   target[:,:,2:6,...],
                   target[:,:,4:8,...],
                   target[:,:,6:10,...],
                   target[:,:,8:12,...],    
                   target[:,:,10:14,...],
                   target[:,:,12:16,...],
                   target[:,:,14:18,...],
                   target[:,:,16:20,...],
                   target[:,:,18:,...]]
        
        with autocast():
            outputs = model(inputs) # (bs,c,t,x,y)
            # output = output.permute([0,2,1,3,4]) # (bs,t,c,x,y)
            loss = criterion(outputs, targets)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    	# scaler.scale(loss).backward()
    	# scaler.step(optimizer)
    	# scaler.update()
        
        train_loss += loss.item()
        print(f'step {i+1:02}/{le}   Step Train Loss: {loss.item():.6f}',end='\r')
    
    return train_loss/le

def validate_one_epoch(model,data_loader,criterion):
    model.eval()
    l = 0.
    with torch.no_grad():
        for i, (inputs, target) in enumerate(data_loader): # (bs,t,c,x,y)
            inputs = inputs.permute([0,2,1,3,4]) # (bs,c,t,x,y)
            target = target.permute([0,2,1,3,4])
            inputs = inputs.cuda()
            target = target.cuda()
            targets = [target[:,:,:4,...],
                   target[:,:,2:6,...],
                   target[:,:,4:8,...],
                   target[:,:,6:10,...],
                   target[:,:,8:12,...],    
                   target[:,:,10:14,...],
                   target[:,:,12:16,...],
                   target[:,:,14:18,...],
                   target[:,:,16:20,...],
                   target[:,:,18:,...]]
            
            outputs = model(inputs) # (bs,c,t,x,y)
            # output = output.permute([0,2,1,3,4]) # (bs,t,c,x,y)
            loss = criterion(outputs, targets)
            
            l += loss.item()
            
    return l/len(data_loader)

print('Training ...')

train_loader = DataLoader(TSDataSet(ds[:-22*4,...],2,20), batch_size=1, shuffle=True)
val_loader = DataLoader(TSDataSet(ds[-22*4:,...],2,20), batch_size=1, shuffle=False)

epochs = args.epochs
model = TSCNN(pred_step=2)
criterion = GroupRMSELoss(n=model.block_n,cff=5.)
optimizer = torch.optim.AdamW(model.parameters(), 0.00002)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs+1)
recorder = HisRecorder(args.log)
che_saver = CheckSaver(args.log)
las_saver = CheckSaver(args.log,sfx='last')
s = 1

if args.assume != '':
    print('Loading checkpoint ...')
    stat = torch.load(args.assume)
    model.load_state_dict(stat['state_dict'])
    optimizer.load_state_dict(stat['optimizer'])
    scheduler.load_state_dict(stat['scheduler'])
    s = stat['epoch']
    
    che_saver.best_val_loss = stat['best_val_loss']
    che_saver.best_train_loss = stat['best_trn_loss']
    
model.cuda()

for epoch in range(s,epochs+1):
    # trn_loss,val_loss = 0.,0.
    trn_loss = train_one_epoch(model,train_loader,criterion,optimizer,scaler)
    val_loss = validate_one_epoch(model,val_loader,criterion)
    che_saver.save_check(model,optimizer,scheduler,trn_loss,val_loss,epoch)
    if epoch < 30:
        scheduler.step()
    recorder.record_info(epoch,trn_loss,val_loss)
    print(' '*100,end='\r')
    print(f'Epoch: {epoch:03}  Train Loss: {trn_loss:.8f}  Val Loss: {val_loss:.8f}')
    las_saver.save_last(model,optimizer,scheduler,trn_loss,val_loss,epoch)
recorder.save_hist()
