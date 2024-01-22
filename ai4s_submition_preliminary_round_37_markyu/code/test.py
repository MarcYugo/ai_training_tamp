import numpy as np
import pandas as pd
import xarray as xr
import argparse,os,torch,zipfile
from datetime import datetime
from tqdm import tqdm

from utils.TSCNN import TSCNN

parser = argparse.ArgumentParser()
parser.add_argument("--assume",type=str,default='',help='model parameters')
parser.add_argument("--test_root",type=str,default='./data/',help='test data root')
parser.add_argument("--output",type=str,default='./submit/',help='submit files root')
args = parser.parse_args()

# def main(args):
pred_files = [f'{args.test_root}/weather_round1_test/input/{i}' for i in os.listdir(f'{args.test_root}/weather_round1_test/input/')]

model = TSCNN(pred_step=2)
model.load_state_dict(torch.load(args.assume)['state_dict'])
model.cuda()

for i,p in enumerate(tqdm(pred_files,ncols=50)):
    model.eval()
    inputs = torch.load(p)
    inputs = inputs.permute([1,0,2,3])
    inputs = inputs.unsqueeze(0).cuda()
    # print(inputs.shape)
    with torch.no_grad():
        feats = model(inputs)
    for i in range(len(feats)):
        feats[i] = feats[i][0,-5:,-2:,...]
    feats = torch.cat(feats,dim=1)
    feats = feats.cpu()
    feats = feats.permute([1,0,2,3])
    # feats = feats.cpu()
    # print(feats.shape)
    save_file = p.split('/')[-1]
    torch.save(feats.half(), f'output/{save_file}', ) # 精度转为半精度浮点类型
    # print(f'{i+1:03}/{len(pred_files)}',end='\r')
print('Model has completed prediction.')

t = datetime.now().strftime('%Y%m%d_%H%M%S')
zipName = f'{args.output}/submit_{t}.zip' #压缩后文件的位置及名称
f = zipfile.ZipFile( zipName, 'w', zipfile.ZIP_DEFLATED )
i = 0
for dirpath, dirnames, filenames in os.walk('./output'):
    for filename in filenames:
        i += 1
        print(f'{filename} has been zipped  {i:03}/{300}',end='\r')
        f.write(os.path.join(dirpath,filename))
f.close()
print('All predicting results have been zipped.')
