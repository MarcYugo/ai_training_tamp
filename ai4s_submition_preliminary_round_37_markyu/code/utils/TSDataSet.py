import xarray as xr
from torch.utils.data.dataset import Dataset

class TSDataSet(Dataset):
    def __init__(self,data_xarray,in_window,out_window):
        super().__init__()
        self.xarr = data_xarray
        self.in_win = in_window
        self.out_win = out_window
        self.sli_win = in_window+out_window
        # self.norm_p = norm_param
    def __getitem__(self,i):
        in_seq = self.xarr[i:i+self.in_win,...].values
        tgt_seq = self.xarr[i:i+self.sli_win,...].values
        return in_seq,tgt_seq
    def __len__(self):
        return self.xarr.shape[0]-self.sli_win
