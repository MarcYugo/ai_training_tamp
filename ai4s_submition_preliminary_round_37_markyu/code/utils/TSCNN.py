from torch import nn

class Layer(nn.Module):
    def __init__(self,in_chs,kernel_size,padding,coord_size):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chs,in_chs,kernel_size=kernel_size,padding=padding)
        self.ln1 = nn.Sequential(
            nn.LayerNorm(coord_size),nn.GELU())
        self.ffn = nn.Conv3d(in_chs,in_chs,kernel_size=1)
        self.ln2 = nn.Sequential(
            nn.LayerNorm(coord_size),nn.GELU())
    def forward(self,x):
        u = self.conv1(self.ln1(x)) + x
        u = self.ffn(self.ln2(u)) + u
        return u

class Block(nn.Module):
    def __init__(self,in_chs=70,hid_chs=128,kernel_size=(5,7,7),padding=(4,3,3),coord_size=(161,161)):
        super().__init__() # input (bs,70,2,161,161)
        self.embed = nn.Conv3d(in_chs,hid_chs,kernel_size=kernel_size,padding=padding)
        self.layer = nn.Sequential(
            Layer(hid_chs,kernel_size=3,padding=1,coord_size=coord_size),
            Layer(hid_chs,kernel_size=5,padding=2,coord_size=coord_size)
            )
        self.out_map = nn.Conv3d(hid_chs,in_chs,kernel_size=5,padding=2) # output (bs,70,2+nt,161,161)

    def forward(self,x):
        x = self.embed(x)
        x = self.layer(x)
        x = self.out_map(x)
        return x

class TSCNN(nn.Module):
    def __init__(self,pred_step=4):
        super().__init__()
        # pred_step = pad1*2 - k1 + 1
        # pad1 = (k1 + pred_step - 1) // 2
        ks = (5,7,7)
        p1 = (ks[0] + pred_step - 1)//2
        pd = (p1,3,3)
        self.t = 20
        self.pred_step = pred_step
        self.block_n = self.t//pred_step

        for i in range(self.block_n):
            setattr(
                self,
                f'block{i}',
                Block(kernel_size=ks,hid_chs=192,padding=pd)
            )
        
    def forward(self,x):
        feats = []

        for i in range(self.block_n):
            block = getattr(self,f'block{i}')
            u = block(x)
            feats.append(u) # (bs,70,2*pred_step,161,161)
            x = u[:,:,-self.pred_step:,...]
        # i: 0 s: 0 l: in_window + pred_step
        # i: 1 s: in_window l: 2*pred_step
        # i: 2 s: in_window + 1*pred_step l: 2*pred_step
        
        return feats