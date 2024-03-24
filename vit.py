from torch import nn 
import torch 

class ViT(nn.Module):
    def __init__(self,emb_size=16):
        super().__init__()
        self.patch_size=4
        self.patch_count=28//self.patch_size # 7
        
        self.conv=nn.Conv2d(in_channels=1,out_channels=self.patch_size**2,kernel_size=self.patch_size,padding=0,stride=self.patch_size) # 图片转patch
        self.patch_emb=nn.Linear(in_features=self.patch_size**2,out_features=emb_size)    # patch做emb
        self.cls_token=nn.Parameter(torch.rand(1,1,emb_size))   # 分类头输入
        self.pos_emb=nn.Parameter(torch.rand(1,self.patch_count**2+1,emb_size))   # position位置向量 (1,seq_len,emb_size)
        self.tranformer_enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emb_size,nhead=2,batch_first=True),num_layers=3)   # transformer编码器
        self.cls_linear=nn.Linear(in_features=emb_size,out_features=10) # 手写数字10分类
        
    def forward(self,x): # (batch_size,channel=1,width=28,height=28)
        x=self.conv(x) # (batch_size,channel=16,width=7,height=7)
        
        x=x.view(x.size(0),x.size(1),self.patch_count**2)   # (batch_size,channel=16,seq_len=49)
        x=x.permute(0,2,1)  # (batch_size,seq_len=49,channel=16)
        
        x=self.patch_emb(x)   # (batch_size,seq_len=49,emb_size)
        
        cls_token=self.cls_token.expand(x.size(0),1,x.size(2))  # (batch_size,1,emb_size)
        x=torch.cat((cls_token,x),dim=1)   # add [cls] token
        x=self.pos_emb+x
        
        y=self.tranformer_enc(x) # 不涉及padding，所以不需要mask
        return self.cls_linear(y[:,0,:])   # 对[CLS] token输出做分类
    
if __name__=='__main__':
    vit=ViT()
    x=torch.rand(5,1,28,28)
    y=vit(x)
    print(y.shape)