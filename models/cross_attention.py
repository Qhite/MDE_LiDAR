import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def Positional_encoding(batch, H, d_model):
    PE = []
    for i in range(H):
        PE.append(math.sin(i/pow(10000,2/d_model)))
    return torch.tensor(PE).unsqueeze(0).unsqueeze(2).expand(batch,1,H,-1)
    
class Cross_Attention_Block(nn.Module):
    def __init__(self, cfg=None, feature_dim=[]):
        super(Cross_Attention_Block, self).__init__()
        self.num_head = cfg.model.num_head
        self.d_model = cfg.model.d_model
        self.head_dim = cfg.model.d_model // cfg.model.num_head
        self.bin_size = cfg.model.bin_size
        self.H, self.W = [228, 304]

        self.Conv = nn.Sequential(                                                          
            nn.Conv2d(feature_dim[-1], feature_dim[-1], kernel_size=3, stride=2, padding=1, bias=False), # depth-wise
            nn.BatchNorm2d(feature_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(feature_dim[-1], 1, kernel_size=3, stride=1, padding=1, bias=False), # point-wise
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )
        
        self.Linear = nn.Sequential(
            nn.Linear(cfg.data.lidar_size, self.W//4),
            nn.LeakyReLU(),
        )

        # Q, K, V Linear layers
        self.W_q = nn.Linear(self.W//4, self.d_model, bias=False) # self.W//4
        self.W_k = nn.Linear(self.W//4, self.d_model, bias=False)
        self.W_v = nn.Linear(self.W//4, self.d_model, bias=False)

        # Output
        self.W_o1 = nn.Sequential(nn.BatchNorm2d(1),
                                  nn.Linear(self.d_model, self.W//2),
                                  nn.LeakyReLU(),
                                  )
        
        self.W_o2 = nn.Sequential(nn.Linear(self.W//2, self.bin_size),
                                  nn.ReLU(),
                                  )
        

    def forward(self, data_dict):
        x, y = data_dict["feature"], data_dict["lidar"]
        batch_size = x.size(0)
        position = Positional_encoding(batch_size, self.H//4, self.d_model).to(x.device) # self.H//4
        x = self.Conv(x) + position
        y = self.Linear(y)

        # Q, K, V Linear Projection
        Q = self.W_q(x).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        K = self.W_k(y).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        V = self.W_v(y).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)

        # Get Score
        Score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        Score = F.softmax(Score, dim=-1)

        # Matmul Score and Value
        Attention_Value = torch.matmul(Score, V)

        # Concatenate and Feed Forward/Dense Layer
        Concate_Attention_Value = Attention_Value.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        Q_res = (Q.transpose(1,2).view(batch_size, -1, self.num_head * self.head_dim))
        res = Concate_Attention_Value + Q_res
        output = F.avg_pool2d(self.W_o1(res.unsqueeze(1)), (self.H//4, 1)).view(batch_size, -1) # self.H//4
        output = self.W_o2(output)

        return output
    
class CA_Block(nn.Module):
    def __init__(self, cfg=None, feature_dim=[]):
        super(CA_Block, self).__init__()
        self.num_head = cfg.model.num_head
        self.d_model = cfg.model.d_model
        self.head_dim = cfg.model.d_model // cfg.model.num_head
        self.bin_size = cfg.model.bin_size
        self.H, self.W = [228, 304]

        set_feature = 0
        f_channel = feature_dim[-set_feature-1]

        self.Conv = nn.Sequential(
            nn.Conv2d(f_channel, f_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(f_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # Positional Encoding
        pe_h = math.ceil(self.H//pow(2, set_feature+1))+1
        self.PE = nn.Parameter(torch.rand(pe_h, 1), requires_grad=True)

        # Q, K, V Linear layers
        feat_w = math.ceil(self.W//pow(2, set_feature+1))
        self.W_q = nn.Linear(feat_w, self.d_model, bias=False)
        self.W_k = nn.Linear(cfg.data.lidar_size, self.d_model, bias=False)
        self.W_v = nn.Linear(cfg.data.lidar_size, self.d_model, bias=False)

        # Output
        self.W_o = nn.Sequential(nn.Linear(self.d_model, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, self.bin_size),
                                 )

    def forward(self, data_dict):
        x, y = data_dict["feature"], data_dict["lidar"]
        batch_size = x.size(0)

        x = self.Conv(x)
        x = F.pad(x, (0,0,0,1))# + self.PE # Add Token at the end

        # Q, K, V Linear Projection
        Q = self.W_q(x).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        K = self.W_k(y).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        V = self.W_v(y).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)

        # Get Score
        Score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        Score = F.softmax(Score, dim=-1)

        # Matmul Score and Value
        Attention_Value = torch.matmul(Score, V)

        # Concatenate and Feed Forward/Dense Layer
        Concate_Attention_Value = Attention_Value.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        bins = Concate_Attention_Value[:,-1,:]
        bins = self.W_o(bins)

        return bins