import torch
import torch.nn as nn

class DAG_MTL_Model(nn.Module):
    def __init__(self, input_dim=17, trunk_dim=64, hidden_dim=32):
        super().__init__()
        
        # Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 1. Diabetes Head
        self.diab_layer = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.diab_out = nn.Linear(hidden_dim, 1)
        
        # 2. CHF Head (Takes Trunk + Diab Hidden)
        self.chf_layer = nn.Sequential(
            nn.Linear(trunk_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.chf_out = nn.Linear(hidden_dim, 1)
        
        # 3. Sepsis Head (Takes Trunk + CHF Hidden)
        self.sep_layer = nn.Sequential(
            nn.Linear(trunk_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.sep_out = nn.Linear(hidden_dim, 1)
        
        # 4. AKI Head (Takes Trunk + Sepsis Hidden)
        self.aki_layer = nn.Sequential(
            nn.Linear(trunk_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.aki_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        trunk_emb = self.trunk(x)
        
        # Diabetes
        h_diab = self.diab_layer(trunk_emb)
        out_diab = self.diab_out(h_diab)
        
        # CHF (stop-gradient on h_diab)
        chf_in = torch.cat([trunk_emb, h_diab.detach()], dim=1)
        h_chf = self.chf_layer(chf_in)
        out_chf = self.chf_out(h_chf)
        
        # Sepsis (stop-gradient on h_chf)
        sep_in = torch.cat([trunk_emb, h_chf.detach()], dim=1)
        h_sep = self.sep_layer(sep_in)
        out_sep = self.sep_out(h_sep)
        
        # AKI (stop-gradient on h_sep)
        aki_in = torch.cat([trunk_emb, h_sep.detach()], dim=1)
        h_aki = self.aki_layer(aki_in)
        out_aki = self.aki_out(h_aki)
        
        # Return logits
        return torch.cat([out_diab, out_chf, out_sep, out_aki], dim=1)
