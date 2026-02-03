import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class ResNetMLP(nn.Module):
    def __init__(
        self, 
        input_dim=10, 
        hidden_dim=128, 
        out_dim=3, 
        dropout=0.1, 
        n_blocks=3  # <--- [NEW] Số lượng ResNet block muốn dùng
    ):
        super().__init__()
        
        # 1. ENTRY: Input -> Hidden
        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. BODY: Stack nhiều ResNet Block
        # Dùng ModuleList để chứa các block
        self.blocks = nn.ModuleList([
            ResBlock(dim=hidden_dim, dropout=dropout) 
            for _ in range(n_blocks)
        ])

        # 3. HEAD: Hidden -> 64 -> Output (Giảm sốc)
        # Thay vì 128 -> 3, ta dùng 128 -> 64 -> 3
        head_hidden = hidden_dim // 2  # Ví dụ: 128 -> 64
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(), # Thêm phi tuyến tính ở lớp đệm
            # Có thể thêm Layernorm ở đây nếu muốn cực kỳ ổn định
            # nn.LayerNorm(head_hidden), 
            nn.Linear(head_hidden, out_dim)
        )

    def forward(self, x):
        # Entry
        x = self.entry(x)
        
        # Body (Qua từng block ResNet)
        for block in self.blocks:
            x = block(x)
            
        # Head
        x = self.head(x)
        return x

# Tách ResBlock ra cho gọn code
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Layer 1
        self.linear1 = nn.Linear(dim, dim)
        self.bn1     = nn.BatchNorm1d(dim)
        
        # Layer 2
        self.linear2 = nn.Linear(dim, dim)
        self.bn2     = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        # Skip Connection
        out = out + identity
        out = self.act(out) # Activation sau khi cộng
        
        return out

# Hàm build
def build_mlp(input_dim=10, hidden_dims=(128, 64), out_dim=3, dropout=0.1, bias=True):
    # Lấy tham số đầu tiên làm chiều rộng mạng
    width = hidden_dims[0] if len(hidden_dims) > 0 else 128
    
    # Ở đây mình hard-code là 3 blocks theo ý bạn, 
    # hoặc bạn có thể thêm tham số n_blocks vào hàm build_mlp
    return ResNetMLP(
        input_dim=input_dim, 
        hidden_dim=width, 
        out_dim=out_dim, 
        dropout=dropout,
        n_blocks=3 # <--- 3 Block 128
    )

def build_residual_mlp_from_state_dict(state_dict):
    entry_weight = state_dict.get("entry.0.weight")
    if entry_weight is None:
        raise ValueError("state_dict missing entry.0.weight for residual MLP")
    hidden_dim, input_dim = entry_weight.shape

    head_out_weight = state_dict.get("head.2.weight")
    if head_out_weight is None:
        raise ValueError("state_dict missing head.2.weight for residual MLP")
    out_dim = head_out_weight.shape[0]

    block_indices = set()
    for key in state_dict.keys():
        match = re.match(r"blocks\.(\d+)\.", key)
        if match:
            block_indices.add(int(match.group(1)))
    n_blocks = max(block_indices) + 1 if block_indices else 0

    model = ResNetMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=0.1,
        n_blocks=n_blocks,
    )
    return model

if __name__ == "__main__":
    model = build_mlp()
    print(model)
    print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))
    
    x = torch.randn(4, 10)
    print("Output shape:", model(x).shape)
