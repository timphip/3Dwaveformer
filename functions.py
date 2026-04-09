import os
import numpy as np
import cv2
import math
import pandas as pd
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm



## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()





class UCF101VideoDataset(data.Dataset):
    def __init__(self, csv_path, data_root, frames_per_clip=28, img_x=256, img_y=342):
        self.data_root = data_root
        self.frames_per_clip = frames_per_clip
        self.img_x = img_x
        self.img_y = img_y

        df = pd.read_csv(csv_path)
        path_col  = 'clip_path'
        label_col = 'label'

        self.video_paths = df[path_col].tolist()

        try:
            self.labels = df[label_col].astype(int).tolist()
            num_classes = max(self.labels) + 1
            self.classes = [str(i) for i in range(num_classes)]
            self.class_to_idx = {str(i): i for i in range(num_classes)}
        except (ValueError, TypeError):
            self.classes = sorted(df[label_col].unique().tolist())
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.labels = [self.class_to_idx[c] for c in df[label_col].tolist()]

        print(f"Data scan complete! CSV: {csv_path}, total of {len(self.video_paths)} videos, {len(self.classes)} classes.")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = os.path.join(self.data_root, self.video_paths[index].lstrip('/'))
        label = self.labels[index]

        if index < 3:
            print(f"Video path: {video_path}, Exists: {os.path.exists(video_path)}")
        
        label = self.labels[index]

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_y, self.img_x))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return torch.zeros(3, self.frames_per_clip, self.img_x, self.img_y), torch.tensor(label, dtype=torch.long)

        indices = np.linspace(0, len(frames) - 1, self.frames_per_clip).astype(int)
        sampled_frames = [frames[i] for i in indices]

        tensor_frames = torch.FloatTensor(np.array(sampled_frames)) / 255.0
        tensor_frames = tensor_frames.permute(3, 0, 1, 2)
        tensor_frames = (tensor_frames - 0.5) / 0.5

        return tensor_frames, torch.tensor(label, dtype=torch.long)






















def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred
















## ---------------------- Model Definition ---------------------- ##
class Wave3D(nn.Module):
    def __init__(self, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dwconv = nn.Conv3d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        
        self.v0_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        # 【修正点1】: 将常数 c 替换为物理意义上的波速 v (对应公式中的 v)
        self.c = nn.Parameter(torch.ones(1) * 1.0)
        # 阻尼系数 (对应公式中的 alpha)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)

    @staticmethod
    def get_cos_map(N, device, dtype=torch.float):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    def _dct_along_last_dim(self, x, cos_map):
        return torch.matmul(x, cos_map.t())

    def _idct_along_last_dim(self, x, cos_map):
        return torch.matmul(x, cos_map)

    def dct_3d(self, x, cos_T, cos_H, cos_W):
        x = self._dct_along_last_dim(x, cos_W)
        x = x.transpose(-2, -1)
        x = self._dct_along_last_dim(x, cos_H)
        x = x.transpose(-2, -1)
        x = x.permute(0, 1, 3, 4, 2)
        x = self._dct_along_last_dim(x, cos_T)
        x = x.permute(0, 1, 4, 2, 3)
        return x

    def idct_3d(self, x, cos_T, cos_H, cos_W):
        x = self._idct_along_last_dim(x, cos_W)
        x = x.transpose(-2, -1)
        x = self._idct_along_last_dim(x, cos_H)
        x = x.transpose(-2, -1)
        x = x.permute(0, 1, 3, 4, 2)
        x = self._idct_along_last_dim(x, cos_T)
        x = x.permute(0, 1, 4, 2, 3)
        return x

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, T_dim, H, W = x.shape
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        x = self.linear(x)
        x, z = x.chunk(chunks=2, dim=-1)
        
        v0 = self.v0_proj(x)
        
        x  = x.permute(0, 4, 1, 2, 3).contiguous()
        z  = z.permute(0, 4, 1, 2, 3).contiguous()
        v0 = v0.permute(0, 4, 1, 2, 3).contiguous()
        
        cos_T = self.get_cos_map(T_dim, x.device, x.dtype).detach()
        cos_H = self.get_cos_map(H, x.device, x.dtype).detach()
        cos_W = self.get_cos_map(W, x.device, x.dtype).detach()
        
        # 将初始位移 (u0) 和初始速度 (v0) 转换到频域
        x_u0 = self.dct_3d(x,  cos_T, cos_H, cos_W) 
        x_v0 = self.dct_3d(v0, cos_T, cos_H, cos_W) 
        
        if freq_embed is not None:
            tau = self.to_k(freq_embed.unsqueeze(0).expand(B,-1,-1,-1,-1))
        else:
            tau = torch.ones((B, T_dim, H, W, self.hidden_dim), device=x.device, dtype=x.dtype)
        
        tau = tau.permute(0, 4, 1, 2, 3).contiguous() 
        
        # ---------------------------------------------------------------------
        # # 【物理公式严格对齐区】
        # # 1. 提取各个维度的空间频率: i/W, j/H, l/T (对应公式中的 mu/v, rho/v, lambda/v)
        # freq_w = torch.arange(W, device=x.device, dtype=x.dtype) * (math.pi / W)
        # freq_h = torch.arange(H, device=x.device, dtype=x.dtype) * (math.pi / H)
        # freq_t = torch.arange(T_dim, device=x.device, dtype=x.dtype) * (math.pi / T_dim)

        # # 构建3D频率网格 (对应形状: 1, 1, T, H, W)
        # grid_t = freq_t.view(1, 1, T_dim, 1, 1)
        # grid_h = freq_h.view(1, 1, 1, H, 1)
        # grid_w = freq_w.view(1, 1, 1, 1, W)

        # # 2. 计算波数平方: k^2 = v^2 * (mu^2 + rho^2 + lambda^2)
        # spatial_freq_sq = grid_t**2 + grid_h**2 + grid_w**2
        # k_sq = (self.v ** 2) * spatial_freq_sq
        
        # # 3. 计算角频率 omega: sqrt(k^2 - (alpha/2)^2)
        # # 公式: \frac{\sqrt{4k^2 - \alpha^2}}{2}
        # alpha_half_sq = (self.alpha / 2) ** 2
        # # 使用 relu 防止过阻尼态产生负数导致 sqrt 报 NaN，加 1e-8 维持数值稳定性
        # omega = torch.sqrt(F.relu(k_sq - alpha_half_sq) + 1e-8)
        
        # # 计算相位: omega * t
        # omega_t = omega * tau
        
        # # 4. 阻尼衰减项: e^{-\frac{\alpha}{2}t}
        # damping = torch.exp(- (self.alpha / 2) * tau)
        
        # # 5. 时间演化项: 
        # # T(t) = e^{-\alpha t / 2} [ A cos(omega t) + B sin(omega t) ]
        # # 其中 A = u0, B = (v0 + alpha*u0/2) / omega
        # cos_term = torch.cos(omega_t)
        # sin_term = torch.sin(omega_t) / omega
        
        # wave_term     = cos_term * x_u0
        # velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        
        # # 频域下经过时间 tau 后的最终频谱
        # self.final_spectrum = damping * (wave_term + velocity_term) # 保存以供外部测试验证


        c_tau = self.c * tau                              # ω_d·t，形状 [B,hidden_dim,T,H,W]
        damping = torch.exp(-self.alpha / 2 * tau)        # e^{-α/2·τ}
        cos_term = torch.cos(c_tau)
        sin_term = torch.sin(c_tau) / (self.c + 1e-8)
        
        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        
        # 频域下经过时间 tau 后的最终频谱
        self.final_spectrum = damping * (wave_term + velocity_term) # [B, hidden_dim, T, H, W]
        # ---------------------------------------------------------------------

        x_final = self.idct_3d(self.final_spectrum, cos_T, cos_H, cos_W)
        
        x_final = x_final.permute(0, 2, 3, 4, 1).contiguous()
        x_final = self.out_norm(x_final)
        x_final = x_final.permute(0, 4, 1, 2, 3).contiguous()
        x_final = x_final * F.silu(z)
        x_final = x_final.permute(0, 2, 3, 4, 1).contiguous()
        x_final = self.out_linear(x_final)
        x_final = x_final.permute(0, 4, 1, 2, 3).contiguous()
        
        return x_final


import torch
import torch.nn as nn

class VideoPatchEmbed3D(nn.Module):
    """
    ViT风格的3D Patch Embedding
    用于将视频(物理空间)干净地切分为 Tokens，输入到频域求解器中
    """
    def __init__(self, in_channels=3, hidden_dim=96, patch_size=(2, 4, 4)):
        super().__init__()
        # 使用无重叠的卷积直接进行 Patch 投影，替代厚重的 CNN Stem
        # patch_size 可以根据你的显存和视频分辨率调整，例如 (时间2帧, 空间4x4)
        self.proj = nn.Conv3d(
            in_channels, 
            hidden_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = self.proj(x)
        
        # LayerNorm 通常在 Channel 维度上做，需要 permute
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x



class WaveVideoClassifier(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=96, num_classes=101):
        super().__init__()

        # 1. 替换为轻量且物理意义明确的 Patch Embedding
        self.stem = VideoPatchEmbed3D(
            in_channels=in_channels, 
            hidden_dim=hidden_dim,
            patch_size=(2, 4, 4) # 你可以根据具体数据集(如Kinetics或UCF101)调节此参数
        )

        # 2. 全局物理演化模块 (你的波动方程 PDE 求解器)
        self.wave_blocks = nn.Sequential(
            Wave3D(dim=hidden_dim, hidden_dim=hidden_dim),
            Wave3D(dim=hidden_dim, hidden_dim=hidden_dim),
            Wave3D(dim=hidden_dim, hidden_dim=hidden_dim),
            Wave3D(dim=hidden_dim, hidden_dim=hidden_dim)
        )

        # 3. 分类头 (保持不变，AdaptiveAvgPool 很好地适配了变长输入)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(), # 建议使用 GELU 替代 ReLU，在现代 Transformer 架构中表现更好
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 输入: (B, 3, T, H, W)
        x = self.stem(x)
        x = self.wave_blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x