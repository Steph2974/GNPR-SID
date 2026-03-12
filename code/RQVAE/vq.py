import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm

import torch
import torch.nn as nn


class VQBridge_v1(nn.Module):
    """
    轻量级 VQBridge 模块：用于缓解向量量化(VQ)中的码本崩溃与冲突问题。
    通过在量化匹配前引入码本的全局信息交互，将原本的“稀疏梯度”转化为“密集梯度”。
    参考思路：FVQ (Compress-Process-Recover)
    """
    def __init__(self, dim, hidden_dim=32, num_heads=4):
        """
        参数:
        dim: 原始码本的维度 (在你的 GNPR-SID 中对应 e_dim，通常为 64)
        hidden_dim: 压缩后的隐层交互维度 (建议设为 32，保持轻量)
        num_heads: 多头注意力的头数 (必须能被 hidden_dim 整除)
        """
        super().__init__()
        
        # 1. 压缩层 (Compress): 将原始维度映射到更紧凑的特征空间
        self.compress = nn.Linear(dim, hidden_dim)
        
        # 2. 处理层 (Process): 核心交互层。
        # 使用自注意力机制让所有码字(Code)互相“看”到彼此的位置，感知拥挤度并进行排斥/融合。
        self.process = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2,  # 缩小 FFN 维度以节省显存
            batch_first=True,                # 输入格式设为 (Batch, Seq_len, Feature)
            norm_first=True,                 # 采用 Pre-Norm，提升训练初期的稳定性
            dropout=0.1
        )
        
        # 3. 恢复层 (Recover): 将交互后的特征映射回原始维度
        self.recover = nn.Linear(hidden_dim, dim)
        
        # 归一化层，用于残差连接后稳定分布
        self.norm = nn.LayerNorm(dim)

        # 初始化权重 (建议将新加入的映射层权重初始化得小一点，避免破坏 K-Means 初始化的分布)
        nn.init.normal_(self.compress.weight, std=0.02)
        nn.init.normal_(self.recover.weight, std=0.02)

    def forward(self, codebook):
        """
        前向传播
        输入 codebook 形状: (N_e, dim) 例如 (32, 64)
        输出 refined_codebook 形状: (N_e, dim) 例如 (32, 64)
        """
        # Transformer 期望的输入带有 Batch 维度。在这里，整个码本就是一个 Sequence。
        # 扩展维度: (32, 64) -> (1, 32, 64)
        x = codebook.unsqueeze(0)
        
        # 压缩: (1, 32, 64) -> (1, 32, 32)
        x = self.compress(x)
        
        # 全局信息交互: 此时 32 个码字通过 Self-Attention 交换位置与梯度信息
        x = self.process(x)
        
        # 恢复: (1, 32, 32) -> (1, 32, 64)
        x = self.recover(x)
        
        # 降维: 去掉 Batch 维度 -> (32, 64)
        x = x.squeeze(0)
        
        # 核心设计：残差连接 (Residual Connection)
        # 这保证了模型在初始阶段依然具有强大的 K-Means 聚类特征，随着训练深入，Bridge 提供的位移修正逐渐生效。
        refined_codebook = self.norm(codebook + x)
        
        return refined_codebook

class VQBridge_v2(nn.Module):
    def __init__(self, dim, hidden_dim=32, num_heads=4):
        super().__init__()
        self.compress = nn.Linear(dim, hidden_dim)
        
        self.process = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2,  
            batch_first=True,                
            norm_first=True,                 
            dropout=0.1
        )
        
        self.recover = nn.Linear(hidden_dim, dim)
        
        # 引入一个可学习的缩放因子，让模型自己决定吸收多少交互信息
        self.alpha = nn.Parameter(torch.zeros(1))

        # 【关键修改 1】：只用常规方式初始化 compress
        nn.init.xavier_uniform_(self.compress.weight)
        
        # 【关键修改 2】：将 recover 层严格初始化为 0。
        # 这样在初始状态下，Bridge 的输出完全是 0，保证 K-Means 初始化不被破坏
        nn.init.zeros_(self.recover.weight)
        nn.init.zeros_(self.recover.bias)
        
        # 【关键修改 3】：删除 self.norm

    def forward(self, codebook):
        x = codebook.unsqueeze(0)
        x = self.compress(x)
        x = self.process(x)
        x = self.recover(x).squeeze(0)
        
        # 【关键修改 4】：使用门控残差连接，且去除了 LayerNorm
        refined_codebook = codebook + self.alpha * x
        
        return refined_codebook

class VQBridge(nn.Module):
    def __init__(self, dim, hidden_dim=32, num_heads=4):
        super().__init__()
        self.compress = nn.Linear(dim, hidden_dim)
        
        self.process = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim * 4,  # 加大 FFN
                batch_first=True,
                norm_first=True,
                dropout=0.1
            ),
            num_layers=2   # 加到 2 层
        )
        
        self.recover = nn.Linear(hidden_dim, dim)

        nn.init.normal_(self.compress.weight, std=0.02)
        nn.init.zeros_(self.compress.bias)
        nn.init.normal_(self.recover.weight, std=1e-4)   # 极小 std
        nn.init.zeros_(self.recover.bias)
        self.alpha = nn.Parameter(torch.tensor([0.01]))

        

    def forward(self, codebook):
        x = codebook.unsqueeze(0)
        x = self.compress(x)
        x = self.process(x)
        x = self.recover(x).squeeze(0)
        refined_codebook = codebook + self.alpha * x
        return refined_codebook

        
class VectorQuantizer(nn.Module):

    def __init__(
            self,
            n_e,
            e_dim,
            beta=0.25,
            kmeans_init=False,
            kmeans_iters=10,
            sk_epsilon=0.01,
            sk_iters=100,
            use_linear=0,
            use_sk=False,
            diversity_loss=0.0,
            use_bridge=False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear
        self.use_sk = use_sk
        self.diversity_loss = diversity_loss

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if use_linear == 1:
            self.codebook_projection = torch.nn.Linear(self.e_dim, self.e_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=self.e_dim ** -0.5)
        
        self.use_bridge = use_bridge
        if use_bridge:
            self.vq_bridge = VQBridge(e_dim, hidden_dim=32, num_heads=4)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, epoch_idx):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        # # 添加噪声
        # scale = 0.01 * (1 - epoch_idx / 100)
        # latent = latent + torch.randn_like(latent) * scale

        if not self.initted and self.training:
            self.init_emb(latent)

        if self.use_linear == 1:
            embeddings_weight = self.codebook_projection(self.embedding.weight)
        else:
            embeddings_weight = self.embedding.weight

        # 在计算距离与查表前，用 VQBridge 对码本做全局交互（缓解码本崩溃）
        if self.use_bridge:
            embeddings_weight = self.vq_bridge(embeddings_weight)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(embeddings_weight ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, embeddings_weight.t())
       
        indices = torch.argmin(d, dim=-1)
        x_q = F.embedding(indices, embeddings_weight).view(x.shape)

        if self.use_sk and self.sk_epsilon > 0:
            d_soft = self.center_distance_for_constraint(d)
            d_soft = d_soft.double()
            Q = sinkhorn_algorithm(d_soft, self.sk_epsilon, self.sk_iters)  # [B, N]
        else:
            Q = F.softmax(-d, dim=-1)  # [B, N]

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        if epoch_idx >= 100: # 防止在训练初期就计算 diversity_loss，但是把 1000 改成 100（2025-12-14）    
            if self.diversity_loss > 0:
                soft_counts = Q.sum(0)  # [N]
                mean_soft_count = soft_counts.mean()
                mean_count_loss = torch.mean((soft_counts - mean_soft_count) ** 2) / (mean_soft_count ** 2 + 1e-5)
                # pairwise
                # pairwise_loss = 0
                # for i in range(self.n_e):
                #     codebook_vectors = x_q[indices == i]
                #     if len(codebook_vectors) > 1:
                #         pairwise_distances = torch.cdist(codebook_vectors, codebook_vectors, p=2)
                #         pairwise_loss += pairwise_distances.mean()
                # print(f"mean_count_loss: {mean_count_loss}")
                diversity_loss = (
                    # 0.1 * (pairwise_loss / self.n_e) +
                    0.05 * mean_count_loss
                )
                # print(f"diversity_loss: {diversity_loss}")
                loss = codebook_loss + self.beta * commitment_loss + self.diversity_loss * diversity_loss
            else:
                loss = codebook_loss + self.beta * commitment_loss
        else:
            loss = codebook_loss + self.beta * commitment_loss
        
        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, d
