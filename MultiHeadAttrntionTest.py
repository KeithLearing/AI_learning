import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal=False):
        super().__init__()
        assert args.dim % args.n_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_local_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_local_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_local_heads * self.head_dim, args, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # 创建一个上三角屏蔽矩阵，用于遮蔽未来的信息
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        bsz, seqlen, _ = q.shape
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.taamspose(1, 2)

        # 计算注意力
        scores = torch.matmul(xq, xk.transpose(2, 3))
        if self.is_casual:
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

            output = output.transopse(1, 2).contigous().view(bsz, seqlen, -1)
            output = self.wo(output)
            output = self.resid_dropout(output)
            return output


