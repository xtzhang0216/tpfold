from torch import nn
import torch
from einops import rearrange, repeat
import torch.nn.functional as F
import numpy as np

class AttentionMap(nn.Module):
    def __init__(self, tmbed_dim, embed_dim, num_heads=128, gated=False):
        super().__init__()
        # embed_dim2560为头数×qkv维数
        # assert embed_dim == num_heads * head_width
        head_width = embed_dim/num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width
        self.dimensionup = nn.Linear(tmbed_dim, 2560, bias=False)

        self.proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gated = gated
        if gated:
            self.g_proj = nn.Linear(embed_dim, embed_dim)
            torch.nn.init.zeros_(self.g_proj.weight)
            torch.nn.init.ones_(self.g_proj.bias)

        self.rescale_factor = self.head_width**-0.5

        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, x, mask=None, bias=None, indices=None):
        """
        Basic self attention with optional mask and external pairwise bias.
        To handle sequences of different lengths, use mask.

        Inputs:
          x: batch of input sequneces (.. x L x C)
          mask: batch of boolean masks where 1=valid, 0=padding position (.. x L_k). optional.
          bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads). optional.

        Outputs:
          sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """
        x = self.dimensionup(x)
        t = rearrange(self.proj(x), "... l (h c) -> ... h l c", h=self.num_heads)
        q, k = t.chunk(2, dim=-1)
        # 注意力点乘后除以的根号常数没有显示修改
        q = self.rescale_factor * q
        a = torch.einsum("...qc,...kc->...qk", q, k)


        # Add external attention bias.
        if bias is not None:
            a = a + rearrange(bias, "... lq lk h -> ... h lq lk")

        # Do not attend to padding tokens.
        if mask is not None:
            mask = repeat(
                mask, "... lk -> ... h lq lk", h=self.num_heads, lq=q.shape[-2]
            )
            a = a.masked_fill(mask == False, 0.0)

        # a = F.softmax(a, dim=-1)
        a = torch.einsum("... k q c -> ... q c k", a)
        # y = torch.einsum("...hqk,...hkc->...qhc", a, v)
        # y = rearrange(y, "... h c -> ... (h c)", h=self.num_heads)
        #
        # if self.gated:
        #     y = self.g_proj(x).sigmoid() * y
        # y = self.o_proj(y)

        return a



# tmbed_dim=5;embed_dim=2560;num_heads=128;head_width=embed_dim/num_heads
# b=60;maxlen=38
# att = Attention(tmbed_dim, embed_dim, num_heads, gated=True)
# x=torch.rand(b,maxlen,tmbed_dim)
# mask=torch.zeros(b,maxlen)
# att(x, mask)
