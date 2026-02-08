import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch import nn
import numbers
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base


def DWConv3X3(c, bias):
    return nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, groups=c, bias=bias)


def Conv3X3(c_in, c_out, bias):
    return nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=bias)


def Conv1X1(c_in, c_out, bias):
    return nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, stride=1, padding=0, bias=bias)


def mask_window_partition(x, window_size):
    B, C, H, W = x.shape
    #  b c wz wh wz ww
    #  0 1 2  3  4  5
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    #  b c wz wh wz ww   ------> b wz wz c wh ww ------> b*wz**2  c wh ww
    #                            0  2 4  1  3  5
    windows = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, window_size, window_size).contiguous()
    return windows


def winP(x, window_size, num_head, num_dim):
    B, C, H, W = x.shape
    #  b 3 nh nd wh wz ww wz  ----> 3 b wh ww nh wz wz nd ---> 3 br nh ws nd
    #  0 1 2  3  4  5  6  7  ----> 1 0 4  6  2  5  7  3
    x = x.view(B, 3, num_head, num_dim, H // window_size, window_size, W // window_size, window_size)
    windows = (x.permute(1, 0, 4, 6, 2, 5, 7, 3).reshape(3, B, -1, num_head, window_size, window_size, num_dim)
               .flatten(-3, -2))
    return windows


def winR(windows, window_size, H, W, num_head, head_dim):
    # input: b rn nh ws nc  Or br nh ws nc
    # b wh ww nh wz wz nc
    # 0  1  2  3  4  5  6
    # b nh nc wh wz ww wz ----> 0 3 6 1 4 2 5
    x = windows.view(-1, H // window_size, W // window_size, num_head, window_size, window_size, head_dim)
    x = x.permute(0, 3, 6, 1, 4, 2, 5).reshape(-1, num_head * head_dim, H, W).contiguous()
    return x


class IRBlock(nn.Module):
    def __init__(self, dim, train_size=256, win_size=8, shift_size=0, num_head=1, bias=True):
        super(IRBlock, self).__init__()

        self.dim = dim
        self.win_size = win_size
        self.num_head = num_head
        self.shift_size = shift_size
        self.train_size = train_size
        self.bias = bias

        self.norm = LayerNorm2d(self.dim)
        self.conv1 = Conv1X1(self.dim, self.dim * 3, self.bias)
        self.swf = SwinTransFormer(self.dim // 2, self.train_size, self.win_size,
                                   self.shift_size, self.num_head, self.bias)

        if shift_size > 0:
            self.rwf = RwinFormer(self.dim // 2, self.train_size, self.win_size, 'C',
                                  self.num_head, self.bias)
        else:
            self.rwf = RwinFormer(self.dim // 2, self.train_size, self.win_size, 'W',
                                  self.num_head, self.bias)

        self.conv2 = Conv1X1(self.dim, self.dim, self.bias)

    def check_feature_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.win_size - h % self.win_size) % self.win_size
        mod_pad_w = (self.win_size - w % self.win_size) % self.win_size
        x = F.pad(x, (0, int(mod_pad_w), 0, int(mod_pad_h)), 'reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.check_feature_size(x)

        # norm
        x = self.norm(x)
        # linear
        x = self.conv1(x)
        # split
        x1, x2 = x.chunk(2, dim=1)
        # short-range attention
        x1 = self.swf(x1)
        # middle-range attention
        x2 = self.rwf(x2)
        # concat
        x = torch.cat([x1, x2], dim=1)
        # linear
        x = self.conv2(x)

        return x[:, :, :H, :W]


class SwinTransFormer(nn.Module):
    def __init__(self, dim, train_size=256, win_size=8, shift_size=0, num_head=1, bias=True):
        super().__init__()

        # pre-define
        self.dim = dim
        self.win_size = win_size
        self.num_head = num_head
        self.head_dim = dim // self.num_head
        self.win_seq = win_size ** 2
        self.shift_size = shift_size
        self.scale = self.head_dim ** -0.5
        self.bias = bias
        self.train_size = train_size

        if self.shift_size > 0:
            attn_mask = self.get_shift_mask(self.train_size, self.train_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        # relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.win_size - 1) * (2 * self.win_size - 1), self.num_head))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)
        relative_position_index = self.get_relative_index()  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_shift_mask(self, H, W):
        # attention mask for SW-MSA
        shift_mask = torch.zeros((1, 1, H, W))
        h_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                shift_mask[:, :, h, w] = cnt
                cnt += 1
        #  1 1 H W ---> r^2 1 wh ww
        shift_mask_windows = mask_window_partition(shift_mask, self.win_size)
        # r^2 wN
        shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)
        # r^2 1 wN - r^2 wN 1 ---> r^2 wN wN
        shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)
        shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
            shift_attn_mask == 0, float(0.0))
        return shift_attn_mask

    def get_relative_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size)
        coords_w = torch.arange(self.win_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size - 1
        relative_coords[:, :, 0] *= 2 * self.win_size - 1
        return relative_coords.sum(-1)

    def forward(self, x):
        B, _, H, W = x.shape

        # main entry
        if self.shift_size > 0:
            # r^2 wN wN
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            x = x

        qkv = (winP(x, self.win_size, self.num_head, self.head_dim)
               .flatten(1, 2).contiguous())  # b c h w ---> 3 b rn nh ws nc
        q, k, v = qkv[0], qkv[1], qkv[2]

        #  br nh wN c  @ br nh c wN ---> br nh wN wN
        att = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size * self.win_size, self.win_size * self.win_size, -1)  # Wh*Ww,Wh*Ww nh
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nh Wh*Ww, Wh*Ww

        # br nh wN wN * nh 1 1
        att = att * self.scale + relative_position_bias.unsqueeze(0)

        if self.shift_size > 0:
            if H == self.train_size and W == self.train_size:
                mask = self.attn_mask
            else:
                mask = self.get_shift_mask(H, W).to(x.device)
            # r^2 wN wN
            nW = mask.shape[0]
            # r^2 1 wN wN ---> 1 r^2 1 wN wN  ---> 1 1 r^2 1 wN wN
            att = att.view(B, nW, self.num_head, self.win_seq, self.win_seq) + mask.unsqueeze(1).unsqueeze(0)
            att = att.view(-1, self.num_head, self.win_seq, self.win_seq)
        else:
            att = att
        att = F.softmax(att, dim=-1)  # br nh wN wN'
        # br nh wN wN' @ br^2 nh wN c ---> br^2 nh wN c
        att_feat = att @ v
        x = winR(att_feat, self.win_size, H, W, self.num_head, self.head_dim)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = x
        return x


class RwinFormer(nn.Module):
    def __init__(self, dim, train_size=256, win_size=8, category='W', num_head=1, bias=True):
        super(RwinFormer, self).__init__()

        self.dim = dim
        self.win_size = win_size
        self.category = category
        self.num_head = num_head
        self.bias = bias
        self.train_size = train_size
        self.head_dim = self.dim // self.num_head
        self.win_seq = self.win_size ** 2
        self.scale = self.head_dim ** -0.5

        # if self.category == 'C' and self.train_size // self.win_size <= 2:  # special
        #     self.category = 'W'

        # middle-range attention
        if self.category == 'W':
            self.pad = nn.ReflectionPad2d(3 // 2)
            self.pos_embedding = (
                nn.Parameter(torch.zeros(9, self.num_head, (2 * self.win_size - 1) * (2 * self.win_size - 1))))
        elif self.category == 'C':
            self.pad_size = 2
            self.pad = nn.ReflectionPad2d(self.pad_size)
            self.pos_embedding = nn.Parameter(
                torch.zeros(self.pad_size * 4 + 1, self.num_head,
                            (2 * self.win_size - 1) * (2 * self.win_size - 1)))
        else:
            assert "the routing window category need in [W,C]!"
        trunc_normal_(self.pos_embedding, std=.02)

        pos_index = self.get_relative_index()  # Wh*Ww, Wh*Ww
        self.register_buffer("pos_index", pos_index)

        rh = self.train_size // self.win_size
        rw = self.train_size // self.win_size
        table = torch.arange(rh * rw)
        tpd = self.get_table(rh, rw, table)
        self.register_buffer("tpd", tpd)

    def get_relative_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size)
        coords_w = torch.arange(self.win_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size - 1
        relative_coords[:, :, 0] *= 2 * self.win_size - 1
        return relative_coords.sum(-1)

    def get_vector(self, q, k):
        # generate the super-window vector (simple)
        # qk: B C rh rw
        # Q processing
        qr = q.detach()
        # b c h w ---> b c rh rw
        qr = F.avg_pool2d(qr, self.win_size, stride=self.win_size)
        # qr = F.max_pool2d(qr, self.win_size, stride=self.win_size)

        # # K processing
        kr = k.detach()
        kr = F.avg_pool2d(kr, self.win_size, stride=self.win_size)
        # kr = F.max_pool2d(kr, self.win_size, stride=self.win_size)

        return qr, kr

    def get_table(self, rh, rw, table):
        # Create index mapping table with padding
        # rh rw ---》 1 1 rh rw
        table = table.reshape(1, 1, rh, rw)
        table = self.pad(table)
        if self.category == 'C':
            # 1 1 rh rw+2 ---> 1 1 rh rw 5 ---> 1 1 rn 5
            table_H = table[..., self.pad_size:-self.pad_size, :]
            # 1 1 rh+2 rw ---> 1 1 rh rw 5 ---> 1 1 rn 5
            table_V = table[..., self.pad_size:-self.pad_size]
            table_unfold_H = table_H.unfold(3, self.pad_size * 2 + 1, 1).flatten(2, 3)
            table_unfold_V = table_V.unfold(2, self.pad_size * 2 + 1, 1).flatten(2, 3)
            # 1 1 rn (2+2+2+2)
            # 1 rn 1 10
            table_unfold = torch.cat([table_unfold_H,
                                      table_unfold_V[..., :self.pad_size],
                                      table_unfold_V[..., self.pad_size + 1:]], dim=-1)
            table_unfold = table_unfold.permute(0, 2, 1, 3).contiguous()

        else:
            # 1 1 rh rw ----> 1 1*9 rn ----> 1 rn 1*9 ----> 1 rn 1 9
            table_unfold = F.unfold(table.float(),
                                    kernel_size=3).long().permute(0, 2, 1).contiguous().reshape(1, rh * rw, 1, 9)

        return table_unfold

    def get_unfold(self, kr, B, C, rh, rw):
        kr = self.pad(kr)
        if self.category == 'C':
            # sampling
            # b c rh rw+2
            kr_H = kr[..., self.pad_size:-self.pad_size, :]
            # b c rh+2 rw
            kr_V = kr[..., self.pad_size:-self.pad_size]
            # b c rh rw 5 ----> b c rn 8  ----> b rn c 8
            kr_fold_H = kr_H.unfold(3, self.pad_size * 2 + 1, 1).flatten(2, 3)
            kr_fold_V = kr_V.unfold(2, self.pad_size * 2 + 1, 1).flatten(2, 3)
            kr_fold = torch.cat([kr_fold_H, kr_fold_V[..., :self.pad_size], kr_fold_V[..., self.pad_size + 1:]], dim=-1)
            kr_fold = kr_fold.permute(0, 2, 1, 3).contiguous()
        else:
            # sampling
            # 1 1 rh+1 rw+1
            # b c*9 nr  ----> b c 9 nr  default: 3X3 neighborhood
            kr_fold = F.unfold(kr, kernel_size=3).reshape(B, C, 9, rh * rw)
            # b c 9 rn ---> b rn c 9
            kr_fold = kr_fold.permute(0, 3, 1, 2).contiguous()  # Adjust dimensions

        return kr_fold

    def router(self, q, k):

        B, C, H, W = q.shape
        rh = H // self.win_size
        rw = W // self.win_size
        qr, kr = self.get_vector(q, k)
        # b c rh rw ----> b c rn ----> b rn c ---> b rn 1 c
        qr = qr.flatten(2).permute(0, 2, 1).contiguous().unsqueeze(-2)

        # win----> b c rh+2 rw+2
        # H or V ----> b c rh+2 rw+2
        kr_fold = self.get_unfold(kr, B, C, rh, rw)

        sp_size = self.train_size // self.win_size

        if rh == sp_size and rw == sp_size:
            table_unfold = self.tpd
        else:
            table = torch.arange(rh * rw, device=qr.device)
            table_unfold = self.get_table(rh, rw, table)

        # qr = F.normalize(qr, dim=-1)
        # kr_fold = F.normalize(kr_fold, dim=-2)
        # Compute similarity scores
        # b nr 1 c @ b nr c 8 ----> b nr 1 8
        a_r = qr @ kr_fold

        # mask step(directly operation)
        if self.category == 'C':
            a_r[..., self.pad_size] -= 100.0
        else:
            a_r[..., 4] -= 100.0

        a_r = F.softmax(a_r, dim=-1)

        # most relative region selection
        # b rn 1 1
        _, idx = torch.max(a_r, dim=-1, keepdim=True)

        # Map idx to actual indices
        # table: B rn 1 9  idx: B rn 1 1
        idx_reflect = table_unfold.expand(B, -1, -1, -1).gather(-1, idx)

        # 获取所需的 pos_bias，形状为 (9,nh, win_seq, 2*win_seq) ---> 1 rn 9 nh 2wz-1*2wz-1
        # 8 nh 2ws-1*2ws-1 ---> 1 1 8 nh 2ws-1*2ws-1 ----> 1 rn 8 nh 2ws-1*2ws-1
        pos_bias = self.pos_embedding.unsqueeze(0).expand(B, rh * rw, -1, -1, -1)

        if self.category == 'C':
            itself_pos_bias = pos_bias[:, :, self.pad_size, ...]
        else:
            itself_pos_bias = pos_bias[:, :, 4, ...]

        # idx: b rn 1 1 ---> b rn 1 1 1 ---> b rn 1 nh 2wz-1*2wz-1
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, self.num_head, (2 * self.win_size - 1) ** 2)
        # pos_bias: b rn 9 nh winseq 2*winseq ---> b rn 1 nh winseq winseq ---> b rn nh ws ws
        dir_pos_bias = pos_bias.gather(2, idx).squeeze(-4)

        itself_pos_bias = (itself_pos_bias[..., self.pos_index.view(-1)]
                           .view(B, rh * rw, self.num_head, self.win_seq, self.win_seq))

        dir_pos_bias = (dir_pos_bias[..., self.pos_index.view(-1)]
                        .view(B, rh * rw, self.num_head, self.win_seq, self.win_seq))

        # Return the valid indices
        # B rn nh ws 2*ws or b rn nh ws ws
        return idx_reflect, torch.cat([itself_pos_bias, dir_pos_bias], dim=-1)
        # return idx_reflect, dir_pos_bias

    def forward(self, x):

        # regional similarity
        qr, kr, _ = x.chunk(3, dim=1)
        B, C, H, W = qr.shape
        # index & pos_embed
        idx, pos_embed = self.router(qr, kr)

        # split channel
        # b c h w ---> 3 b rn nh ws nd
        qkv = winP(x, self.win_size, self.num_head, self.head_dim).contiguous()
        # b rn nh ws nc
        q, k, v = qkv[0], qkv[1], qkv[2]

        # idx: b rn 1 1 --> b rn 1 1 1 ---> b rn nh wseq nc
        # b rn nh ws nc ----> b rn nh ws nc
        k_direct = torch.gather(k, 1, idx.unsqueeze(-1).expand(-1, -1, self.num_head, self.win_seq, self.head_dim))
        v_direct = torch.gather(v, 1, idx.unsqueeze(-1).expand(-1, -1, self.num_head, self.win_seq, self.head_dim))
        # b rn nh 2*ws nc
        k_g = torch.cat([k, k_direct], dim=3)
        v_g = torch.cat([v, v_direct], dim=3)

        # prob b nh nr 1 1
        # b rn nh ws nc @ b rn nh nc 2*ws -----> b rn nh ws 2*ws
        attn = q @ k_g.transpose(-1, -2)
        # b rn nh ws 2*ws
        attn = attn * self.scale + pos_embed
        attn = F.softmax(attn, dim=-1)
        # b rn nh ws (2*ws)' @ b rn nh 2*ws nc  ----> b rn nh ws nc
        out = attn @ v_g
        out = winR(out, self.win_size, H, W, self.num_head, self.head_dim)
        return out


class FFN(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()

        self.dim = dim

        self.norm = LayerNorm2d(self.dim)
        self.conv1 = Conv1X1(self.dim, 2 * self.dim, bias)
        self.dwconv = DWConv3X3(self.dim, bias)
        self.cmlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv1X1(self.dim, self.dim, bias),
        )
        self.conv2 = Conv1X1(self.dim, self.dim, bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x = F.gelu(x1) * x2
        x = self.cmlp(x) * x
        x = self.conv2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, win_size=8, shift_size=0, num_head=1, train_size=256, bias=True):
        super().__init__()

        self.dim = dim
        self.win_size = win_size
        self.shift_size = shift_size
        self.num_head = num_head
        self.train_size = train_size
        self.bias = bias
        self.irb = IRBlock(dim, self.train_size, self.win_size, self.shift_size,
                           self.num_head, self.bias)
        self.ffn = FFN(dim, self.bias)

        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        B, C, H, W = x.shape
        # main arch
        x = self.irb(x)
        x = x * self.gamma + inp

        # 线性变换部分 ffn
        y = self.ffn(x)
        return y * self.beta + x


class Encoder(nn.Module):
    def __init__(self, dim, nums, num_heads, win_size, shift_size, train_size,
                 bias):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.mid_enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.cut = len(nums) - 1
        self.train_size = train_size
        dim_ = dim
        for idx, (num, head) in enumerate(zip(nums, num_heads)):
            self.encoders.append(
                nn.Sequential(
                    *[Block(dim_, win_size, 0 if i % 2 == 0 else shift_size, head, self.train_size, bias)
                      for i in range(num)]
                )
            )
            self.train_size //= 2
            if idx != self.cut:
                self.downs.append(
                    nn.Conv2d(dim_, 2 * dim_, kernel_size=2, stride=2)
                )
            dim_ *= 2

    def forward(self, inp):
        encs = []
        x = inp
        for idx in range(len(self.encoders)):
            x = self.encoders[idx](x)
            encs.append(x)
            if idx != self.cut:
                x = self.downs[idx](x)
        return encs


class Decoder(nn.Module):
    def __init__(self, dim, img_ch, nums, num_heads, win_size, shift_size, train_size, bias):
        super().__init__()

        self.length = len(nums)
        self.cut = self.length - 1
        self.train_size = train_size // (2 ** self.cut)
        dim_ = dim * (2 ** self.cut)
        nums = list(reversed(nums))
        num_heads = list(reversed(num_heads))

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.DSAMs = nn.ModuleList(
            nn.Conv2d(in_channels=dim * 2 ** i, out_channels=img_ch, kernel_size=3, padding=1,
                      stride=1, bias=bias) for i in range(self.length)
        )
        for idx, (num, head) in enumerate(zip(nums, num_heads)):
            if idx != 0:
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(dim_, 2 * dim_, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )
                dim_ //= 2
            self.decoders.append(
                nn.Sequential(
                    *[Block(dim_, win_size, 0 if i % 2 == 0 else shift_size, head, self.train_size, bias)
                      for i in range(num)]
                )
            )
            self.train_size *= 2

    def forward(self, encs):
        x = encs[-1]
        decs = []
        differs = []
        for idx in range(len(self.decoders)):
            x = self.decoders[idx](x)
            differ = self.DSAMs[-1 - idx](x)
            differs.append(differ)
            decs.append(x)
            if idx != self.cut:
                x = self.ups[idx](x) + encs[-2 - idx]
        return decs, differs


class RouteWinFormer(nn.Module):
    def __init__(self, img_ch=3, dim=32, win_size=8, enc_nums=[2, 2, 2, 2], dec_nums=[2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8], img_size=256, bias=False):
        super().__init__()

        shift_size = win_size // 2
        self.expand = nn.Conv2d(in_channels=img_ch, out_channels=dim, kernel_size=3, padding=1, stride=1,
                                bias=bias)

        self.length = len(enc_nums)
        self.encoders = Encoder(dim, enc_nums, num_heads, win_size, shift_size, img_size, bias)
        self.decoders = Decoder(dim, img_ch, dec_nums, num_heads, win_size, shift_size, img_size,
                                bias)

        self.padder_size = 2 ** (len(enc_nums))

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
        
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, inp):
        B, C, H, W = inp.shape
        x = self.check_image_size(inp)
        feat = self.expand(x)

        # network
        encs = self.encoders(feat)
        decs, differs = self.decoders(encs)
        out = differs[-1][:, :, :H, :W] + inp


        return [differs[:-1], out]


class RouteWinFormerLocal(Local_Base, RouteWinFormer):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, factor_h=1.5, factor_w=1.5, **kwargs):
        Local_Base.__init__(self)
        RouteWinFormer.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * factor_h), int(W * factor_w))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    enc_nums = [2, 4, 4, 8]
    dec_nums = [2, 4, 4, 8]
    num_heads = [1, 2, 4, 8]

    net = RouteWinFormerLocal(img_ch=img_channel, dim=width, enc_nums=enc_nums,
                              dec_nums=dec_nums, num_heads=num_heads)

    inp_shape = (3, 256, 256)

    num_parameters = sum(map(lambda x: x.numel(), net.parameters()))
    print('{} : {:<.4f} [M]'.format('#Params', num_parameters / 10 ** 6))
