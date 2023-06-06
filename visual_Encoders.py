from model_utils.BoundingBox_Utils import PositionEmbeddingSine, NestedTensor
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self, num_feature_levels, vf_dim, hidden_dim):
        super(BaseEncoder, self).__init__()
        self.pos_embed = PositionEmbeddingSine(hidden_dim//2, normalize=True)
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim

        if num_feature_levels > 1:
            input_proj_list = []
            in_channels = vf_dim
            input_proj_list.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            for _ in range(num_feature_levels - 1):
                input_proj_list.append(nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(vf_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, vf, mask, duration):
        # vf: (N, L, C), mask: (N, L),  duration: (N)
        vf = vf.transpose(1, 2)  # (N, L, C) --> (N, C, L)
        vf_nt = NestedTensor(vf, mask, duration)
        pos0 = self.pos_embed(vf_nt)
        srcs = []
        masks = []
        poses = []
        src0, mask0 = vf_nt.decompose()
        srcs.append(self.input_proj[0](src0))
        masks.append(mask0)
        poses.append(pos0)
        assert mask is not None
        for l in range(1, self.num_feature_levels):
            if l == 1:
                src = self.input_proj[l](vf_nt.tensors)
            else:
                src = self.input_proj[l](srcs[-1])
            m = vf_nt.mask
            mask = F.interpolate(m[None].float(), size=src.shape[-1:]).to(torch.bool)[0]
            pos_l = self.pos_embed(NestedTensor(src, mask, duration)).to(src.dtype)
            srcs.append(src)
            masks.append(mask)
            poses.append(pos_l)
        return srcs, masks, poses