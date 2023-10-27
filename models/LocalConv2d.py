import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalConv2d(nn.Module):

    def __init__(self, num_rows, num_feats_in, num_feats_out, kernel=1, padding=0):
        super(LocalConv2d, self).__init__()

        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.pad = padding

        self.group_conv = nn.Conv2d(num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1,
                                    groups=num_rows)

    def forward(self, x):
        b, c, h, w = x.size()

        if self.pad: x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)

        t = int(h / self.num_rows)

        # unfold by rows
        x = x.unfold(2, t + self.pad * 2, t)
        x = x.permute([0, 2, 1, 4, 3]).contiguous()
        x = x.view(b, c * self.num_rows, t + self.pad * 2, (w + self.pad * 2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.group_conv(x)
        y = y.view(b, self.num_rows, self.out_channels, t, w).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(b, self.out_channels, h, w)

        return y
