import torch
import torch.nn as nn
import torch.nn.functional as F

#selective scale Res-Unet 2: change concat to summation
#from squeeze_and_excitation import ChannelSELayer
BN_EPS = 1e-4
MAX_NUM_FEATURES = 384

def norm(planes, num_channels_per_group=16):
    assert(planes >= num_channels_per_group)
    if num_channels_per_group > 0:
        num_groups = int(planes / num_channels_per_group)
        return nn.GroupNorm(num_groups = num_groups, num_channels=planes, affine=True)
    else:
        return nn.BatchNorm3d(planes, eps=BN_EPS)

class SelectiveScaleLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, num_scales, reduction_ratio=2, group_norm=0):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SelectiveScaleLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.global_pooling = nn.AdaptiveAvgPool3d(1)

        self.conv_down = nn.Sequential(
            nn.Conv3d(num_channels, num_channels_reduced, kernel_size=1, bias=False),
            #norm(num_channels_reduced, num_channels_per_group=group_norm), # not sure if it's needed
            # nn.ReLU(inplace=True)
            nn.ReLU(inplace=False)
        )
        # self.conv_up = nn.Conv3d(num_channels_reduced, num_channels, kernel_size=1, bias=False)
        self.conv_up = nn.ModuleList([])
        # self.num_scales = num_scales
        for i in range(num_scales):
            self.conv_up.append(
                nn.Conv3d(num_channels_reduced, num_channels, kernel_size=1, bias=False)
            )

        # self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        # self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        # self.relu = nn.ELU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: X, shape = (batch_size, num_scale, num_channels, X, Y, Z)
        :return: output tensor = (batch_size, num_channels, X, Y, Z)
        """

        # Average along each channel
        out = torch.sum(x, dim=1)
        # out = out.mean(-1).mean(-1).mean(-1) # dim = (batch_size x num_channels)
        out = self.global_pooling(out) # batch_size x num_channels x 1 x 1 x 1

        out = self.conv_down(out) # (batch_size x num_channels_reduced x 1 x 1 x 1)
        for i, conv_up in enumerate(self.conv_up):
            # vector = conv_up(out).unsqueeze_(dim=1)
            vector = torch.unsqueeze(conv_up(out), dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        out = self.softmax(attention_vectors) # bz x num_scale x num_channels x 1 x 1 x 1
        out = torch.mul(x, out).sum(dim=1)
        return out

# my implementation of ChannelSELayer
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.conv_down = nn.Conv3d(num_channels, num_channels_reduced, kernel_size=1, bias=False)
        self.conv_up = nn.Conv3d(num_channels_reduced, num_channels, kernel_size=1, bias=False)
        # self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        # self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        # self.relu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """

        # Average along each channel
        out = self.global_pooling(input_tensor)

        # channel excitation
        out = self.relu(self.conv_down(out))
        out = self.sigmoid(self.conv_up(out))
        out = torch.mul(input_tensor, out)

        return out

# relu2 = nn.ReLU(inplace=True)

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBnRelu3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, dilation=1, groups=1, padding=1, group_norm=0):
        super(ConvBnRelu3d, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              dilation=dilation, groups=groups, padding=padding, bias=False)
        self.bn = norm(out_channels, num_channels_per_group=group_norm)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock3d(nn.Module):
    def __init__(self, inplanes, planes, kernal_size=3, padding=1, stride=1, downsample=None, group_norm=0):#, dropout_p=0.0):
        super(ResidualBlock3d, self).__init__()

        self.cbr1 = ConvBnRelu3d(in_channels=inplanes, out_channels=planes, kernel_size=kernal_size, padding=padding,
                                 stride=stride, group_norm=group_norm)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm(planes, num_channels_per_group=group_norm)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        if inplanes != planes :
            self.downsample = nn.Sequential(
                conv1x1x1(inplanes, planes, stride),
                norm(planes, num_channels_per_group=group_norm)
            )
        # self.stride = stride
        self.se = ChannelSELayer(num_channels=planes, reduction_ratio=4)
        # if dropout_p == 0:
        #     self.dropout = None
        # else:
        #     self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        identity = x
        out = self.cbr1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        # print identity.size(), out.size()
        # assert(0)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = out.clone() + identity
        out = self.relu(out)

        return out

class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, padding=1, stride=(2,2,2), group_norm=0):
        super(StackEncoder, self).__init__()
        # padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ResidualBlock3d(x_channels, y_channels, kernal_size=kernel_size, padding=padding, stride=stride,
                            group_norm=group_norm),
            ResidualBlock3d(y_channels, y_channels, stride=1, group_norm=group_norm)
        )

    def forward(self, x):
        y = self.encode(x)
        return y

class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=(2,2,2), stride=(2,2,2), up_mode='upconv', group_norm=0):
        super(StackDecoder, self).__init__()
        # padding = (kernel_size - 1) // 2
        if up_mode == 'upsample':
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(x_big_channels, x_channels, kernel_size=kernel_size, stride=stride),
                norm(x_channels, num_channels_per_group=group_norm),
                nn.ReLU()
                #nn.BatchNorm2d(x_big_channels)
            )
        self.decode = ResidualBlock3d(x_channels, y_channels, stride=1, downsample=None, group_norm=group_norm)

    def forward(self, x_big, x):
        # N, C, H, W = x_big.size()
        # y = F.upsample(x, size=(H, W), mode='bilinear')
        y = self.up(x_big)
        # y += x
        y = y.clone() + x
        y = self.decode(y)
        return y


class ChannelAdjustment(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm=0):
        super(ChannelAdjustment, self).__init__()
        self.ca = nn.Sequential(
                conv1x1x1(in_planes=in_channels, out_planes=out_channels, stride=1),
                norm(out_channels, num_channels_per_group=group_norm),
                nn.ReLU(inplace=False)
                # nn.ReLU(inplace=True)
            )
    def forward(self, x):
        y = self.ca(x)
        return y   

class Net(nn.Module):
    def __init__(self, in_channels=1, nclasses=1, nf=16, relu='relu',
                 up_mode='upconv', group_norm=0, dropout_p=0, depth=3, padding=True, deep_supervision=False):
        super(Net, self).__init__()
        '''
        if relu == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif relu == 'relu6':
            self.relu = nn.ReLU6(inplace=True)
        elif relu == 'elu':
            self.relu = nn.ELU(inplace=True)
        elif relu == 'leakyrelu':
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('{} is not valid'.format(relu))
        '''
        # self.relu = nn.ReLU(inplace=True)

        self.input_block = ResidualBlock3d(inplanes=in_channels, planes=nf, kernal_size=3, padding=1, stride=1,
                                           group_norm=group_norm) # nf x N x N
        #self.fc00 = nn.Identity() # conv3x3x3(nf, nf)
        self.fc01 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),
            ConvBnRelu3d(in_channels=nf, out_channels=2*nf, group_norm=group_norm)
            #conv3x3x3(nf, 2*nf)
        )# nf x N/2 x N/2
        self.fc02 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(4,4,4), stride=(4,4,4)),
            ConvBnRelu3d(in_channels=nf, out_channels=4*nf, group_norm=group_norm)
            #conv3x3x3(nf, 4*nf)
        )# nf x N/4 x N/4
        self.fc03 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(8,8,8), stride=(8,8,8)),
            ConvBnRelu3d(in_channels=nf, out_channels=8*nf, group_norm=group_norm)
            #conv3x3x3(nf, 8*nf)
        ) # nf x N/8 x N/8
        self.ss0 = SelectiveScaleLayer(num_channels=nf, num_scales=4, reduction_ratio=4, group_norm=group_norm)
        # self.se0  = ChannelSELayer(num_channels=nf, reduction_ratio=4)

        self.resdown1 = StackEncoder(x_channels=nf,   y_channels=2*nf, stride=(2,2,2), group_norm=group_norm) # 2*nf X N/2 X N/2
        self.fc10 = nn.Sequential(
            #conv3x3x3(2 * nf, nf),
            ConvBnRelu3d(in_channels=2*nf, out_channels=nf, group_norm=group_norm),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        #self.fc11 = nn.Identity() #conv3x3x3(2*nf, nf)               # nf x N/2 x N/2
        self.fc12 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),
            ConvBnRelu3d(in_channels=2*nf, out_channels=4*nf, group_norm=group_norm)
            #conv3x3x3(2*nf, 4*nf)
        )# nf x N/4 x N/4
        self.fc13 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(4,4,4), stride=(4,4,4)),
            ConvBnRelu3d(in_channels=2*nf, out_channels=8*nf, group_norm=group_norm)
            #conv3x3x3(2*nf, 8*nf)
        )
        self.ss1 = SelectiveScaleLayer(num_channels=2*nf, num_scales=4, reduction_ratio=4, group_norm=group_norm)
        # self.se1  = ChannelSELayer(num_channels=2*nf, reduction_ratio=4)
        
        self.resdown2 = StackEncoder(x_channels=2*nf, y_channels=4*nf, stride=(2,2,2), group_norm=group_norm) # 4*nf X N/4 X N/4
        self.fc20 = nn.Sequential(
            #conv3x3x3(4 * nf, nf),
            ConvBnRelu3d(in_channels=4*nf, out_channels=nf, group_norm=group_norm),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        )
        self.fc21 = nn.Sequential(
            #conv3x3x3(4 * nf, 2*nf),
            ConvBnRelu3d(in_channels=4*nf, out_channels=2*nf, group_norm=group_norm),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )
        #self.fc22 = nn.Identity() #conv3x3x3(4*nf, nf)
        self.fc23 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),
            ConvBnRelu3d(in_channels=4*nf, out_channels=8*nf, group_norm=group_norm)
            #conv3x3x3(4*nf, 8*nf)
        )
        self.ss2 = SelectiveScaleLayer(num_channels=4*nf, num_scales=4, reduction_ratio=4, group_norm=group_norm)
        # self.se2  = ChannelSELayer(num_channels=4*nf, reduction_ratio=4)

        self.resdown3 = StackEncoder(x_channels=4*nf, y_channels=8*nf, stride=(2,2,2), group_norm=group_norm) # 8*nf X N/8 X N/8
        self.fc30 = nn.Sequential(
            #conv3x3x3(8 * nf, nf),
            ConvBnRelu3d(in_channels=8*nf, out_channels=nf, group_norm=group_norm),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True),
        )
        self.fc31 = nn.Sequential(
            #conv3x3x3(8 * nf, 2 * nf),
            ConvBnRelu3d(in_channels=8*nf, out_channels=2*nf, group_norm=group_norm),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
        )
        self.fc32 = nn.Sequential(
            #conv3x3x3(8 * nf, 4 * nf),
            ConvBnRelu3d(in_channels=8*nf, out_channels=4*nf, group_norm=group_norm),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )
        # self.fc33 = conv3x3x3(in_planes=8*nf, out_planes=nf, stride=1)
        self.ss3 = SelectiveScaleLayer(num_channels=8*nf, num_scales=4, reduction_ratio=4, group_norm=group_norm)
        # self.se3  = ChannelSELayer(num_channels=8*nf, reduction_ratio=4)
        max_nf = min(MAX_NUM_FEATURES, 16 * nf)
        self.center_block = ResidualBlock3d(inplanes=8*nf, planes=max_nf, stride=2) # 16*nf X N/16 X N/16

        self.resup3 = StackDecoder(x_big_channels=max_nf, x_channels=8*nf, y_channels=8*nf, up_mode=up_mode, group_norm=group_norm)  # 4*nf X N/4 X N/4
        self.resup2 = StackDecoder(x_big_channels=8*nf,  x_channels=4*nf, y_channels=4*nf, up_mode=up_mode, group_norm=group_norm) # 4*nf X N/4 X N/4
        self.resup1 = StackDecoder(x_big_channels=4*nf,  x_channels=2*nf, y_channels=2*nf, up_mode=up_mode, group_norm=group_norm) # 2*nf X N/2 X N/2
        self.resup0 = StackDecoder(x_big_channels=2*nf, x_channels=nf,   y_channels=nf,   up_mode=up_mode, group_norm=group_norm) # nf X N X N

        if dropout_p == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout3d(p=dropout_p)

        self.last = conv1x1x1(in_planes=nf, out_planes=nclasses, stride=1)

        if deep_supervision:
            self.dp3 = nn.Sequential(
                conv1x1x1(8 * nf, nclasses),
                nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=True)
                # conv1x1x1(4 * nf, nclasses)
                # self.last
            )
            self.dp2 = nn.Sequential(
                conv1x1x1(4 * nf, nclasses),
                nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=True)
                # conv1x1x1(2 * nf, nclasses)
                # self.last
            )
            self.dp1 = nn.Sequential(
                conv1x1x1(2 * nf, nclasses),
                nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
                # conv1x1x1(nf, nclasses)
                # self.last
            )
        else:
            self.dp3 = None
            self.dp2 = None
            self.dp1 = None



    def forward(self, x):
        # out = x
        down0 = out = self.input_block(x)
        # print('down0', down0.size())
        #down0 = self.fc00(out)
        down1 = out = self.resdown1(out)
        # print('down1', down1.size())
        #down1 = self.fc11(out)
        down2 = out = self.resdown2(out)
        # print('down2', down2.size())
        #down2 = self.fc22(out)
        down3 = out = self.resdown3(out)
        # print('down3', down3.size())
        #down3 = self.fc33(out)

        out = self.center_block(out)

        #print(self.fc23(down2).size())
        fuse3 = self.ss3(torch.cat([torch.unsqueeze(down3, dim=1),
                                    torch.unsqueeze(self.fc23(down2), dim=1),
                                    torch.unsqueeze(self.fc13(down1), dim=1),
                                    torch.unsqueeze(self.fc03(down0), dim=1)], dim=1))
        # print('fuse3', fuse3.size())
        # print('before resup3', out.size())
        out = self.resup3(out, fuse3)
        # print('resup3 out', out.size())
        if self.dp3 is not None:
            dp3_out = self.dp3(out)

        fuse2 = self.ss2(torch.cat([torch.unsqueeze(self.fc32(down3), dim=1),
                                    torch.unsqueeze(down2, dim=1),
                                    torch.unsqueeze(self.fc12(down1), dim=1),
                                    torch.unsqueeze(self.fc02(down0), dim=1)], dim=1))
        # fuse2 = self.se2(fuse2)
        # print('fuse2', fuse2.size())
        out = self.resup2(out, fuse2)
        if self.dp2 is not None:
            dp2_out = self.dp2(out)



        fuse1 = self.ss1(torch.cat([torch.unsqueeze(self.fc31(down3), dim=1),
                                    torch.unsqueeze(self.fc21(down2), dim=1),
                                    torch.unsqueeze(down1, dim=1),
                                    torch.unsqueeze(self.fc01(down0), dim=1)], dim=1))
        # fuse1 = self.se1(fuse1)
        out = self.resup1(out, fuse1)
        if self.dp1 is not None:
            dp1_out = self.dp1(out)

        # print(r'resup1 out', out.size())
        # assert(0)
        fuse0 = self.ss0(torch.cat([torch.unsqueeze(self.fc30(down3), dim=1),
                                    torch.unsqueeze(self.fc20(down2), dim=1),
                                    torch.unsqueeze(self.fc10(down1), dim=1),
                                    torch.unsqueeze(down0, dim=1)], dim=1))
        # fuse0 = self.se0(fuse0)
        #fuse0 = torch.cat([self.fc30(down3), self.fc20(down2), self.fc10(down1), down0], 1)
        out = self.resup0(out, fuse0)
        # print('resup0 out', out.size())


        if self.dropout is not None:
            out = self.dropout(out)

        out = self.last(out)
        # print('final out', out.size())
        # assert (0)
        if self.dp3 is None:
            return [out]
        else:
            return [out, dp1_out, dp2_out, dp3_out]

#################################################################3
def main():
    from torchinfo import summary

    net = Net(in_channels=4, nclasses=3, nf=24, relu='relu', up_mode='upconv', dropout_p=0.2, group_norm=1,
              deep_supervision=True)
    # net = Net(in_channels=3, batch_norm=True, up_mode='upsample', padding=True, depth=6, wf=5)
    n_w = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # n_w = count_parameters(net)
    print('n_w = {:,}'.format(n_w))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    summary(net, (2, 4, 64, 64, 64), depth=4)

if __name__ == '__main__':
    main()

