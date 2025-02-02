import torch
from torch import nn
import torchvision.transforms.functional as TV


class Unet(nn.Module):
    def __init__(self,
                 num_classes,
                 input_channels=3,
                 deep_supervision=False,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1],
                 sr_ratios=[8, 4, 2, 1],
                 features=[64, 128, 256, 512],
                 out_channels=1,
                 **kwargs):
        super().__init__()

        # We can not use self.downs = [], because it stores the convs
        # and we want do do eval on these.
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downward path
        for feature in features:
            self.downs.append(DoubleConv(input_channels, feature))
            input_channels = feature

        # Upward path
        # TODO : For a better result we should use Transpose Convolutions
        for feature in reversed(features):
            # First append is the UP
            self.ups.append(
                nn.ConvTranspose2d(
                    # x2 because of the skip connection.
                    feature*2,
                    feature,
                    # These two will double the height, width of the image
                    kernel_size=2,
                    stride=2
                )
            )
            # Second append are the TWO CONVS
            self.ups.append(DoubleConv(feature*2, feature))

        # This is the horizontal path between downward and upward
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # This is the very last conv, from 388,388,64 to 388,388 or as in the
        # paper: 388,388,2. It does not change the size of the image, it only
        # changes the channels.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # Simply reversing the list, because of the upward path
        # will use it in inverse order
        skip_connections = skip_connections[::-1]
        # Step=2 because the upward path has a UP and a DoubleConv,
        # but the skip only applies to the UP part.
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # Integer division by 2 because, altough we want to skip the
            # DoubleConv, we also want to run through
            # the skip_connections one by one.
            skip_connection = skip_connections[idx//2]

            # The INPUT needs to be shaped on a multiple of 16, since it is
            # four down ways. If that is not the case, there will be an error
            # to concatenate because of the MAXPOOL, since them both need same
            # height and width. One work around this is to check if they are
            # different and resize the X
            assert x.shape == skip_connection.shape

            if x.shape != skip_connection.shape:
                # Shape has: 0 BATCH_SIZE, 1 N_CHANNELS, 2 HEIGHT, 3 WIDTH.
                # With [2:] we are taking only height and width.
                x = TV.resize(x, size=skip_connection.shape[2:])

            # We have 4 dims, 0 BATCH, 1 CHANNEL, 2 HEIGHT, 3 WIDTH. We are
            # concatenating them along the channel dimension.
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # This will do the DoubleConv after we did the UP and concatenated
            # the skip connection.
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            # First conv
            # Conv2d(in_cha, out_cha, kernel, stride, padding)
            # When we set stride and padding to one, it is called a
            # SAME CONVOLUTION, the input height and width is the same
            # after the convolution.
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # There was no BachNorm at the time uNet was published, but
            # it helps, so we are going to use it, and to do that,
            # Conv2d 'bias' argument has to be False.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # TODO: What the fuck is this doing? Isn't self.conv just
    # initiated inside __init__?
    def forward(self, x):
        return self.conv(x)
