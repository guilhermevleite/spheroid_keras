from torch import nn
from monai.networks.net import ViT


class UnetR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image
    Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
      self,
      num_classes: int,
      input_channels: int = 3,
      deep_supervision: bool = False,
      in_channels: int = 3,
      out_channels: int = 1,
      img_size: Tuple[int, int, int] = (96, 96, 96),
      feature_size: int = 16,
      hidden_size: int = 768,
      mlp_dim: int = 3072,
      num_heads: int = 12,
      pos_embed: str = "perceptron",
      norm_name: Union[Tuple, str] = "instance",
      conv_block: bool = False,
      res_block: bool = True,
      dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
        in_channels: dimension of input channels.
        out_channels: dimension of output channels.
        img_size: dimension of input image.
        feature_size: dimension of network feature size.
        hidden_size: dimension of hidden layer.
        mlp_dim: dimension of feedforward layer.
        num_heads: number of attention heads.
        pos_embed: position embedding layer type.
        norm_name: feature normalization type and arguments.
        conv_block: bool argument to determine if convolutional block is used
        res_block: bool argument to determine if residual block is used.
        dropout_rate: faction of the input units to drop.

        Examples::

        # for single channel input 4-channel output with patch size of
        (96,96,96), feature size of 32 and batch norm
        >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96),
        feature_size=32, norm_name='batch')

        for 4-channel input 3-channel output with patch size of
        (128,128,128), conv position embedding and instance norm
        >>> net = UNETR(in_channels=4, out_channels=3,
        img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
          img_size[0] // self.patch_size[0],
          img_size[1] // self.patch_size[1],
          img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
          in_channels=in_channels,
          img_size=img_size,
          patch_size=self.patch_size,
          hidden_size=hidden_size,
          mlp_dim=mlp_dim,
          num_layers=self.num_layers,
          num_heads=num_heads,
          pos_embed=pos_embed,
          classification=self.classification,
          dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
          spatial_dims=3,
          in_channels=in_channels,
          out_channels=feature_size,
          kernel_size=3,
          stride=1,
          norm_name=norm_name,
          res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
          spatial_dims=3,
          in_channels=hidden_size,
          out_channels=feature_size * 2,
          num_layer=2,
          kernel_size=3,
          stride=1,
          upsample_kernel_size=2,
          norm_name=norm_name,
          conv_block=conv_block,
          res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
          spatial_dims=3,
          in_channels=hidden_size,
          out_channels=feature_size * 4,
          num_layer=1,
          kernel_size=3,
          stride=1,
          upsample_kernel_size=2,
          norm_name=norm_name,
          conv_block=conv_block,
          res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
          spatial_dims=3,
          in_channels=hidden_size,
          out_channels=feature_size * 8,
          num_layer=0,
          kernel_size=3,
          stride=1,
          upsample_kernel_size=2,
          norm_name=norm_name,
          conv_block=conv_block,
          res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
          spatial_dims=3,
          in_channels=hidden_size,
          out_channels=feature_size * 8,
          kernel_size=3,
          upsample_kernel_size=2,
          norm_name=norm_name,
          res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
          spatial_dims=3,
          in_channels=feature_size * 8,
          out_channels=feature_size * 4,
          kernel_size=3,
          upsample_kernel_size=2,
          norm_name=norm_name,
          res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
          spatial_dims=3,
          in_channels=feature_size * 4,
          out_channels=feature_size * 2,
          kernel_size=3,
          upsample_kernel_size=2,
          norm_name=norm_name,
          res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
          spatial_dims=3,
          in_channels=feature_size * 2,
          out_channels=feature_size,
          kernel_size=3,
          upsample_kernel_size=2,
          norm_name=norm_name,
          res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore
