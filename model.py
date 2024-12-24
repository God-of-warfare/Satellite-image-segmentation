import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double Convolution block with batch normalization
    (Conv2d -> BN -> ReLU) × 2
    """

    def __init__(self, in_channels, out_channels,dropout_p=0.2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, init_features=64,dropout_p=0.2):
        super().__init__()

        # Save parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        features = init_features

        # Encoder path
        self.encoder1 = DoubleConv(in_channels, features,dropout_p=dropout_p)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(features, features * 2,dropout_p=dropout_p)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(features * 2, features * 4,dropout_p=dropout_p)

        # Decoder path
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = DoubleConv(features * 4, features * 2,dropout_p=dropout_p)  # *4 because of concat

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = DoubleConv(features * 2, features,dropout_p=dropout_p)  # *2 because of concat

        # Final convolution
        self.final_conv = nn.Conv2d(features, num_classes, kernel_size=1)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))

        # Decoder
        dec2 = self.upconv2(bottleneck)
        # Handle cases where dimensions don't match perfectly
        if dec2.shape != enc2.shape[2:]:
            enc2 = nn.functional.interpolate(enc2, size=dec2.shape[2:])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        # Handle cases where dimensions don't match perfectly
        if dec1.shape != enc1.shape[2:]:
            enc1 = nn.functional.interpolate(enc1, size=dec1.shape[2:])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final convolution and activation
        return nn.functional.softmax(self.final_conv(dec1), dim=1)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def get_number_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)