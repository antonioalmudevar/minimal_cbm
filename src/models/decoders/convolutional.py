import torch.nn as nn


class ConvDecoder(nn.Module):
    def __init__(self, dim_z, ch_in, size_in):
        super().__init__()
        self.dim_z = dim_z
        self.ch_in = ch_in
        self.size_in = size_in

        self.init_size = size_in // 8
        self.init_channels = 32

        self.fc = nn.Linear(dim_z, self.init_channels * self.init_size * self.init_size)

        # Decoder block: upsample then conv
        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),  # or 'bilinear'
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.decoder = nn.Sequential(
            up_block(self.init_channels, 16),
            up_block(16, 8),
            up_block(8, ch_in),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1),  # Final smoothing
            #nn.Sigmoid()  # If your images are in [0,1]. Use Tanh if [-1,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        x = self.decoder(x)
        return x