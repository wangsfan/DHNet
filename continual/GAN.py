import torch
import torch.nn as nn

noise_channel = 100
fake_image_channel = 1
figure_size = 64


# output_size=(input_size−1)×stride + kernel_size + −2×padding + output_padding
class Generator(nn.Module):
    def __init__(self, noise_channel, fake_image_channel, figure_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_channel, figure_size * 8, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(figure_size * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 5 x 5
            nn.ConvTranspose2d(figure_size * 8, figure_size * 4, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(figure_size * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 11 x 11
            nn.ConvTranspose2d(figure_size * 4, figure_size * 2, kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(figure_size * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 23 x 23
            nn.ConvTranspose2d(figure_size * 2, figure_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(figure_size),
            nn.ReLU(True),
            # state size. (ngf) x 48 x 48
            nn.ConvTranspose2d(figure_size, fake_image_channel, kernel_size=2, stride=2, padding=1, bias=False),
            # nn.Sigmoid()
            # state size. (nc) x 96 x 96
        )

    def forward(self, input):
        return self.main(input)


from torchsummary import summary
if __name__ == '__main__':
    G = Generator(noise_channel, fake_image_channel, figure_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (noise_channel,1,1)
    x = torch.randn((1,noise_channel, 1, 1),device=device)
    summary(G.to(device), input_size=input_shape)
    out = G(x)
    print(out.shape)


## 384
#     k_list = [11,11,11,9,8]
#     s_list = [1,3,2,2,2]
#     p_list = [0,0,0,0,0]

# ## 192
#     k_list = [7,7,5,5,4]
#     s_list = [1,3,2,2,2]
#     p_list = [0,1,1,1,0]
#
# ## 96
#     k_list = [5,5,5,4,4]
#     s_list = [1,2,2,2,2]
#     p_list = [0,1,1,0,1]



## 96