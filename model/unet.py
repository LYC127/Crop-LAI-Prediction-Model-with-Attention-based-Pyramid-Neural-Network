from unet_parts import *
from torchvision import models

pretrained_net = models.vgg16(pretrained=True)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, pretrained_net, bilinear=True):
        super(UNet, self).__init__()
        self.pretrained_net = pretrained_net
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        with torch.no_grad():
          output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        
        
class newVGG16(nn.Module):
    def __init__(self):
        super(newVGG16, self).__init__()
        self.features = pretrained_net.features
        self.ranges = ((0, 4), (4, 9), (9, 16), (16, 23), (23, 30))
    
    def forward(self, x):
        output = {}
        for idx, (begin, end) in enumerate(self.ranges):
            # self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) #(vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d" % (idx+1)] = x
        return output