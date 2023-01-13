from torch import nn
import torchvision
import time

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14, pretrained=True):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet50(pretrained=pretrained) 

        resnet_short = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*resnet_short)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()


    def forward(self, images):
        out = self.resnet(images)  
        out = self.adaptive_pool(out)  
        out = out.permute(0, 2, 3, 1) 
        return out


    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune