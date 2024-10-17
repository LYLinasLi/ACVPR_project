import torch.nn as nn
import torch
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Autoencoder(nn.Module):
    def __init__(self, feature_length):
        super(Autoencoder, self).__init__()
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_length, feature_length//2),
            nn.ReLU(),
            nn.Linear(feature_length//2, feature_length//4),
            nn.ReLU(),
        )
        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_length//4, feature_length//2),
            nn.ReLU(),  # apply ReLU to output layer as well
            nn.Linear(feature_length//2, feature_length),
            nn.ReLU(),  # apply ReLU to output layer as well
        )
        print('using autoencoder')

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class BaseNet(nn.Module):
    def __init__(self, backbone, global_pool=None, poolkernel=7, norm=None, p=3):
        super(BaseNet, self).__init__()
        self.backbone = backbone
        for name, param in self.backbone.named_parameters():
                n=param.size()[0]
        self.feature_length=n
        #self.autoencoder = Autoencoder(self.feature_length)

        if global_pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        elif global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif global_pool == "GeM":
            self.pool = GeM(p=p)
        else:
            self.pool = None
        self.norm=norm

    def forward(self, x0):
        out = self.backbone.forward(x0)

        # out.last_hidden_state for mobilevit
        out = self.pool.forward(out).squeeze(-1).squeeze(-1)

        #out = self.autoencoder(out)  # pass output through the autoencoder

        if self.norm == "L2":
            out = nn.functional.normalize(out)
        return out


class SiameseNet(BaseNet):
    def __init__(self, backbone, global_pool=None, poolkernel=7, norm=None, p=3):
        super(SiameseNet, self).__init__(backbone, global_pool, poolkernel, norm=norm, p=p)
    
    def forward_single(self, x0):
        return super(SiameseNet, self).forward(x0)
    def forward(self, x0, x1):
        out0 = super(SiameseNet, self).forward(x0)
        out1 = super(SiameseNet, self).forward(x1)
        return out0, out1


class TripletNet(BaseNet):
    def __init__(self, backbone, global_pool=None, poolkernel=7, norm=None, p=3):
        super(TripletNet, self).__init__(backbone, global_pool, poolkernel, norm=norm, p=p)
    
    def forward_single(self, x0):
        return super(TripletNet, self).forward(x0)
        
    def forward(self, anchor, positive, negative):
        out_anchor = super(TripletNet, self).forward(anchor)
        out_positive = super(TripletNet, self).forward(positive)
        out_negative = super(TripletNet, self).forward(negative)
        return out_anchor, out_positive, out_negative