import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, out0, out1, label):
        gt = label.float()
        D = self.distance(out0, out1).float().squeeze()
        loss = gt * 0.5 * torch.pow(D, 2) + (1 - gt) * 0.5 * torch.pow(torch.clamp(self.margin - D, min=0.0), 2)
        return loss
    

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze()
        D_anchor_negative = self.distance(anchor, negative).float().squeeze()

        dist_positive = torch.pow(D_anchor_positive, 2) #* gt_anchor_positive #* (1.5 - gt_anchor_positive)
        dist_negative = torch.pow(D_anchor_negative, 2) #* (1 - gt_anchor_negative) #* (0.5 + gt_anchor_negative)
        
        loss = F.relu(dist_positive - dist_negative + self.margin)

        return loss

class TripletLoss_2(torch.nn.Module):
    def __init__(self, margin=.1, **kwargs):
        super(TripletLoss_2, self).__init__()
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze()
        D_anchor_negative = self.distance(anchor, negative).float().squeeze()

        loss = F.relu(- D_anchor_negative + D_anchor_positive + self.margin)

        return loss


class AdaptiveTripletLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(AdaptiveTripletLoss, self).__init__()
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze()
        D_anchor_negative = self.distance(anchor, negative).float().squeeze()

        margin = 0.02 + 0.1*(gt_anchor_positive - gt_anchor_negative)

        loss = F.relu(D_anchor_positive - D_anchor_negative + margin)

        return loss



class TripletContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=.5, **kwargs):
        super(TripletContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze()
        D_anchor_negative = self.distance(anchor, negative).float().squeeze()

        pos_side = gt_anchor_positive * 0.5 * torch.pow(D_anchor_positive, 2) +\
                (1 - gt_anchor_positive) * 0.5 * torch.pow(torch.clamp(self.margin - D_anchor_positive, min=0.0), 2)
        neg_side = gt_anchor_negative * 0.5 * torch.pow(D_anchor_negative, 2) +\
                (1 - gt_anchor_negative) * 0.5 * torch.pow(torch.clamp(self.margin - D_anchor_negative, min=0.0), 2)

        loss = pos_side + neg_side
        return loss
    
    
class TripletContrastiveLoss_2(torch.nn.Module):
    def __init__(self, margin=.5, **kwargs):
        super(TripletContrastiveLoss_2, self).__init__()
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze()
        D_anchor_negative = self.distance(anchor, negative).float().squeeze()

        gt_comb = gt_anchor_positive - gt_anchor_negative

        loss = gt_comb * 0.5 * torch.pow(D_anchor_positive - D_anchor_negative, 2) +\
                (1 - gt_comb) * 0.5 * torch.pow(torch.clamp(self.margin + D_anchor_positive - D_anchor_negative, min=0.0), 2)
        
        return loss
    

class TripletContrastiveLoss_3(torch.nn.Module):
    def __init__(self, margin=.5, **kwargs):
        super(TripletContrastiveLoss_3, self).__init__()
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze()
        D_anchor_negative = self.distance(anchor, negative).float().squeeze()

        loss = 0.5 * torch.pow(D_anchor_positive, 2) * gt_anchor_positive +\
                0.5 * torch.pow(torch.clamp(self.margin - D_anchor_negative, min=0.0), 2) * (1 - gt_anchor_negative)
        
        return loss
        


class SARELoss(nn.Module):
    def __init__(self, **kwargs):
        super(SARELoss, self).__init__()
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, q, p, n, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(q, p).float().squeeze()
        D_anchor_negative = self.distance(q, n).float().squeeze()
        
        qp_dist = torch.pow(D_anchor_positive, 2) * (1.5 - gt_anchor_positive)
        qn_dist = torch.pow(D_anchor_negative, 2) * (0.5 + gt_anchor_negative)

        loss =  torch.log(1 + torch.exp(qp_dist - qn_dist))
        return loss


class AdaptiveContrastiveLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(AdaptiveContrastiveLoss, self).__init__()
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze()
        D_anchor_negative = self.distance(anchor, negative).float().squeeze()

        margin = 0.1 + (gt_anchor_positive - gt_anchor_negative) * 0.75

        dist_positive = 0.5 * torch.pow(D_anchor_positive, 2)

        dist_negative = 0.5 * torch.pow(torch.clamp(margin - D_anchor_negative, min=0.0), 2)

        loss = dist_positive + dist_negative

        return loss