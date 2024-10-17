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
    
'''
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=.5, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze()
        D_anchor_negative = self.distance(anchor, negative).float().squeeze()

        dist_positive = D_anchor_positive * gt_anchor_positive
        dist_negative = D_anchor_negative * (1 - gt_anchor_negative)
        
        loss = F.relu(dist_positive - dist_negative + self.margin)
        #print(f'anchors: {gt_anchor_positive[0].item()}, {gt_anchor_negative[0].item()}\nLosses: pos - {dist_positive[0].item()}, neg - {dist_negative[0].item()}')

        return loss
'''

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=.5, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze()
        D_anchor_negative = self.distance(anchor, negative).float().squeeze()

        #loss = gt * 0.5 * torch.pow(D, 2) + (1 - gt) * 0.5 * torch.pow(torch.clamp(self.margin - D, min=0.0), 2)
        pos_side = gt_anchor_positive * 0.5 * torch.pow(D_anchor_positive, 2) + (1 - gt_anchor_positive) * 0.5 * torch.pow(torch.clamp(self.margin - D_anchor_positive, min=0.0), 2)
        neg_side = gt_anchor_negative * 0.5 * torch.pow(D_anchor_negative, 2) + (1 - gt_anchor_negative) * 0.5 * torch.pow(torch.clamp(self.margin - D_anchor_negative, min=0.0), 2)

        loss = pos_side + neg_side
        #print(f'anchors: {gt_anchor_positive[0].item()}, {gt_anchor_negative[0].item()}\nLosses: pos - {dist_positive[0].item()}, neg - {dist_negative[0].item()}')

        return loss

class SARELoss(nn.Module):
    def __init__(self):
        super(SARELoss, self).__init__()

    def forward(self, q, p, n, gt_anchor_positive, gt_anchor_negative):
        qp_dist_squared = torch.sum((q - p) ** 2, dim=-1) * gt_anchor_positive
        qn_dist_squared = torch.sum((q - n) ** 2, dim=-1) * (1 - gt_anchor_negative)

        cp_q = torch.exp(-qp_dist_squared) / (torch.exp(-qp_dist_squared) + torch.exp(-qn_dist_squared))

        loss = -torch.log(cp_q)

        return loss # mean is taken in train function
    

class AdaptiveMarginTripletLoss(torch.nn.Module):
    def __init__(self, base_margin=.3, scale_factor=0.1, **kwargs):
        super(AdaptiveMarginTripletLoss, self).__init__()
        self.base_margin = base_margin
        self.scale_factor = scale_factor
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = self.distance(anchor, positive).float().squeeze() * gt_anchor_positive
        D_anchor_negative = self.distance(anchor, negative).float().squeeze() * (1 - gt_anchor_negative)

        # Calculate the adaptive margin.
        margin = self.base_margin + self.scale_factor * (1.0 - torch.abs(gt_anchor_positive - gt_anchor_negative))

        loss = F.relu(D_anchor_positive - D_anchor_negative + margin)
        return loss
    
class CircleLoss(torch.nn.Module):
    def __init__(self, scale=16, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        if self.similarity == 'dot':
            sim_pos = torch.sum(anchor * positive, dim=-1)
            sim_neg = torch.sum(anchor * negative, dim=-1)
        elif self.similarity == 'cos':
            sim_pos = F.cosine_similarity(anchor, positive)
            sim_neg = F.cosine_similarity(anchor, negative)
        else:
            raise ValueError('This similarity is not implemented.')

        alpha_p = torch.relu(-sim_pos + 1 + self.margin)
        alpha_n = torch.relu(sim_neg + self.margin)
        
        # Scale FoV values to adjust the loss
        alpha_p = alpha_p * gt_anchor_positive
        alpha_n = alpha_n * gt_anchor_negative


        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - self.scale * alpha_p * (sim_pos - delta_p)
        logit_n = self.scale * alpha_n * (sim_neg - delta_n)

        loss = torch.log(1 + torch.exp(logit_n + logit_p))

        return loss

'''
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=.5, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cos_sim = F.cosine_similarity

    def forward(self, anchor, positive, negative, gt_anchor_positive, gt_anchor_negative):
        D_anchor_positive = (1 - self.cos_sim(anchor, positive).float().squeeze()) * gt_anchor_positive
        D_anchor_negative = (1 + self.cos_sim(anchor, negative).float().squeeze()) * (1 - gt_anchor_negative)
 
        
        #dist_positive *= gt_anchor_positive
        #dist_negative *= torch.ones_like(gt_anchor_negative) - gt_anchor_negative

        loss = D_anchor_positive + D_anchor_negative #+ self.margin
        #print(f'Positive GT: {gt_anchor_positive[0].item()}, Negative GT: {gt_anchor_negative[0].item()}')
        #print(f'Positive loss: {dist_positive[0].item()}, Negative loss: {dist_negative[0].item()}, Result loss {loss[0].item()}')

        return loss
'''