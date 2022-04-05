from torch import nn
import torch

class CJMELoss(nn.Module):

    def __init__(self, margin, distance_fn):
        super(CJMELoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.composite_triplet_loss = CompositeCJMETripletLoss(self.margin, self.distance_fn)
        self.softmax=torch.nn.Softmax()
        self.epsilon=0.1
        self.loss_mse=nn.MSELoss()

    def forward(self, a1, v1, t1, a2, v2, t2, attention_weights,threshold_attention, embeddings):
        logits_audio = self.softmax(torch.matmul(a1.detach(), embeddings.t().detach()))
        logits_video=self.softmax(torch.matmul(v1.detach(), embeddings.t().detach()))
        entropy_audio=torch.distributions.Categorical(logits_audio).entropy()
        entropy_video=torch.distributions.Categorical(logits_video).entropy()
        entropy_resultant_video=entropy_video-entropy_audio-self.epsilon
        entropy_resultant_audio=entropy_audio-entropy_video-self.epsilon

        indices_entropy_resultant_video=entropy_resultant_video>0
        indices_entropy_resultant_audio=entropy_resultant_audio>0
        indices_neutral_audio=entropy_resultant_audio<=0
        indices_neutral_video=entropy_resultant_video<=0
        indices_neutral=indices_neutral_video==indices_neutral_audio
        supervision_attention=torch.ones(entropy_video.size())
        supervision_attention[indices_entropy_resultant_video]=0
        supervision_attention[indices_neutral]=0.5

        loss_supervision_attention=self.loss_mse(attention_weights.cuda(), supervision_attention.unsqueeze(dim=1).cuda())

        ct_loss, ct_debug = self.composite_triplet_loss(a1, v1, t1, a2, v2, t2, attention_weights)
        L_av=self.loss_mse(a1,v1)
        ct_loss=ct_loss.mean()
        return ct_loss+L_av+loss_supervision_attention, {"ct": ct_debug}





class AVGZSLLoss(nn.Module):
    """
    Total Loss which combines the cross-modal decoder loss with the composite triplet loss.
    """

    def __init__(self, margin, distance_fn):
        super(AVGZSLLoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.cross_modal_loss = CrossModalDecoderLoss(self.margin, self.distance_fn)
        self.composite_triplet_loss = CompositeTripletLoss(self.margin, self.distance_fn)

    def forward(self, x_t1, a1, v1, t1, a2, v2, t2, x_ta1, x_tv1, x_tt1, x_ta2, x_tv2):
        cmd_loss, cmd_debug = self.cross_modal_loss(x_t1, x_ta1, x_tv1, x_tt1, x_ta2, x_tv2)
        ct_loss, ct_debug = self.composite_triplet_loss(a1, v1, t1, a2, v2, t2)
        return cmd_loss + ct_loss, {"cmd": cmd_debug, "ct": ct_debug}


class TripletLoss(nn.Module):
    """
    Generic triplet loss class with L2 distance.
    Might be replaced later with torch.nn.TripletLoss()
    """

    def __init__(self, margin, distance_fn):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.torch_l2_loss = nn.TripletMarginLoss(margin=self.margin)

    def forward(self, anchor, positive, negative, size_average=True):

        losses = self.torch_l2_loss(anchor=anchor, positive=positive, negative=negative)
        return losses.mean() if size_average else losses.sum()


class CrossModalDecoderLoss(nn.Module):
    """
    Calculates the cross modal decoder loss.
    """

    def __init__(self, margin, distance_fn):
        super(CrossModalDecoderLoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.triplet_loss = TripletLoss(self.margin, self.distance_fn)

    def forward(self, x_t1, x_ta1, x_tv1, x_tt1, x_ta2, x_tv2):
        distance1 = self.distance_fn(x_tt1, x_t1)
        distance2 = self.distance_fn(x_ta1, x_t1)
        distance3 = self.distance_fn(x_tv1, x_t1)

        l_rec = distance1.mean() + distance2.mean() + distance3.mean()
        l_cta = self.triplet_loss(anchor=x_tt1, positive=x_ta1, negative=x_ta2)
        l_ctv = self.triplet_loss(anchor=x_tt1, positive=x_tv1, negative=x_tv2)
        return l_rec + l_cta + l_ctv, {"l_rec": l_rec.detach(), "l_cta": l_cta.detach(), "l_ctv": l_ctv.detach()}


class CompositeCJMETripletLoss(nn.Module):

    def __init__(self, margin, distance_fn):
        super(CompositeCJMETripletLoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.triplet_loss=nn.TripletMarginLoss(margin=self.margin , reduction="none")


    def forward(self, a1, v1, t1, a2, v2, t2, attention_weights):
        l_ta = self.triplet_loss(anchor=t1, positive=a1, negative=a2)
        l_tv = self.triplet_loss(anchor=t1, positive=v1, negative=v2)
        return (1-attention_weights)*l_ta  + attention_weights*l_tv , {"l_ta": l_ta.detach(), "l_tv": l_tv.detach()}




class CompositeTripletLoss(nn.Module):
    """
    Calculated the composite triplet loss
    """

    def __init__(self, margin, distance_fn):
        super(CompositeTripletLoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.triplet_loss = TripletLoss(self.margin, self.distance_fn)

    def forward(self, a1, v1, t1, a2, v2, t2):
        l_ta = self.triplet_loss(anchor=t1, positive=a1, negative=a2)
        l_at = self.triplet_loss(anchor=a1, positive=t1, negative=t2)
        l_tv = self.triplet_loss(anchor=t1, positive=v1, negative=v2)
        l_vt = self.triplet_loss(anchor=v1, positive=t1, negative=t2)
        return l_ta + l_at + l_tv + l_vt, {"l_ta": l_ta.detach(), "l_at": l_at.detach(), "l_tv": l_tv.detach(),
                                           "l_vt": l_vt.detach()}

class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, x, y):
        raise NotImplementedError()


class L2Loss(DistanceLoss):
    def forward(self, x, y):
        return (x - y).pow(2).sum(1).pow(0.5)


class SquaredL2Loss(DistanceLoss):
    def forward(self, x, y):
        return (x - y).pow(2).sum(1)



class ClsContrastiveLoss(nn.Module):
    '''compute contrastive loss
    '''

    def __init__(self, margin=0.2, max_violation=False, topk=1, reduction='sum'):

        super().__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.topk = topk
        self.reduction = reduction
        if self.reduction == 'weighted':
            self.betas = torch.zeros(701)
            self.betas[1:] = torch.cumsum(1 / (torch.arange(700).float() + 1), 0)

    def forward(self, scores, int_labels):
        batch_size = scores.size(0)

        pos_scores = torch.gather(scores, 1, int_labels.unsqueeze(1))
        pos_masks = torch.zeros_like(scores).bool()
        pos_masks.scatter_(1, int_labels.unsqueeze(1), True)

        d1 = pos_scores.expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_s = cost_s.masked_fill(pos_masks, 0)
        if self.max_violation:
            cost_s, _ = torch.topk(cost_s, self.topk, dim=1)
            if self.reduction == 'mean':
                cost_s = cost_s / self.topk
        else:
            if self.reduction == 'mean':
                cost_s = cost_s / (scores.size(1) - 1)
            elif self.reduction == 'weighted':
                gt_ranks = torch.sum(cost_s > 0, 1).unsqueeze(1)
                weights = self.betas.to(scores.device).unsqueeze(0).expand(batch_size, -1).gather(1, gt_ranks)
                weights = weights / (gt_ranks + 1e-8)
                cost_s = cost_s * weights

        cost_s = torch.sum(cost_s) / batch_size
        return cost_s, {"cost_s": cost_s.detach()}



class APN_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_regre = nn.MSELoss()


    def forward(self, model, output, pre_attri, pre_class, label_a,
                label_v):

        loss_xe =self.criterion(output, label_v)
        loss = loss_xe

        loss_attri =  self.criterion_regre(pre_attri['final'], label_a)
        loss += loss_attri

        for name in model.extract:

            layer_xe = self.criterion(pre_class[name], label_v)
            loss += layer_xe

            loss_attri = self.criterion_regre(pre_attri[name], label_a)
            loss += loss_attri

        return loss, {"loss":loss.detach()}