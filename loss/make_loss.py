import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth



def make_loss(Cfg, num_classes):    # modified by gu
    feat_dim = 2048

    if 'triplet' in Cfg.LOSS.TYPE:
        triplet = TripletLoss(Cfg.SOLVER.MARGIN)  # triplet loss
    if 'center' in Cfg.LOSS.TYPE:
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, '
              'range_center,triplet_center, triplet_range_center '
              'but got {}'.format(Cfg.LOSS.TYPE))

    if Cfg.LOSS.LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if Cfg.LOSS.TYPE == 'triplet+center+softmax':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if Cfg.LOSS.LABELSMOOTH == 'on':
                return Cfg.SOLVER.CE_LOSS_WEIGHT * xent(score, target) + \
                       Cfg.SOLVER.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                       Cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return Cfg.SOLVER.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                       Cfg.SOLVER.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                        Cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        if Cfg.LOSS.TYPE == 'center+softmax':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if Cfg.LOSS.LABELSMOOTH == 'on':
                return Cfg.SOLVER.CE_LOSS_WEIGHT * xent(score, target) + \
                       Cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return Cfg.SOLVER.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                        Cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('unexpected loss type')
    return loss_func, center_criterion