"""

"""

import torch
from torch.nn import functional as t_functional

from . import readout


def oddx_net(args, train_kwargs):
    num_features = len(train_kwargs['features'])
    args.net_params = [num_features]
    net_class = OddOneOutSingle if train_kwargs['single_img'] is not None else OddOneOut
    return readout.make_model(net_class, args, *args.net_params)


class OddOneOut(readout.ClassifierNet):
    def __init__(self, num_features, classifier_kwargs, readout_kwargs):
        super(OddOneOut, self).__init__(3, 1, **classifier_kwargs, **readout_kwargs)
        self.odd_fc = torch.nn.Linear(self.feature_units * 4, num_features)

    def forward(self, x0, x1, x2, x3):
        x0 = self.do_features(x0)
        x1 = self.do_features(x1)
        x2 = self.do_features(x2)
        x3 = self.do_features(x3)

        comp3 = self.do_classifier(torch.abs(torch.cat([x3 - x0, x3 - x1, x3 - x2], dim=1)))
        comp2 = self.do_classifier(torch.abs(torch.cat([x2 - x0, x2 - x1, x2 - x3], dim=1)))
        comp1 = self.do_classifier(torch.abs(torch.cat([x1 - x0, x1 - x2, x1 - x3], dim=1)))
        comp0 = self.do_classifier(torch.abs(torch.cat([x0 - x1, x0 - x2, x0 - x3], dim=1)))

        y = self.odd_fc(torch.cat([x0, x1, x2, x3], dim=1))
        return torch.cat([comp0, comp1, comp2, comp3], dim=1), y

    @staticmethod
    def loss_function(output, target):
        o_ind, o_class = output
        t_ind, t_class = target
        loss_odd_ind = 0
        for i in range(4):
            loss_odd_ind += t_functional.binary_cross_entropy_with_logits(o_ind[:, i], t_ind[:, i])
        loss_odd_ind = loss_odd_ind / (4 * o_ind.shape[0])
        loss_odd_class = t_functional.cross_entropy(o_class, t_class)
        return loss_odd_ind, loss_odd_class


class OddOneOutSingle(readout.ClassifierNet):
    def __init__(self, num_features, classifier_kwargs, readout_kwargs):
        super(OddOneOutSingle, self).__init__(1, 4, **classifier_kwargs, **readout_kwargs)
        self.odd_fc = torch.nn.Linear(self.feature_units, num_features)

    def forward(self, x):
        x = self.do_features(x)
        odd_ind = self.do_classifier(x)
        odd_class = self.odd_fc(x)
        return odd_ind, odd_class

    @staticmethod
    def loss_function(output, target):
        out_odd_ind, out_odd_class = output
        target_odd_ind, target_odd_class = target
        loss_odd_ind = t_functional.cross_entropy(out_odd_ind, target_odd_ind)
        loss_odd_class = t_functional.cross_entropy(out_odd_class, target_odd_class)
        return loss_odd_ind, loss_odd_class
