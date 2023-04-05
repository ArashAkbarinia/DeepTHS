"""

"""

import torch
from torch.nn import functional as t_functional

from . import readout


def oddx_net(args, train_kwargs=None):
    if train_kwargs is None:
        net_class = OddOneOutSingle if args.single_img else OddOneOut
    else:
        num_features = len(train_kwargs['features']) if args.class_loss else None
        args.net_params = [num_features]
        net_class = OddOneOutSingle if train_kwargs['single_img'] else OddOneOut
    return readout.make_model(net_class, args, *args.net_params)


class OddOneOutDiff(readout.ClassifierNet):
    def __init__(self, num_features, classifier_kwargs, readout_kwargs):
        super(OddOneOutDiff, self).__init__(3, 1, **classifier_kwargs, **readout_kwargs)
        if num_features in [1, None]:
            self.odd_fc = None
        else:
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

        y = None if self.odd_fc is None else self.odd_fc(torch.cat([x0, x1, x2, x3], dim=1))
        return torch.cat([comp0, comp1, comp2, comp3], dim=1), y

    @staticmethod
    def loss_function(output, target):
        o_ind, o_class = output
        t_ind, t_class = target
        loss_odd_ind = 0
        for i in range(4):
            loss_odd_ind += t_functional.binary_cross_entropy_with_logits(o_ind[:, i], t_ind[:, i])
        loss_odd_ind = loss_odd_ind / (4 * o_ind.shape[0])
        loss_odd_class = 0 if o_class is None else t_functional.cross_entropy(o_class, t_class)
        return loss_odd_ind, loss_odd_class


class OddOneOut(readout.ClassifierNet):
    def __init__(self, num_features, classifier_kwargs, readout_kwargs):
        super(OddOneOut, self).__init__(4, 4, **classifier_kwargs, **readout_kwargs)
        if num_features in [1, None]:
            self.odd_fc = None
        else:
            self.odd_fc = torch.nn.Linear(self.feature_units * 4, num_features)

    def forward(self, x0, x1, x2, x3):
        x0 = self.do_features(x0)
        x1 = self.do_features(x1)
        x2 = self.do_features(x2)
        x3 = self.do_features(x3)

        d0 = torch.std(torch.stack([x0 - x1, x0 - x2, x0 - x3]), dim=0)
        d1 = torch.std(torch.stack([x1 - x0, x1 - x2, x1 - x3]), dim=0)
        d2 = torch.std(torch.stack([x2 - x0, x2 - x1, x2 - x3]), dim=0)
        d3 = torch.std(torch.stack([x3 - x0, x3 - x1, x3 - x2]), dim=0)

        x = torch.cat([d0, d1, d2, d3], dim=1)
        odd_ind = self.do_classifier(x)
        odd_class = None if self.odd_fc is None else self.odd_fc(x)
        return odd_ind, odd_class

    @staticmethod
    def loss_function(output, target):
        o_ind, o_class = output
        t_ind, t_class = target
        loss_odd_ind = t_functional.cross_entropy(o_ind, t_ind)
        loss_odd_class = 0 if o_class is None else t_functional.cross_entropy(o_class, t_class)
        return loss_odd_ind, loss_odd_class


class OddOneOutSingle(readout.ClassifierNet):
    def __init__(self, num_features, classifier_kwargs, readout_kwargs):
        super(OddOneOutSingle, self).__init__(1, 4, **classifier_kwargs, **readout_kwargs)
        if num_features in [1, None]:
            self.odd_fc = None
        else:
            self.odd_fc = torch.nn.Linear(self.feature_units, num_features)

    def forward(self, x):
        x = self.do_features(x)
        odd_ind = self.do_classifier(x)
        odd_class = None if self.odd_fc is None else self.odd_fc(x)
        return odd_ind, odd_class

    @staticmethod
    def loss_function(output, target):
        o_ind, o_class = output
        t_ind, t_class = target
        loss_odd_ind = t_functional.cross_entropy(o_ind, t_ind)
        loss_odd_class = 0 if o_class is None else t_functional.cross_entropy(o_class, t_class)
        return loss_odd_ind, loss_odd_class
