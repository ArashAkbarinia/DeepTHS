"""

"""

import torch
from torch.nn import functional as t_functional

from . import readout


def oddx_net(args):
    return readout.make_model(OddOneOut, args)


class OddOneOut(readout.ClassifierNet):
    def __init__(self, classifier_args, **readout_kwargs):
        super(OddOneOut, self).__init__(3, 1, *classifier_args, **readout_kwargs)

    def forward(self, x0, x1, x2, x3):
        x0 = self.do_features(x0)
        x1 = self.do_features(x1)
        x2 = self.do_features(x2)
        x3 = self.do_features(x3)

        comp3 = self.do_classifier(torch.abs(torch.cat([x3 - x0, x3 - x1, x3 - x2], dim=1)))
        comp2 = self.do_classifier(torch.abs(torch.cat([x2 - x0, x2 - x1, x2 - x3], dim=1)))
        comp1 = self.do_classifier(torch.abs(torch.cat([x1 - x0, x1 - x2, x1 - x3], dim=1)))
        comp0 = self.do_classifier(torch.abs(torch.cat([x0 - x1, x0 - x2, x0 - x3], dim=1)))

        return torch.cat([comp0, comp1, comp2, comp3], dim=1)

    def loss_function(self, output, target):
        loss = 0
        for i in range(4):
            loss += t_functional.binary_cross_entropy_with_logits(output[:, i], target[:, i])
        return loss / (4 * output.shape[0])
