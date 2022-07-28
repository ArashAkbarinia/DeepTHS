"""

"""

import torch
from torch.nn import functional as t_functional

from . import readout


def network_class(paradigm):
    return ColourDiscrimination2AFC if paradigm == '2afc' else ColourDiscriminationOddOneOut


def colour_discrimination_net(paradigm, test_net, architecture, target_size, transfer_weights,
                              classifier):
    net_class = network_class(paradigm)

    if test_net:
        model = readout.load_model(architecture, target_size, net_class, classifier)
    else:
        model = net_class(architecture, target_size, transfer_weights, classifier)
    return model


class ColourDiscriminationOddOneOut(readout.ReadOutNetwork):
    def __init__(self, architecture, target_size, transfer_weights, classifier):
        super(ColourDiscriminationOddOneOut, self).__init__(
            architecture, target_size, transfer_weights, 3, classifier, 1
        )

    def forward(self, x0, x1, x2, x3):
        x0 = self.extract_features(x0)
        x0 = x0.view(x0.size(0), -1).float()
        x1 = self.extract_features(x1)
        x1 = x1.view(x1.size(0), -1).float()
        x2 = self.extract_features(x2)
        x2 = x2.view(x2.size(0), -1).float()
        x3 = self.extract_features(x3)
        x3 = x3.view(x3.size(0), -1).float()

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


class ColourDiscrimination2AFC(readout.ReadOutNetwork):
    def __init__(self, architecture, target_size, transfer_weights, classifier):
        super(ColourDiscrimination2AFC, self).__init__(
            architecture, target_size, transfer_weights, 1, classifier, 1
        )

    def forward(self, x0, x1):
        x0 = self.extract_features(x0)
        x0 = x0.view(x0.size(0), -1).float()
        x1 = self.extract_features(x1)
        x1 = x1.view(x1.size(0), -1).float()

        # x = self.fc(torch.cat([x0, x1], dim=1))
        x = torch.abs(x0 - x1)

        return self.do_classifier(x)

    def loss_function(self, output, target):
        loss = t_functional.binary_cross_entropy_with_logits(output, target)
        return loss
