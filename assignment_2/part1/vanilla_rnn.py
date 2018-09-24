################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.h_init = torch.zeros(num_hidden, batch_size).cuda()

        self.W_hx = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_hh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.W_ph = nn.Parameter(torch.randn(num_classes, num_hidden))

        self.b_h = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

    def forward(self, x):
        # Implementation here ...
        h_1 = self.h_init


        # print()
        # for t in range(self.seq_length):
        for t in range(1):
            x_t = torch.t(x)[t].unsqueeze(0)
            # print(x_t)
        #
        #
            a = self.W_hx @ x_t + self.W_hh @ h_1 + self.b_h.expand(-1, self.batch_size)
            # print(a)
            # print()
            h = (torch.exp(a) - torch.exp(-a)) / (torch.exp(a) + torch.exp(-a))
        #
            p = self.W_ph @ h + self.b_p.expand(-1, self.batch_size)
        #
            # h_1 = h
        return torch.t(p)
        # pass
