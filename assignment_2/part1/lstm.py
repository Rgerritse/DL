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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.h_init = torch.zeros(num_hidden, batch_size).to(device=device)
        self.c_init = torch.zeros(num_hidden, batch_size).to(device=device)

        self.W_gx = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_gh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_g = nn.Parameter(torch.zeros(num_hidden, 1))

        self.W_ix = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_ih = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_i = nn.Parameter(torch.zeros(num_hidden, 1))

        self.W_fx = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_fh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_f = nn.Parameter(torch.zeros(num_hidden, 1))

        self.W_ox = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_oh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_o = nn.Parameter(torch.zeros(num_hidden, 1))

        self.W_ph = nn.Parameter(torch.randn(num_classes, num_hidden))
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

    def forward(self, x):
        h_1 = self.h_init
        c_1 = self.c_init

        for t in range(self.seq_length):
            x_t = torch.t(x)[t].unsqueeze(0)

            g = torch.tanh(self.W_gx @ x_t + self.W_gh @ h_1 + self.b_g.expand(-1, self.batch_size))
            i = torch.sigmoid(self.W_ix @ x_t + self.W_ih @ h_1 + self.b_i.expand(-1, self.batch_size))
            f = torch.sigmoid(self.W_fx @ x_t + self.W_fh @ h_1 + self.b_f.expand(-1, self.batch_size))
            o = torch.sigmoid(self.W_ox @ x_t + self.W_oh @ h_1 + self.b_o.expand(-1, self.batch_size))
            c = g * i + c_1 * f
            h = torch.tanh(c) * o

            p = self.W_ph @ h + self.b_p.expand(-1, self.batch_size)

            h_1 = h
            c_1 = c
        return torch.t(p)
