# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

import random

################################################################################

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size).to(device=device)  # fixme

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # fixme
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)  # fixme



    saved_step = 0

    checkpoints = sorted([int(cp) for cp in os.listdir('checkpoints')])
    if checkpoints:
        state = torch.load('checkpoints/{}'.format(checkpoints[-1]))
        saved_step = state['step'] + 1
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        step = step + saved_step
        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################
        model.zero_grad()
        model.hidden = model.init_hidden(config.batch_size)

        inputs = torch.unsqueeze(torch.stack(batch_inputs), 2).float().to(device=device)
        targets = torch.cat(batch_targets).to(device=device)

        out = model(inputs)
        batch_loss = criterion(out, targets)

        # optimizer.zero_grad()
        batch_loss.backward()

        optimizer.step()

        loss = batch_loss   # fixme

        o = torch.max(out, 1)[1].cpu().numpy()
        t = targets.cpu().numpy()
        compared = np.equal(o, t)
        correct = np.sum(compared)
        accuracy = correct / len(compared)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    int(config.train_steps), config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            random_ix = torch.tensor([random.choice(list(dataset._ix_to_char.keys()))])
            ix_list = [random_ix]

            model.hidden = model.init_hidden(1)
            for i in range(config.seq_length):
                tensor = torch.unsqueeze(torch.unsqueeze(ix_list[-1], 0), 0).float().to(device=config.device)
                out = model(tensor)
                o = torch.max(out, 1)[1]
                ix_list.append(o)
            char_ix = [x.cpu().numpy()[0] for x in ix_list]
            gen_sen = dataset.convert_to_string(char_ix)
            with open('generated_sentences.txt', 'a') as file:
                file.write('{}: {}\n'.format(step, gen_sen))

        if step % config.save_every == 0:
            state = {
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, 'checkpoints/{}'.format(step))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--save_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
