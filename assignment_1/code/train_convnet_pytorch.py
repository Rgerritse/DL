"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
from torch import optim, nn
import math
import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  o = np.argmax(predictions, axis=1)
  t = np.argmax(targets, axis=1)
  compared = np.equal(o, t)
  correct = np.sum(compared)
  accuracy = correct / len(compared)
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model.

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  learning_rate = FLAGS.learning_rate
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  eval_freq = FLAGS.eval_freq

  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')

  convnet = ConvNet(3, 10).cuda()

  opt = optim.Adam(convnet.parameters(), lr = learning_rate)
  loss_function = nn.CrossEntropyLoss()

  train_losses = []
  accuracies = []
  steps = []

  for step in range(max_steps):
    total_loss = 0

    x, y = cifar10['train'].next_batch(batch_size)
    x_tensor = torch.from_numpy(x).cuda()
    y_tensor = torch.from_numpy(y).cuda()

    out = convnet(x_tensor)
    loss = loss_function(out, torch.max(y_tensor, 1)[1])
    total_loss += loss

    opt.zero_grad()

    loss.backward()
    opt.step()

    train_losses.append(total_loss)
    print('Step: {} Loss: {:.4f}'.format(step + 1, total_loss))
    if (step + 1) % eval_freq == 0:
      test_pred_seq = []
      test_labels_seq = []
      while(cifar10['test']._epochs_completed == 0):
        test_x, test_y = cifar10['test'].next_batch(batch_size)
        test_x_tensor = torch.from_numpy(test_x).cuda()
        test_y_tensor = torch.from_numpy(test_y).cuda()
        test_out = convnet(test_x_tensor)
        test_out_cpu = test_out.cpu().detach().numpy()
        test_y_tensor_cpu = test_y_tensor.cpu().detach().numpy()
        test_pred_seq.append(test_out_cpu)
        test_labels_seq.append(test_y_tensor_cpu)
      test_pred = np.vstack(test_pred_seq)
      test_labels = np.vstack(test_labels_seq)
      test_accuracy = accuracy(test_pred, test_labels)
      accuracies.append(test_accuracy)
      steps.append(step + 1)

      cifar10['test']._epochs_completed = 0
      print('Step: {} Accuracy {:.2f}'.format(step + 1, test_accuracy))

  plt.plot(range(max_steps), train_losses)
  plt.xlabel("Step")
  plt.ylabel("Training loss")
  plt.show()

  plt.plot(steps, accuracies)
  plt.xlabel("Step")
  plt.ylabel("Test Accuracy")
  plt.show()
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
