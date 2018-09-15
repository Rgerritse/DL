"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
from torch import optim, nn
import math
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
  o = torch.max(predictions, 1)[1].cpu().numpy()
  t = torch.max(targets, 1)[1].cpu().numpy()
  compared = np.equal(o, t)
  correct = np.sum(compared)
  accuracy = correct / len(compared)
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model.

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  #######################
  # PUT YOUR CODE HERE  #
  learning_rate = FLAGS.learning_rate
  epochs = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  eval_freq = FLAGS.eval_freq

  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')

  mlp = MLP(32*32*3, dnn_hidden_units, 10).cuda()

  opt = optim.SGD(mlp.parameters(), lr = learning_rate)
  loss_function = nn.CrossEntropyLoss()

  losses = []
  accuracies = []
  epochs_test = []

  num_batches = math.ceil(cifar10['train'].labels.shape[0]/batch_size)
  for epoch in range(epochs):
    total_loss = 0

    # cifar10['train']._index_in_epoch = 0
    for batch in range(num_batches):

      x, y = cifar10['train'].next_batch(batch_size)
      x_tensor = torch.from_numpy(np.reshape(x, [batch_size, 32 * 32 * 3])).cuda()
      y_tensor = torch.from_numpy(y).cuda()

      out = mlp(x_tensor)
      loss = loss_function(out, torch.max(y_tensor, 1)[1])
      total_loss += loss

      opt.zero_grad()

      loss.backward()
      opt.step()

    losses.append(total_loss)
    print('Epoch: {} Loss: {:.4f}'.format(epoch, total_loss))
    if (epoch + 1) % eval_freq == 0:
        test_x = cifar10['test'].images
        test_y = cifar10['test'].labels
        test_x_tensor = torch.from_numpy(np.reshape(test_x, [test_x.shape[0], 32*32*3])).cuda()
        test_y_tensor = torch.from_numpy(test_y).cuda()

        test_out = mlp(test_x_tensor)
        test_accuracy = accuracy(test_out, test_y_tensor)
        accuracies.append(test_accuracy)
        epochs_test.append(epoch + 1)
        print("\n===================================")
        print('Accuracy {}'.format(test_accuracy))
        print("===================================\n")

  plt.plot(losses)
  plt.show()

  plt.plot(epochs_test, accuracies)
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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
