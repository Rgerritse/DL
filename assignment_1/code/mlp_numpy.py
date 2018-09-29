"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object.

    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP

    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    super(MLP, self).__init__()
    #
    num_hidden = len(n_hidden)
    self.hidden = []
    # self.hidden = nn.ModuleList()
    self.soft = SoftMaxModule()
    self.relu = ReLUModule()
    #
    for i in range(num_hidden):
        if i == 0:
            self.hidden.append(LinearModule(n_inputs, n_hidden[i]))
        else:
            self.hidden.append(LinearModule(n_hidden[i-1], n_hidden[i]))
    #
    if num_hidden == 0:
        self.out = LinearModule(n_inputs, n_classes)
    else:
        self.out = LinearModule(n_hidden[num_hidden - 1], n_classes)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = x
    for layer in self.hidden:
        out = layer.forward(out)
        out = self.relu.forward(out)
    out = self.out.forward(out)
    out = self.soft.forward(out)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss.

    Args:
      dout: gradients of the loss

    TODO:
    Implement backward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dout = self.soft.backward(dout)
    dout = self.out.backward(dout)
    for layer in reversed(self.hidden):
        dout = self.relu.backward(dout)
        dout = layer.backward(dout)
    ########################
    # END OF YOUR CODE    #
    #######################

    return
