"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data.
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module.

    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0.

    Also, initialize gradients with zeros.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': np.random.normal(0, 0.0001, (out_features, in_features)), 'bias': np.zeros(out_features)}
    self.grads = {'weight': np.zeros((out_features, in_features)), 'bias': np.zeros(out_features)}
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    Wx = x @ self.params['weight'].T
    out = np.add(Wx, self.params['bias'])
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to
    layer parameters in self.grads['weight'] and self.grads['bias'].
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.grads['weight'] = dout.T @ self.x
    self.grads['bias'] = np.mean(dout.T, axis=1)
    dx = dout @ self.params['weight']
    # dx = 0
    ########################
    # END OF YOUR CODE    #
    #######################
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    argmax = np.argmax(y)
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    out = np.maximum(0, x)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = dout * np.sign(np.maximum(0, self.x))
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick -

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # print("===============softmax forward")
    self.x = x
    # print("x")
    # print(x.shape)
    b = np.amax(x, axis=1)
    # print("b")
    # print(b.shape)
    y = np.exp(np.add(x.T, -b).T)
    out = y/np.sum(y, axis=1)[:, None]

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # print("dout")
    # print("***************softmax dout")

    temp = np.diag(np.diag(self.x @ (1-self.x).T)) - (self.x @ self.x.T)
    dx = (dout.T @ temp).T
    # dx = np.transpose(self.x + dout @ self.x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    argmax = np.argmax(y, axis=1)
    arr = x[np.arange(x.shape[0]), argmax]
    out = np.sum(-np.log(arr))
    ########################
    # END OF YOUR CODE    #
    #######################
    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = -y/x
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
