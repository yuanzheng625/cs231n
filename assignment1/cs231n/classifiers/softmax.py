import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_sample = X.shape[0]
  num_feature = X.shape[1]
  num_class = W.shape[1]
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W) # score (N, C)
  score -= np.amax(score, axis=1).reshape(num_sample, 1)
  exp_score = np.exp(score)
  score_sum = np.sum(exp_score, 1).reshape(num_sample, 1)
  
  score_label = np.zeros_like(score_sum)

  for i in xrange(num_sample):
      score_label[i] = exp_score[i, y[i]]
      loss += -np.log(score_label[i]/score_sum[i])
      
      #print dW[:, y[i]].shape
      #print X[i].shape
      for c in xrange(num_class):
          if c == y[i]:
              dW[:, c] += (score_label[i]/score_sum[i] - 1)*X[i,:]
          else:
              dW[:, c] += (exp_score[i, c]/score_sum[i])*X[i,:]
  
  loss/=num_sample
  loss += reg/2*(np.multiply(W, W)).sum()
  dW/=num_sample
  dW += reg*W
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_sample = X.shape[0]
  num_feature = X.shape[1]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W) # score (N, C)
  score -= np.amax(score, axis=1).reshape(num_sample, 1)
  exp_score = np.exp(score)
  score_sum = np.sum(exp_score, 1).reshape(num_sample, 1)
  
  score_label = np.zeros_like(score_sum)

  for i in xrange(num_sample):
    score_label[i] = exp_score[i, y[i]]
    loss += -np.log(score_label[i]/score_sum[i])
            
    #print dW[:, y[i]].shape
    #print X[i].shape
    for c in xrange(num_class):
        if c == y[i]:
            dW[:, c] += (score_label[i]/score_sum[i] - 1)*X[i,:]
        else:
            dW[:, c] += (exp_score[i, c]/score_sum[i])*X[i,:]
                                
  loss/=num_sample
  loss += reg/2*(np.multiply(W, W)).sum()
  dW/=num_sample
  dW += reg*W
                                                

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

