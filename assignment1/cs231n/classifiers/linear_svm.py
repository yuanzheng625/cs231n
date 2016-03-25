import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  
  for i in xrange(num_train):
    scores = X[i].dot(W) # X[i] 1*D W D*C -> scores 1*C C classes
    scores_deriv = np.zeros(W.shape)
    
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        scores_deriv[:,j] += X[i] # broadcasting
        scores_deriv[:,y[i]] += -X[i]
        #print num_margin
    dW += scores_deriv
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X, W)
  #print scores.shape
  correct_score_col = scores[np.arange(X.shape[0]), y]
  #print correct_score_col.shape
  correct_class_score = np.tile(correct_score_col.reshape(X.shape[0],1), (1, W.shape[1]))
  scores -= correct_class_score
  scores += 1
  scores[np.arange(X.shape[0]), y] -= 1
  
  mask = scores <= 0
  scores[mask] = 0
  
  #print scores.shape
  
  loss = np.sum(scores)/X.shape[0] + 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  num_classes = W.shape[1]
  
  num_train = X.shape[0]
  
  for i in xrange(num_train):
    mask2 = scores[i] > 0
    deri1 = X[i].reshape(X.shape[1],1)
    deri2 = np.zeros(dW.shape)
    deri2[:,mask2] = deri1
    deri2[:,y[i]] = 0
  
    num = np.sum(mask2)

    #for j in xrange(X.shape[1]):
    #deri2[j,y[i]] = -num*deri1[j]
    deri2[:,y[i]:y[i]+1] = -num*deri1
    dW += deri2

  dW/=num_train
  dW+=reg*W

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
