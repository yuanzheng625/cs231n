import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  next_h = np.tanh(np.dot(x,Wx) + np.dot(prev_h,Wh) + b)
  cache = (x, prev_h, Wx, Wh, next_h)
  #pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state; should be (N, H)?
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  (x, prev_h, Wx, Wh, next_h) = cache
  dtanh = 1- next_h**2
  dx = np.dot(dnext_h*dtanh, Wx.T)
  dWx = np.dot(x.T, dnext_h*dtanh)
  dprev_h = np.dot(dnext_h*dtanh, Wh.T)
  dWh = np.dot(prev_h.T, dnext_h*dtanh)
  db = np.sum(dnext_h*dtanh, 0)
  
  #pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  _, H = h0.shape
  h = np.zeros((N, T, H))
  cache = {}
  prev_h = h0

  for i in range(T):
      h[:,i,:], cache[i] = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)
      prev_h = h[:,i,:]

  #pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  (x, prev_h, Wx, Wh, next_h) = cache[0]
  T = len(cache)
  N, D = x.shape
  _, H = next_h.shape
  
  #print N, D, T, H

  dx = np.zeros((N, T, D))
  dprev_h = np.zeros((N, H))
  dWx = np.zeros((D, H, T))
  dWh = np.zeros((H, H, T))
  db = np.zeros((H, T))
  
  dnext_h = dh[:,T-1,:]
  i = T-1

  while i >= 0:
      dx[:,i,:], dprev_h, dWx[:,:,i], dWh[:,:,i], db[:,i] = rnn_step_backward(dnext_h, cache[i])
      if i > 0:
        dnext_h = dh[:,i-1,:] + dprev_h

      i = i - 1
          
  dh0 = dprev_h

  dWx = np.sum(dWx, 2)
  dWh = np.sum(dWh, 2)
  db = np.sum(db, 1)

  #pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  N, T = x.shape
  V, D = W.shape
  out = np.zeros((N, T, D))

  for i in range(N):
      for j in range(T):
          out[i,j,:] = W[x[i,j],:]
  #pass
  cache = (x, W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  (x, W) = cache
  dW = np.zeros_like(W)
  N, T = x.shape
  V, D = W.shape
  
  for i in range(N):
      for j in range(T):
          dW[x[i,j]] += dout[i,j,:]
  #pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  (N, H) = prev_h.shape
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  ai = a[:,:H]
  af = a[:,H:2*H]
  ao = a[:,2*H:3*H]
  ag = a[:,3*H:]

  i = sigmoid(ai)
  f = sigmoid(af)
  o = sigmoid(ao)
  g = np.tanh(ag)

  next_c = f*prev_c + i*g
  next_h = o*np.tanh(next_c)
  cache = (Wx, Wh, b, i, f, o, g, prev_h, prev_c, next_c, x)
  #pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  (N, H) = dnext_h.shape
  #a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  #i = sigmoid(ai)
  #f = sigmoid(af)
  #o = sigmoid(ao)
  #g = np.tanh(ag)
                
  #  next_c = f*prev_c + i*g
  #  next_h = o*np.tanh(next_c)

  (Wx, Wh, b, i, f, o, g, prev_h, prev_c, next_c, x) = cache
  # derivative should be next_h (connect to output) with respect to the following
  
  #a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  #i = sigmoid(ai)
  #f = sigmoid(af)
  #o = sigmoid(ao)
  #g = np.tanh(ag)
  
  #next_c = f*prev_c + i*g
  #next_h = o*np.tanh(next_c)
  part2 = np.dot(dnext_c*prev_c*f*(1-f), Wx[:,H:2*H].T) + np.dot(dnext_c*g*i*(1-i), Wx[:,:H].T)+ np.dot(dnext_c*i*(1-g**2), Wx[:,3*H:].T)
  part1 = np.dot(dnext_h*np.tanh(next_c)*o*(1-o), Wx[:,2*H:3*H].T) + np.dot(dnext_h*o*(1-np.tanh(next_c)**2)*prev_c*f*(1-f), Wx[:,H:2*H].T) + np.dot(dnext_h*o*(1-np.tanh(next_c)**2)*g*i*(1-i), Wx[:,:H].T)+ np.dot(dnext_h*o*(1-np.tanh(next_c)**2)*i*(1-g**2), Wx[:,3*H:].T)
  dx = part1 + part2  #(N, D)


  ###################################################################
  """
    dg_prev_h = np.dot((1-g**2),Wh[:,3*H:].T)
    di_prev_h = np.dot(i*(1-i), Wh[:,:H].T)
    df_prev_h = np.dot(f*(1-f),Wh[:,H:2*H].T)
    do_prev_h = np.dot(o*(1-o),Wh[:,2*H:3*H].T) # (N, H)
    
    dnext_c_prev_h = df_prev_h*prev_c + di_prev_h*g + dg_prev_h*i
    dnext_h_prev_h = do_prev_h*np.tanh(next_c) + o*(1-np.tanh(next_c)**2)*dnext_c_prev_h #(N, H)
    dprev_h = dnext_h*dnext_h_prev_h + dnext_c*dnext_c_prev_h
  """
  part2 = np.dot(dnext_c*prev_c*f*(1-f), Wh[:,H:2*H].T) + np.dot(dnext_c*g*i*(1-i), Wh[:,:H].T)+ np.dot(dnext_c*i*(1-g**2), Wh[:,3*H:].T)
  part1 = np.dot(dnext_h*np.tanh(next_c)*o*(1-o), Wh[:,2*H:3*H].T) + np.dot(dnext_h*o*(1-np.tanh(next_c)**2)*prev_c*f*(1-f), Wh[:,H:2*H].T) + np.dot(dnext_h*o*(1-np.tanh(next_c)**2)*g*i*(1-i), Wh[:,:H].T)+ np.dot(dnext_h*o*(1-np.tanh(next_c)**2)*i*(1-g**2), Wh[:,3*H:].T)
  dprev_h = part1 + part2
  ###################################################################
  dprev_c = dnext_c*f + dnext_h*o*(1-np.tanh(next_c)**2)*f #(N, H)
  ###################################################################
  dWx0 = np.dot(x.T, dnext_c*g*i*(1-i)) + np.dot(x.T, dnext_h*o*(1-np.tanh(next_c)**2)*g*i*(1-i))
  dWx1 = np.dot(x.T, dnext_c*prev_c*f*(1-f)) + np.dot(x.T, dnext_h*o*(1-np.tanh(next_c)**2)*prev_c*f*(1-f))
  dWx2 = np.dot(x.T, dnext_h*np.tanh(next_c)*o*(1-o))
  dWx3 = np.dot(x.T, dnext_c*i*(1-g**2)) + np.dot(x.T, dnext_h*o*(1-np.tanh(next_c)**2)*i*(1-g**2))
  
  dWx = np.hstack((dWx0, dWx1, dWx2, dWx3)) #(D, 4H)

  ###################################################################
  dWh0 = np.dot(prev_h.T, dnext_c*g*i*(1-i)) + np.dot(prev_h.T, dnext_h*o*(1-np.tanh(next_c)**2)*g*i*(1-i))
  dWh1 = np.dot(prev_h.T, dnext_c*prev_c*f*(1-f)) + np.dot(prev_h.T, dnext_h*o*(1-np.tanh(next_c)**2)*prev_c*f*(1-f))
  dWh2 = np.dot(prev_h.T, dnext_h*np.tanh(next_c)*o*(1-o))
  dWh3 = np.dot(prev_h.T, dnext_c*i*(1-g**2)) + np.dot(prev_h.T, dnext_h*o*(1-np.tanh(next_c)**2)*i*(1-g**2))

  dWh = np.hstack((dWh0, dWh1, dWh2, dWh3)) #(H, 4H)
  ###################################################################
  db0 = np.sum(dnext_c*g*i*(1-i), 0) + np.sum(dnext_h*o*(1-np.tanh(next_c)**2)*g*i*(1-i),0)
  db1 = np.sum(dnext_c*prev_c*f*(1-f), 0) + np.sum(dnext_h*o*(1-np.tanh(next_c)**2)*prev_c*f*(1-f),0)
  db2 = np.sum(dnext_h*np.tanh(next_c)*o*(1-o), 0)
  db3 = np.sum(dnext_c*i*(1-g**2), 0) + np.sum(dnext_h*o*(1-np.tanh(next_c)**2)*i*(1-g**2),0)

  db = np.hstack((db0, db1, db2, db3)) #(4H,)

  #pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  _, H = h0.shape
  h = np.zeros((N, T, H))
  cache = {}
  prev_h = h0
  prev_c = np.zeros((N, H))
  for i in range(T):
    prev_h, prev_c, cache[i] = lstm_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b)
    h[:,i,:] = prev_h

  #pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  (Wx, Wh, b, i, f, o, g, prev_h, prev_c, next_c, x) = cache[0]
  T = len(cache)
  N, D = x.shape
  _, H = prev_h.shape

  #print N, D, T, H
  dx = np.zeros((N, T, D))
  dprev_h = np.zeros((N, H))
  dWx = np.zeros((D, 4*H, T))
  dWh = np.zeros((H, 4*H, T))
  db = np.zeros((4*H, T))
                    
  dnext_h = dh[:,T-1,:]
  dnext_c = np.zeros((N, H)) # why intialize as 0, because it is not connected to output, thus no change with respect to output
  i = T-1
                            
  while i >= 0:
    #dx[:,i,:], dprev_h, dWx[:,:,i], dWh[:,:,i], db[:,i] = rnn_step_backward(dnext_h, cache[i])
    dx[:,i,:], dprev_h, dprev_c, dWx[:,:,i], dWh[:,:,i], db[:,i] = lstm_step_backward(dnext_h, dnext_c, cache[i])
    if i > 0:
       dnext_h = dh[:,i-1,:] + dprev_h
       dnext_c = dprev_c
    i = i - 1
                                                
  dh0 = dprev_h
  dWx = np.sum(dWx, 2)
  dWh = np.sum(dWh, 2)
  db = np.sum(db, 1)
  
  #pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx
