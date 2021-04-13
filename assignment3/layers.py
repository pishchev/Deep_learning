import numpy as np

def softmax(predictions):
    copy_predictions = np.copy(predictions)
    if predictions.ndim == 1:
        copy_predictions -= np.max(copy_predictions)
        calculated_exp = np.exp(copy_predictions)
        copy_predictions = calculated_exp / np.sum(calculated_exp)
    else:
        copy_predictions -= np.amax(copy_predictions, axis=1, keepdims=True)
        calculated_exp = np.exp(copy_predictions)
        copy_predictions = calculated_exp / np.sum(calculated_exp, axis=1, keepdims=True)
    return copy_predictions


def cross_entropy_loss(probs, target_index):
    if probs.ndim == 1:
        loss_func = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        every_batch_loss = -np.log(probs[range(batch_size), target_index])
        loss_func = np.sum(every_batch_loss) / batch_size
    return loss_func

def l2_regularization(W, reg_strength):
    l2_reg_loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W
    return l2_reg_loss, grad


def softmax_with_cross_entropy(preds, target_index):
    d_preds = softmax(preds)
    loss = cross_entropy_loss(d_preds, target_index)
    
    if preds.ndim == 1:
        d_preds[target_index] -= 1
    else:
        batch_size = preds.shape[0]
        d_preds[range(batch_size), target_index] -= 1
        d_preds /= batch_size
    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.where(X >= 0, X, 0)
        return result

    def backward(self, d_out):
        dX = np.where(self.X >= 0, 1, 0) * d_out
        return dX

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        dX = np.dot(d_out, self.W.value.T)
        dW = np.dot(self.X.T, d_out)
        dB = np.dot(np.ones((1, d_out.shape[0])), d_out)
        self.W.grad += dW
        self.B.grad += dB
        return dX

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        X_pad = np.zeros((batch_size , height+ 2*self.padding , width+ 2*self.padding , channels))      
        X_pad[: , self.padding: X_pad.shape[1]-self.padding , self.padding:X_pad.shape[2]-self.padding , :] = X
        self.X = X_pad     
        out_height = X_pad.shape[1] - self.filter_size +1
        out_width = X_pad.shape[2] - self.filter_size +1       
        out = np.zeros((batch_size , out_height , out_width , self.out_channels))        
        for y in range(out_height):
            for x in range(out_width):
                X_local_mat = X_pad[: , y: y+self.filter_size , x:x+self.filter_size, :]                
                X_local_arr = X_local_mat.reshape(batch_size , self.filter_size*self.filter_size * self.in_channels)            
                W_arr = self.W.value.reshape( self.filter_size * self.filter_size * self.in_channels,self.out_channels)
                Res_arr = np.dot(X_local_arr , W_arr) + self.B.value          
                Res_mat = Res_arr.reshape(batch_size , 1 , 1, self.out_channels)              
                out[: , y: y+self.filter_size , x:x+self.filter_size, :]= Res_mat               
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dX = np.zeros((batch_size, height, width, channels))
        W_arr = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        for x in range(out_width):
            for y in range(out_height):
                X_local_mat = self.X[:, x:x +self.filter_size , y:y+self.filter_size, :]           
                X_arr = X_local_mat.reshape(batch_size, self.filter_size * self.filter_size * self.in_channels)
                d_local = d_out[:, x:x + 1, y:y + 1, :]
                dX_arr = np.dot(d_local.reshape(batch_size, -1), W_arr.T)
                dX[:, x:x +self.filter_size , y:y+self.filter_size, :] += dX_arr.reshape(X_local_mat.shape)
                dW = np.dot(X_arr.T, d_local.reshape(batch_size, -1))
                dB = np.dot(np.ones((1, d_local.shape[0])), d_local.reshape(batch_size, -1))
                self.W.grad += dW.reshape(self.W.value.shape)
                self.B.grad += dB.reshape(self.B.value.shape)
        return dX[:, self.padding : (height - self.padding), self.padding : (width - self.padding), :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.masks = {}

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        self.masks.clear()

        out_height = int((height - self.pool_size) / self.stride + 1)
        out_width = int((width - self.pool_size) / self.stride + 1)
        
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        mult = self.stride
        
        for x in range(out_width):
            for y in range(out_height):
                I = X[:, x*mult:x*mult+self.pool_size, y*mult:y*mult+self.pool_size, :]
                self.mask(x=I, pos=(x, y))
                output[:, x, y, :] = np.max(I, axis=(1, 2))
        return output
        

    def backward(self, d_out):
        _, out_height, out_width, _ = d_out.shape
        dX = np.zeros_like(self.X)

        mult = self.stride
        

        for x in range(out_width):
            for y in range(out_height):

                dX[:, x*mult:x*mult+self.pool_size, y*mult:y*mult+self.pool_size, :] += d_out[:, x:x + 1, y:y + 1, :] * self.masks[(x, y)]  
        return dX
    
    def mask(self, x, pos):
        zero_mask = np.zeros_like(x)
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((batch_size, channels))
        zero_mask.reshape(batch_size, height * width, channels)[n_idx, idx, c_idx] = 1
        self.masks[pos] = zero_mask

    def params(self):
        return {}

class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        return {}
