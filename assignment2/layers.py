import numpy as np

def softmax(predictions): 
    pred_s = np.copy(predictions) 
    if pred_s.ndim == 1: 
        pred_s -= np.max(predictions) 
        exp_pred = np.exp(pred_s) 
        exp_sum= np.sum(exp_pred) 
        return exp_pred / exp_sum 
    else: 
        pred_s = (pred_s.T - np.max(predictions,axis = 1)).T 
        exp_pred = np.exp(pred_s) 
        exp_sum= np.sum(exp_pred,axis=1) 
        return (exp_pred.T / exp_sum).T 



def cross_entropy_loss(probs, target_index): 
    if probs.ndim == 1: 
        return - np.log(probs[target_index]) 
    else: 
        shp = probs.shape[0] 
        logs = -np.log(probs[range(shp), target_index]) 
        return np.sum(logs) / shp 


def softmax_with_cross_entropy(predictions, target_index): 
    softm = softmax(predictions) 
    loss = cross_entropy_loss(softm, target_index) 
    dprediction = softm 
    if softm.ndim == 1: 
        dprediction[target_index] -=1 
    else: 
        shp = predictions.shape[0] 
        dprediction[range(shp),target_index] -=1 
        dprediction /= shp 
    return loss, dprediction 


def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W
    
    return loss, grad



class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        
        return np.where(X<0 , 0 , X )
        
    def backward(self, d_out):
        return np.where(self.X < 0, 0 ,1) * d_out       
        

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
        return  np.dot(X,self.W.value)  + self.B.value

    def backward(self, d_out):
        dx = np.dot(d_out,np.transpose(self.W.value))
        dw = np.dot(np.transpose(self.X),d_out)       
        db = np.dot(np.ones((1, d_out.shape[0])), d_out)
             
        self.B.grad += db
        self.W.grad += dw

        return dx

    def params(self):
        return {'W': self.W, 'B': self.B}
