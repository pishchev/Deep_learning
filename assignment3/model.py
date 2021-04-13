import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        self.layers = [ConvolutionalLayer(in_channels=input_shape[2], \
                                          out_channels=input_shape[2], \
                                          filter_size=conv1_channels, \
                                          padding=2),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, \
                                       stride=2),
                       ConvolutionalLayer(in_channels=input_shape[2], \
                                          out_channels=input_shape[2], \
                                          filter_size=conv2_channels, \
                                          padding=2),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, \
                                       stride=2),
                       Flattener(),
                       FullyConnectedLayer(n_input=192, \
                                           n_output=n_output_classes)]

    def compute_loss_and_gradients(self, X, y):       
        for param in self.params().values():
            param.grad.fill(0.0)
        
        forward_out = X
        for layer in self.layers:
            forward_out = layer.forward(forward_out)
               
        loss, d_out = softmax_with_cross_entropy(forward_out, y)
        
        backward_out = d_out
        for layer in reversed(self.layers):
            backward_out = layer.backward(backward_out)
        
        return loss

    def predict(self, X):
        forward_out = X
        for layer in self.layers:
            forward_out = layer.forward(forward_out)
        y_pred = np.argmax(forward_out, axis = 1)
        return y_pred


    def params(self):
        result = {  'W1': self.layers[0].params()['W'],         'B1': self.layers[0].params()['B'], 
                    'W2': self.layers[3].params()['W'],         'B2': self.layers[3].params()['B'], 
                    'W3': self.layers[7].params()['W'],         'B3': self.layers[7].params()['B']}
        return result
