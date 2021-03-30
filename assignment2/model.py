import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = [FullyConnectedLayer(n_input , hidden_layer_size) , 
                       ReLULayer() ,
                       FullyConnectedLayer(hidden_layer_size , n_output)]

    def compute_loss_and_gradients(self, X, y):
        
        params = self.params()
        for key in params:
            params[key].grad = 0
        
        data = X.copy()
        for layer in self.layers:
            data = layer.forward(data)
        
        loss, d_out = softmax_with_cross_entropy(data, y)
        
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        
        for key in params:
            loss_, grad_ = l2_regularization(params[key].value, self.reg)
            params[key].grad += grad_
            
            loss += loss_
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        params = self.params()
        for key in params:
            params[key].grad = 0
        
        data = X.copy()
        for layer in self.layers:
            data = layer.forward(data)
            
        prediction = np.argmax(data, axis = 1)

        return prediction

    def params(self):
        result = {
            'flw':self.layers[0].params()['W'],
            'flb':self.layers[0].params()['B'],
            'slw':self.layers[2].params()['W'],
            'slb':self.layers[2].params()['B'],
        }
        
        return result
