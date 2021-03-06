import numpy as np

from time import time

# RANDOM_SEED = 886

class ActivationFunction:
    ''' An ActivationFunction is applied to Z to get the output A, 
        but its derivative expects the value A, not Z (!):

        A == __call__(Z) and derivative(A) == derivative(__call__(Z)), 
        calling derivative(Z) will often yield WRONG results
    '''
    def __call__(self, Z):
        ''' Z.shape=(n_examples, layer_output_size) '''
        raise NotImplementedError    
    def derivative(self, A):
        ''' A.shape=(n_examples, layer_output_size) '''
        raise NotImplementedError

class Linear(ActivationFunction):
    def __call__(self, Z):
        return Z
    def derivative(self, A):
        return np.ones_like(A)

class Sigmoid(ActivationFunction):
    def __call__(self, Z):
        return 1 / (1 + np.exp(-Z))
    def derivative(self, A):
        return A * (1 - A) # Sigmoid(Z) * (1 - Sigmoid(Z))

class ReLU(ActivationFunction):
    def __call__(self, Z):
        return np.maximum(0, Z)
    def derivative(self, A):
        return np.where(A > 0, 1, 0)

class SoftMax(ActivationFunction):
    def __call__(self, Z):
        exp = np.exp(Z - Z.max(axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    def derivative(self, A, eps=1e-9):
        Ypred = A+eps
        return Ypred * (1 - Ypred) # SoftMax(Z) * (1 - SoftMax(Z))

###############################################################################

class CostFunction:
    ''' A CostFunction is applied to Y (the target values) and Ypred (the predicted values) to get a scalar output
        Its derivative w.r.t. Ypred also expects Y and Ypred, but returns a tensor of shape (n_examples, last_layer_output_size)
        
        obs.: Ypred is the last layer's activation values: last_layer.A == last_layer.g(last.layer.Z), 
              i.e. last_layer is the output layer of the network
    '''
    def __call__(self, Y, Ypred):
        ''' Y.shape == Ypred.shape == (n_examples, last_layer_output_size) '''
        raise NotImplementedError # [J(Y, Ypred) == J(Y, A^L)]
    def derivative(self, Y, Ypred):
        ''' Y.shape == Ypred.shape == (n_examples, last_layer_output_size) '''
        raise NotImplementedError # [dJ/dYpred == dJ/dA^L]
    def deltaL(self, Y, Ypred, activation_function):
        ''' Y.shape == Ypred.shape == (n_examples, last_layer_output_size) 

            activation_function should be the last layer's g function, thus Ypred == activation_function(Z)
            and the value returned is the delta for the output layer of the network (delta^L == dJ/dZ^L)
        '''
        return self.derivative(Y, Ypred) * activation_function.derivative(Ypred) # [dJ/dZ^L == dJ/dYpred . dYpred/dZ^L]
        # obs.: Ypred == A^L == g(Z^L), thus dYpred/dZ^L == dA^L/dZ^L == g'(Z^L)
        #       [dJ/dZ^L == dJ/dYpred . dYpred/dZ^L == dJ/dA^L . dA^L/dZ^L]

class CrossEntropy(CostFunction):
    def __call__(self, Y, Ypred, eps=1e-9):
        return np.mean( -(Y * np.log(Ypred+eps)).sum(axis=1) )
    def derivative(self, Y, Ypred, eps=1e-9):
        m = Ypred.shape[0]
        return - (Y / (Ypred+eps)) / m
    def deltaL(self, Y, Ypred, activation_function):
        if isinstance(activation_function, SoftMax):
            m = Ypred.shape[0]
            # numerically stable
            return (Ypred - Y) / m # (SoftMax(Z) - Y) / m
        else:
            return super().deltaL(Y, Ypred, activation_function)

class SoftmaxCrossEntropy(CostFunction):
    def __call__(self, Y, Ypred):
        exp = np.exp(Ypred - Ypred.max(axis=1, keepdims=True))
        Softmax = exp / np.sum(exp, axis=1, keepdims=True)
        return np.mean( -(Y * np.log(Softmax)).sum(axis=1) )
    def derivative(self, Y, Ypred):
        exp = np.exp(Ypred - Ypred.max(axis=1, keepdims=True))
        Softmax = exp / np.sum(exp, axis=1, keepdims=True)
        m = Softmax.shape[0]
        return (Softmax - Y) / m
    def deltaL(self, Y, Ypred, activation_function):
        if isinstance(activation_function, Linear):
            # Linear.derivative(Z) is a matrix of ones, so 
            # calling it doesn't change the returned value
            return self.derivative(Y, Ypred) # (SoftMax(Z) - Y) / m
        else:
            return super().deltaL(Y, Ypred, activation_function)

###############################################################################

class Optimizer:
    ''' The optimizer's optimization policy should be implemented on its update(layers) function '''
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update(self, layers):
        ''' Updates the parameters (i.e. weights and biases) for each layer in the layers list '''
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
    def update(self, layers):
        for layer in layers[1:]:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db   

###############################################################################

class Layer:
    ''' A.shape == (n_examples, output_size)
        Z.shape == (n_examples, output_size)
        W.shape == (input_size, output_size)
        b.shape == (output_size, )
        X.shape == (n_examples, input_size)
        obs.:
            input_size == prev_layer.output_size
            output_size == next_layer.input_size
    '''
    def __init__(self, output_size, activation_function):
        if activation_function != None:
            # obs.: the activation_function should only be None for the network's input layer
            assert(isinstance(activation_function, ActivationFunction)), "Invalid object type for activation_function"
        
        self.input_size = None
        self.output_size = output_size
        
        # activation function
        self.g = activation_function # g_prime == activation_function.derivative
        
        # activation values
        self.A = None # self.A == self.g(self.Z)
        self.Z = None # prev_layer.A @ self.W + self.b
        
        # output value of the previous layer
        self.X = None # == prev_layer.A
        self.dX = None
        
        # parameters (weights and biases)
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
    
    def init(self, input_size, weight_initialization):
        ''' Sets the layer's input_size and initializes its weights and biases '''
        self.input_size = input_size
        if weight_initialization == 'xavier':
            stddev = np.sqrt(1 / self.input_size)
            self.W = stddev * np.random.randn(self.input_size, self.output_size)
            self.b = np.random.randn(self.output_size, )
        elif weight_initialization == 'xavier_avg':
            stddev = np.sqrt(2 / (self.input_size + self.output_size))
            self.W = stddev * np.random.randn(self.input_size, self.output_size)
            self.b = np.random.randn(self.output_size, )
        elif weight_initialization == 'rand_-1_to_1':
            self.W = 2 * np.random.randn(self.input_size, self.output_size) - 1
            self.b = 2 * np.random.randn(self.output_size, ) - 1
        else:
            raise ValueError(f"Invalid weight_initialization value: '{weight_initialization}'")
    
    @property
    def params_count(self):
        return self.W.size + self.b.size
    
    # receives the activation values of the previous layer (i.e. this layer's input)
    # returns the activation values of the current layer (i.e. next layer's input)
    def feedforward(self, X):
        ''' X.shape == (n_examples, self.input_size) '''
        assert(X.shape[1] == self.input_size)
        self.X = X
        # (n_examples, output_size) = (n_examples, input_size) @ (input_size, output_size) + (output_size, )
        self.Z = self.X @ self.W + self.b
        self.A = self.g(self.Z)
        return self.A
    
    # receives the derivative of the cost function w.r.t. the Z value of the current layer [dJ/dZ = dJ/dA . dA/dZ]
    # returns the derivative of the cost function w.r.t. the A value of the previous layer [dJ/dX = dJ/dZ . dZ/dX]
    # obs.: the A value of the previous layer is this layer's input value X
    def backprop(self, dZ):
        ''' dZ.shape == (n_examples, self.output_size)
        
            Note that only calling backprop doesn't actually update the layer parameters
        '''
        assert(dZ.shape[1] == self.output_size)
        # (input_size, output_size) = (input_size, n_examples)  @ (n_examples, output_size)
        # (output_size, )           = (n_examples, output_size).sum(axis=0)
        # (n_examples, input_size)  = (n_examples, output_size) @ (output_size, input_size), input_size==prev_layer.output_size
        self.dW = (self.X).T @ dZ # [dJ/dW = dJ/dZ . dZ/dX]
        self.db = dZ.sum(axis=0)  # [dJ/db = dJ/dZ . dZ/db]
        self.dX = dZ @ (self.W).T # [dJ/dX = dJ/dZ . dZ/dX]
        return self.dX
        # note that dJ/dX is dJ/dA for the previous layer (since this layer's input X is the previous layer's A)
        

###############################################################################

class NN:
    def __init__(self, layers, cost_function, optimizer, weight_initialization='xavier'):
        assert(isinstance(cost_function, CostFunction)), "Invalid object type for cost_function"
        
        self.J = cost_function # cost_function(Y, Ypred)
        # obs.: cost_function.derivative is the derivative of J w.r.t. the last layer's activation values [dJ/dYpred]
        #       Ypred == self.layers[-1].A, thus [dJ/dYpred == dJ/dA^L]
        #
        #       cost_function.deltaL is the derivative of J w.r.t. the last layer's Z values [dJ/dZ^L]
        #       Z^L == self.layers[-1].Z, thus [dJ/dZ^L == dJ/dA^L . dA^L/dZ^L == dJ/dYpred . dYpred/dZ^L]
        
        self.optimizer = optimizer # obs.: the learning rate is set on the optimizer object
        
        self.layers = []
        self.layers.append(layers[0]) # input layer (we don't init it since we set it's activation values manually)
        for l in range(1, len(layers)):            
            # sets the layer's input_size as the last layer's output_size and initializes its weights and biases
            layers[l].init(input_size=layers[l-1].output_size, weight_initialization=weight_initialization)            
            self.layers.append(layers[l]) # adds the initialized layer to the network
        
        self.history = { "loss": [], "loss_val": [], "acc": [], "acc_val": [] }
    
    # note that we use zero-based indexing here, so
    # the 1st layer is self.layers[0] and the last is self.layers[len(self.layers) - 1]
    
    def predict(self, X):
        ''' X.shape == (n_examples, self.layers[0].input_size) '''
        assert(X.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        activation = X # network's input
        for l in range(1, len(self.layers)):
            Z = activation @ self.layers[l].W + self.layers[l].b
            activation = self.layers[l].g(Z)
        return activation # network's output (Ypred)
    
    def feedforward(self, X):
        ''' X.shape     == (n_examples, self.layers[0].input_size) '''
        assert(X.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        self.layers[0].A = X # input
        for l in range(1, len(self.layers)):
            self.layers[l].feedforward(self.layers[l-1].A)
        Ypred = self.layers[-1].A # output
        return Ypred
    
    def backprop(self, X, Y, Ypred):
        ''' X.shape     == (n_examples, self.layers[0].input_size)
            Y.shape     == (n_examples, self.layers[-1].output_size)
            Ypred.shape == (n_examples, self.layers[-1].output_size)
            where Ypred is the result of feedforward(X)
            
            Note that only calling backprop doesn't actually update the network parameters
        '''
        assert(X.shape[0] == Y.shape[0])
        assert(X.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        assert(Y.shape[1] == self.layers[-1].output_size)
        assert(Ypred.shape == Y.shape)
        
        delta = self.J.deltaL(Y, Ypred, self.layers[-1].g) # delta^L == [dJ/dZ^L]
        self.layers[-1].backprop(dZ=delta)
        for l in reversed(range(1, len(self.layers) - 1)):
            # [dJ/dZ^l == dJ/dA^l . dA^l/dZ^l], note that dJ/dA^l is dJ/dX^{l+1}
            delta = self.layers[l+1].dX * self.layers[l].g.derivative(self.layers[l].A) # delta^l == [dJ/dZ^l]
            self.layers[l].backprop(dZ=delta)
        
        # obs.: we don't backpropagate the input layer since we 
        #       manually set it's activation values A to the network's input X
    
    def __shuffle_X_Y(self, X, Y):
        m = X.shape[0] # == Y.shape[0]
        p = np.random.permutation(m)
        return X[p], Y[p]
    
    def __get_batches(self, X, Y, batch_size, shuffled):
        m = X.shape[0] # == Y.shape[0]
        n_batches = m // batch_size
        if shuffled:
            X, Y = self.__shuffle_X_Y(X, Y)
        return zip(np.array_split(X, n_batches), np.array_split(Y, n_batches))
    
    # test data
    def evaluate(self, X_test, Y_test):
        ''' X_test.shape == (n_test_samples, self.layers[0].input_size)
            Y_test.shape == (n_test_samples, self.layers[-1].output_size)
        '''
        assert(X_test.shape[0] == Y_test.shape[0])
        assert(X_test.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        assert(Y_test.shape[1] == self.layers[-1].output_size)
        
        # loss/cost value for the training set
        Ypred = self.predict(X_test)
        cost = self.J(Y_test, Ypred)
        
        # calculates the values not as one-hot encoded row vectors
        target = np.argmax(Y_test, axis=1)
        prediction = np.argmax(Ypred, axis=1)
        accuracy = (prediction == target).mean()

        return cost, accuracy
    
    # training and validation data
    def train(self, X, Y, X_val, Y_val, n_epochs, batch_size, verbose=True):
        ''' X.shape == (n_training_samples, self.layers[0].input_size)
            Y.shape == (n_training_samples, self.layers[-1].output_size)
            
            X_val.shape == (n_validation_samples, self.layers[0].input_size)
            Y_val.shape == (n_validation_samples, self.layers[-1].output_size)
            
            For each iteration we'll have:
              n_examples = batch_size
              batch_X.shape == (n_examples, self.layers[0].input_size)
              batch_Y.shape == (n_examples, self.layers[-1].output_size)
            Thus, each epoch has ceil(n_training_samples / batch_size) iterations
            obs.: batch_X and batch_Y are rows of X and Y, and after each iteration (i.e. after going through
                  each batch) we update our network parameters (weights and biases)
            
            If n_training_samples is not divisible by batch_size the last training batch will be smaller
        '''
        assert(X.shape[0] == Y.shape[0])
        assert(X.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        assert(Y.shape[1] == self.layers[-1].output_size)
        assert(X_val.shape[0] == Y_val.shape[0])
        assert(X_val.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        assert(Y_val.shape[1] == self.layers[-1].output_size)
        
        n_training_samples = X.shape[0]
        batches_per_epoch = int(np.ceil(n_training_samples / batch_size)) # equal to the number of iterations per epoch
        
        for epoch in range(n_epochs):
            if verbose:
                start_time = time()
                batch_number = 1
                
            for batch_X, batch_Y in self.__get_batches(X, Y, batch_size, shuffled=True):
                # calculates the predicted target values for this batch (with the current network parameters)
                batch_Ypred = self.feedforward(batch_X)
                
                # sets the values of dW and db, used to then update the network parameters
                self.backprop(batch_X, batch_Y, batch_Ypred)
                
                # updates each layer's parameters (i.e. weights and biases) with some flavor of gradient descent
                self.optimizer.update(self.layers)
                
                if verbose:
                    print(f"batch ({batch_number}/{batches_per_epoch})", end='\r')
                    batch_number += 1
            
            # calculate the loss/cost value for this epoch
            epoch_cost, epoch_accuracy = self.evaluate(X, Y) # training set
            epoch_cost_val, epoch_accuracy_val = self.evaluate(X_val, Y_val) # validation set
            self.history["loss"].append(epoch_cost)
            self.history["loss_val"].append(epoch_cost_val)
            self.history["acc"].append(epoch_accuracy)
            self.history["acc_val"].append(epoch_accuracy_val)
            if verbose:
                print(f"epoch ({epoch+1}/{n_epochs}) "
                      f"loss: {epoch_cost:.4f}, loss_val: {epoch_cost_val:.4f} | "
                      f"acc: {epoch_accuracy:.2f}, acc_val: {epoch_accuracy_val:.2f} | "
                      f"Δt: {(time() - start_time):.2f}s")
