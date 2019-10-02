import numpy as np

### Activations
class ActivationFunction:
    def __call__(self, Z):
        raise NotImplementedError    
    def derivative(self, A):
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    def __call__(self, Z):
        return 1 / (1 + np.exp(-Z))
    def derivative(self, A):
        return A * (1 - A) # Sigmoid(Z) * (1 - Sigmoid(Z))

### Costs
class CostFunction:
    def __call__(self, Y, Ypred):
        raise NotImplementedError
    def derivative(self, Y, Ypred):
        raise NotImplementedError

class CrossEntropy(CostFunction):
    def __call__(self, Y, Ypred):
        return np.mean( -(Y * np.log(Ypred)).sum(axis=1) )
    def derivative(self, Y, Ypred):
        # FIXME
        raise NotImplementedError
        #return Y / Ypred
        #return (Ypred - Y) / Ypred.shape[0]

class MSE(CostFunction):
    def __call__(self, Y, Ypred):
        return (Y - Ypred) ** 2
    def derivative(self, Y, Ypred):
        return (Y - Ypred)

### Layers
class Layer:
    def __init__(self, activation, layer_size,
                 weight_initialization='xavier'):
        
        self.weight_initialization = weight_initialization

        self.input_size = None
        self.output_size = layer_size

        self.next_layer = None
        self.previous_layer = None
        
        self.activation = activation

        # activation values shape=(output_size, 1)
        self.z = np.array([]) # prev_layer.a @ self.W + self.b
        self.a = np.array([]) # self.activation(self.z)
        self.delta = np.array([])
        self.delta_weights = np.array([])

        self.errors = np.array([])

    def initialize_weights(self, W=np.array([]), b=np.array([])):
        if self.previous_layer != None: # if not first layer
            self.input_size = self.previous_layer.output_size
            if W.any() and b.any(): # use passed values
                print( (self.input_size, self.output_size) )
                assert(W.shape == (self.input_size, self.output_size))
                assert(b.shape[0] == self.output_size or b.shape[1] == self.output_size)
                self.W = W
                self.b = b
            else:
                if self.weight_initialization == 'xavier':
                    stddev = np.sqrt(1 / input_size)
                    self.W = stddev * np.random.randn(input_size, output_size)
                    self.b = np.random.randn(output_size, 1)
                elif self.weight_initialization == 'xavier avg':
                    stddev = np.sqrt(2 / (input_size + output_size))
                    self.W = stddev * np.random.randn(input_size, output_size)
                    self.b = np.random.randn(output_size, 1)
                elif self.weight_initialization == '-1 to 1':
                    self.W = 2 * np.random.randn(input_size, output_size) - 1
                    self.b = 2 * np.random.randn(output_size, 1) - 1
                else:
                    raise ValueError(f"Invalid weight_initialization value: '{weight_initialization}'")
    
    @property
    def params_count(self):
        return self.W.size + self.b.size
    
    def feedforward(self, input_data=None):
        assert(self.previous_layer.a.any())
        assert(self.previous_layer.a.shape[0] == self.input_size)

        self.z = self.previous_layer.a @ self.W + self.b
        self.a = self.activation(self.z)
    
    def calculate_error(self, cost_function, Y):
        self.errors = cost_function(self.a, Y)

    def backprop(self):
        assert(self.a.any()) # the feed foward has not been run

        if self.next_layer == None: # output layer
            self.delta = (self.errors * self.activation.derivative( self.a ))
            self.delta_weights = (self.previous_layer.a).dot(self.delta.T)
        else: # hidden layer
            self.delta = (self.next_layer.delta @ self.W) * self.activation.derivative(self.z)
            self.delta_weights = self.a @ self.delta
    
    def __update_params(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db



### Neural Net
class NN:
    def __init__(self, cost_function):#, cost_function, cost_function_derivative):
        self.layers = []
        self.cost_function = cost_function
    
    def add_layer(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].previous_layer = self.layers[-2]
            self.layers[-2].next_layer = self.layers[-1]
    
    def initialize(self):
        for layer in self.layers:
            layer.initialize_weights()

    
    def print_net(self):
        print("Number of layers:", len(self.layers))
        for i in range(len(self.layers)):
            print("Layer {}: {} neurons".format(i, self.layers[i].output_size ))
    
    def feed_forward(self, X):
        L = len(self.layers)
        self.layers[0].a = X # input
        for l in range(1, L):
            self.layers[l].feedforward()
        y_pred = self.layers[L-1].a # output
        return y_pred
    
    def back_propagation(self, Y):
        L = len(self.layers)-1
        self.layers[L].calculate_error(self.cost_function, Y) # set errors for last layer
        for l in range(L, 0, -1):
            self.layers[l].backprop()

    def train(self, optimizer):
        raise NotImplementedError


weights1 = np.array(
           [[.15, .20],
            [.25, .30]])

weights2 = np.array(
           [[.40, .45],
            [.50, .55]])

WEIGHTS = np.array([weights1, weights2])

biases = np.array(
         [[.35, .35],
          [.60, .60]])

input  = np.array([.05, .10])
output = np.array([.01, .99])

fwd_correct = [ 
            [0.05, 0.1], 
            [0.593269992, 0.596884378],
            [0.75136507,  0.772928465] ]


# Init nn
net = NN( MSE() )
net.add_layer( Layer(Sigmoid(), 2) )
net.add_layer( Layer(Sigmoid(), 2) )
net.add_layer( Layer(Sigmoid(), 2) )

# net.initialize()
net.layers[1].initialize_weights(weights1, biases[0])
net.layers[2].initialize_weights(weights2, biases[1])

net.print_net()

# Foward
fwd = net.feed_forward(input)
for l in net.layers:
    print(l.a)
print()

# Back-prop
net.back_propagation(fwd_correct[-1])
for l in net.layers[1:]:
    print("delta",l.delta)
    print("delta_weights",l.delta_weights)
    print()

print( "\nnew weights 2 (output layer)\n", weights2 - (0.5 * net.layers[2].delta_weights.T) )
print( "\nnew weights 1 (hidden layer)\n", weights1 - (0.5 * net.layers[1].delta_weights.T) )
print()