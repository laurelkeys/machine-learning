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
    
    # receives the derivative of J w.r.t. the next layer's of the current layer
    # returns the derivative of the cost function w.r.t. activation value of the current layer
    # def backprop(self, last_delta):
    #     if last_delta == None: # output layer
    #         self.delta = (errors * self.activation.derivative( results[-1] ))
    #         result = [ results[-2].dot(deltas[0].T) ]

    #     else: # hidden layer
    #         delta = last_delta * self.g_prime(self.z) # self.g_prime(self.z)
    #         delta = delta * self.g_prime()
    #     return delta



### Neural Net
class NN:
    def __init__(self):#, cost_function, cost_function_derivative):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].previous_layer = self.layers[-2]
            self.layers[-2].next_layer = self.layers[-1]
    
    def print_net(self):
        print("Number of layers:", len(self.layers))
        for i in range(len(self.layers)):
            print("Layer {}: {} neruons".format(i, self.layers[i].output_size ))
    
    def feed_forward(self, X):
        L = len(self.layers)
        self.layers[0].a = X # input
        for l in range(1, L):
            self.layers[l].feedforward()
        y_pred = self.layers[L-1].a # output
        return y_pred


        

def forward(INPUT, weights, biases, activation_function):
    results = [INPUT]
    for weight, bias in zip(weights, biases):
        results.append( activation_function( (results[-1] @ weight) + bias ) )
    return results


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

# Init nn
net = NN()
net.add_layer( Layer(Sigmoid(), 2) )
net.add_layer( Layer(Sigmoid(), 2) )
net.add_layer( Layer(Sigmoid(), 2) )

net.layers[1].initialize_weights(weights1, biases[0])
net.layers[2].initialize_weights(weights2, biases[1])

net.print_net()


fwd = net.feed_forward(input)
for l in net.layers:
    print(l.a)
print("fwd",fwd)


results_correct = [ 
            [0.05, 0.1], 
            [0.593269992, 0.596884378],
            [0.75136507,  0.772928465] ]


# bwd = backward(results_correct, WEIGHTS, output, logistic_derivative)
# print("bwd\n",bwd)

# print( "\nnew weights 2 (output layer)\n", weights2 - (0.5 * bwd[0].T) )
# print( "\nnew weights 1 (hidden layer)\n", weights1 - (0.5 * bwd[1].T) )
# print()