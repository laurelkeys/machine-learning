import numpy as np
from math import e


def forward(INPUT, weights, biases, activation_function):
    results = [INPUT]
    for weight, bias in zip(weights, biases):
        results.append( activation_function( (results[-1] @ weight) + bias ) )
    return results


def backward(results, weights, targets, activation_derivative):
    results = np.array(results).reshape(len(results), len(results[0]), 1)
    errors = results[-1] - targets.reshape(len(targets),1)

    # Output layer
    deltas = [ (errors * activation_derivative(results[-1])) ]
    result = [ results[-2].dot(deltas[0].T) ]

    # Hidden layers
    for i in range(len(weights)-2, -1, -1):
        weight, input_results, output_results  = weights[i], results[i].T, results[i+1].T

        # print( "weight", weight.shape )
        # print( "delta", deltas[-1].shape )

        # print( "W . delta", (weight @ deltas[-1]).shape )
        # print( "activations", activation_derivative(output_results).shape )

        deltas.append( (weight @ deltas[-1]).T * activation_derivative(output_results) )

        # print( "new delta", deltas[-1].shape )
        # print( "input", input_results.shape )
        result.append( input_results.T @ deltas[-1] )

        # print( "result", result[-1].shape )
    return result


def logistic(x):
    return 1 / (1 + (e ** (-x)))

def logistic_derivative(y):
    return y * (1 - y)

# def relu(x):
#     # return 0 if x < 0 else x
#     return x[x < 0] = 0
#     # return np.maximum(0,x)

# def relu_derivative(x):
#     # return int(x > 0)
#     return np.where(x[x == 0], 0, 1) # quicker?

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


fwd = forward(input, WEIGHTS, biases, logistic)
print("fwd",fwd)


results_correct = [ 
            [0.05, 0.1], 
            [0.593269992, 0.596884378],
            [0.75136507,  0.772928465] ]


bwd = backward(results_correct, WEIGHTS, output, logistic_derivative)
print("bwd\n",bwd)

print( "\nnew weights 2 (output layer)\n", weights2 - (0.5 * bwd[0].T) )
print( "\nnew weights 1 (hidden layer)\n", weights1 - (0.5 * bwd[1].T) )
print()