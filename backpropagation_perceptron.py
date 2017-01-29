from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    def __sigmoid(self, x):
        # the weighted sum of the inputs is passed to the sigmoid function
        # to normalize them between 0 and 1
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        # the derivative of the sigmoid curve indicates the degree of 
        # confidence in the current weight
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, iterations):
        # train the neural network through trial and error
        # adjusting the weights each time
        for i in xrange(iterations):
            # pass the training set through our neural network (a signle neuron)
            output = self.think(training_set_inputs)

            # calc the difference between the desired output and
            # the predicted output
            error = training_set_outputs - output

            # multiply the error by the input and again by the gradient of the
            # sigmoid curve. this means less confident weights are adjusted
            # more. Also, inputs that are zero do not cause changes to weights
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            # the .T function transposes the matrix from horizontal to vertical 

            self.synaptic_weights += adjustment

    def think(self, inputs):
        # pass inputs through our (single neuron) neural network
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # training set
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    #train network, n iterations
    n = 10000
    neural_network.train(training_set_inputs, training_set_outputs, n)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))

