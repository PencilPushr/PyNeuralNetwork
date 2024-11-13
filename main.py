import numpy as py
import random as rand
import gzip as gzip
import numpy as np
import idx2numpy

def openZip():

    f = gzip.open('source.gz', 'r')
    arr = idx2numpy.convert_from_file(f)
    # arr is now a np.ndarray type of object of shape 60000, 28, 28

    # Defunct old way of doing the above:
    #f = gzip.open('source.gz','r')
    #image_size = 28
    #num_images = 9
    #f.read(16)
    #buf = f.read(image_size * image_size * num_images)
    #data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = data.reshape(num_images, image_size, image_size, 1)

def sigmoid(input):
	return 1.0 / (1.0 + py.exp(-input))

def transfer_derivative(input):
	return input * (1.0 - input)

class Perceptron:
    def __init__(self,inputnumber):
        self.weights = [0.0]*inputnumber
        self.biasweight = rand.uniform(-0.5, 0.5)
        # randomly assign weight values
        for i in range (0,inputnumber):
            self.weights[i] = rand.uniform(-0.5, 0.5)
    def process (self,inputs):
        total = 0.0
        bias = self.biasweight
        for i in range (0, len(self.weights)):
            total = total + inputs[i]*self.weights[i]
        return sigmoid(total+bias)
    def getWeights(self):
        return self.weights.copy()
    def train(self,input,weightIndex,learning_rate,delta):
        change = (delta*input*learning_rate)
        self.weights[weightIndex] = self.weights[weightIndex] - change
    def trainBias(self,learning_rate,delta):
        change = (delta*learning_rate)
        self.biasweight = self.biasweight - change

class MultiLayer:

    # This determines the MLP. inputNum is the number of inputs (XOR would have two, for instance), and architecture is an array of integers dictating the size of each layer EXCEPT the last layer. The last layer always has just 1 node.
    def __init__(self,inputNum,architecture):
        self.neurons = []
        self.inputnum = inputNum
        for i in range (len(architecture)):
            layer = []
            if i == 0:#first layer, directly taking inputs
                for x in range(architecture[i]):
                    layer.append(Perceptron(inputNum))
            else:
                for x in range(architecture[i]):
                    layer.append(Perceptron(architecture[i-1]))
            self.neurons.append(layer)
        #add the final, 1-element layer
        self.neurons.append([Perceptron(architecture[len(architecture)-1])])

    # This takes an array of inputs (which must match the appropriate value) and returns a single result after processing them through every layer.
    def processInputs(self,inputs):
        results = []
        for i in range (len(self.neurons)):
            layer = []
            if i == 0:#first layer, directly taking inputs
                for x in range(len(self.neurons[i])):
                    layer.append(self.neurons[i][x].process(inputs))
            else:
                for x in range(len(self.neurons[i])):
                    layer.append(self.neurons[i][x].process(results[i-1]))
            results.append(layer)
        # Now, return the value from the only neuron in the final layer.
        return results[len(results)-1][0]

    # Uses back-propagation to correct a network with the known proper output
    def training(self,inputs,expected):
        results = []
        errors = []
        deltas = []
        for i in range (len(self.neurons)):
            layer = []
            deltaLayer = []
            if i == 0:#first layer, directly taking inputs
                for x in range(len(self.neurons[i])):
                    layer.append(self.neurons[i][x].process(inputs))
                    deltaLayer.append(None)
            else:
                for x in range(len(self.neurons[i])):
                    layer.append(self.neurons[i][x].process(results[i-1]))
                    deltaLayer.append(None)
            results.append(layer)
            errors.append(None)
            deltas.append(deltaLayer)
        # Now, compare result to expected
        output = results[len(results)-1][0]
        # Fill in the error at the end.
        errors[len(errors)-1] = (output - expected)
        deltas[len(self.neurons) -1][0] = errors[len(errors)-1] * transfer_derivative(output)
        #now go through each previous layer
        for i in reversed(range(len(self.neurons) - 1)):
            errors[i] = 0.0
            for x in range(len(self.neurons[i])):
                #multiply the delta of the next layer nueron by the weight it gives
                #the current neuron, to get the error amount, then add error amount

                # go through the next-layer neurons
                for j in range(len(deltas[i+1])):
                    output = deltas[i+1][j]
                    errors[i] += (self.neurons[i+1][j].getWeights()[x])*output
                deltas[i][x] = errors[i] * transfer_derivative(results[i][x])

        #Now we have our "deltas", we can train the network, adjusting weights
        #weight = weight - learning_rate * delta * input

        learning_rate = 0.2
        for i in range (len(self.neurons)):
            if i == 0:#first layer, directly taking inputs
                for j in range(len(self.neurons[i])):
                    delta = deltas[i][j]
                    self.neurons[i][j].trainBias(learning_rate,delta)
                    for k in range(len(inputs)):
                        input = inputs[k]
                        self.neurons[i][j].train(input,k,learning_rate,delta)
            else:
                for j in range(len(self.neurons[i])):
                    delta = deltas[i][j]
                    self.neurons[i][j].trainBias(learning_rate,delta)
                    for k in range(len(self.neurons[i-1])):
                        input = results[i-1][k]
                        self.neurons[i][j].train(input,k,learning_rate,delta)            

def train(network,inputs,outputs):
    if (len(inputs) != len(outputs)):
        print("YOU HAVE TRESPASSED AGAINST HEAVEN.")
        return
    for i in range(10000):
        for x in range(len(inputs)):
            network.training(inputs[x],outputs[x])


    
def test(network,inputs,outputs):
    if (len(inputs) != len(outputs)):
        print("YOU HAVE TRESPASSED AGAINST HEAVEN.")
        return
    successes = 0;
    failures = 0;
    for x in range(len(inputs)):
        if (round(network.processInputs(inputs[x])) == round(outputs[x])):
            successes = successes+1
        else:
            failures = failures+1
    print("There were "+successes+", and "+failures+" for this process.")

def main():
    print("Starting Neural Network Program to recognise drawn bitmap numbers.")
    print("--->Building Network...")
    # 29x28 bitmap square, 784 inputs
    # Then two layers of 10
    network = MultiLayer(784,[10,10])
    print("--->Network Built.")
    print("------>Beginning basic tests.")
    print("--->Testing Input: True, True.")
    outcome = network.processInputs([1.0,1.0])
    print("--->Outcome: "+str(outcome))
    print("--->Testing Input: False, True.")
    outcome = network.processInputs([0.0,1.0])
    print("--->Outcome: "+str(outcome)+"")
    print("--->Testing Input: True, False.")
    outcome = network.processInputs([1.0,0.0])
    print("--->Outcome: "+str(outcome)+"")
    print("--->Testing Input: False, False.")
    outcome = network.processInputs([0.0,0.0])
    print("--->Outcome: "+str(outcome)+"")
    print("------>All basic tests complete!")

    #having done XOR, try AND or other functions.

    print("------>That was an untrained, naive test.")
    print("------>We will train it and try again.")

    #XOR version
    #for i in range(10000):
        #print("--->Iteration: "+str(i+1)+".")
        #network.training([1.0,1.0],0.0)#XOR(T,T)
        #network.training([0.0,1.0],1.0)#XOR(F,T)
        #network.training([1.0,0.0],1.0)#XOR(T,F)
        #network.training([0.0,0.0],0.0)#XOR(F,F)
    #print("------>Training complete.\n")

    #AND version
    for i in range(10000):
        network.training([1.0,1.0],1.0)#AND(T,T)
        network.training([0.0,1.0],0.0)#AND(F,T)
        network.training([1.0,0.0],0.0)#AND(T,F)
        network.training([0.0,0.0],0.0)#AND(F,F)
    print("------>Training complete.")
    
    print("------>Beginning trained tests:")
    print("--->Testing Input: True, True.")
    outcome = network.processInputs([1.0,1.0])
    print("--->Outcome: "+str(outcome))
    print("--->Testing Input: False, True.")
    outcome = network.processInputs([0.0,1.0])
    print("--->Outcome: "+str(outcome))
    print("--->Testing Input: True, False.")
    outcome = network.processInputs([1.0,0.0])
    print("--->Outcome: "+str(outcome))
    print("--->Testing Input: False, False.")
    outcome = network.processInputs([0.0,0.0])
    print("--->Outcome: "+str(outcome))
    print("------>All trained tests complete!")
    
    print("Ending Neural Network Program.")

if __name__ == "__main__":
    main()
