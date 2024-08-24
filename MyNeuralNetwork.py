import numpy as np

class NeuralNetwork():

    #############################   CORE COMPONENTS OF NEURAL NETWORK   #############################

    def __init__(self, input_size=784, output_size=10, hidden_layers=[256, 256]):
        assert isinstance(hidden_layers, (np.ndarray, list)), f"{hidden_layers} must be a list or ndarray!"
        
        # Initialize instance variables
        self.h_layers = hidden_layers
        self.activations = []
        self.weights = []
        self.biases = []

        # Initialize input-to-hidden layer weights and biases
        self.weights.append(self.__he_vals__([input_size, hidden_layers[0]]))
        self.biases.append(np.zeros(hidden_layers[0]))

        # Initialize hidden layers weights and biases
        for layer in range(len(self.h_layers)-1):
            self.weights.append(self.__he_vals__([hidden_layers[layer], hidden_layers[layer+1]]))
            self.biases.append(np.zeros(hidden_layers[layer+1]))

        # Initialize hidden-to-output layer weights and biases
        self.weights.append(self.__he_vals__([hidden_layers[len(hidden_layers)-1], output_size]))
        self.biases.append(np.zeros(output_size))
    
    def __passForward__(self, activations):
        self.activations = [activations]
        for i in range(len(self.h_layers)+1):
            # Following the formula: z^L = w^L * a^(L-1) + b^L
            activations = np.dot(activations, self.weights[i]) + self.biases[i]
            
            # Save only the z values WITHOUT any activation functions. Will be useful for backpropagation step
            self.activations.append(activations)
            if i<len(self.h_layers):
                # After appending unactivated z value, use the activation function on the inputs
                activations = self.__relu__(activations)

        # Return softmax values (probability distribution)
        return self.__softmax__(self.activations[-1])
    
    def __backprop__(self, labels):
        # Initialize useful variables for batch backpropagation
        batch_size = labels.shape[0]
        grad_biases = []
        grad_weights = []
        
        for i in range(len(self.h_layers), -1, -1):
            if i==len(self.h_layers):
                # Start with first layer, following the formula: delta = a^L - y
                delta = self.__softmax__(self.activations[i+1]) - labels
            else:
                # Calculate delta for rest of hidden layers.
                delta = (self.__d_relu__(self.activations[i+1]))*(np.dot(delta, self.weights[i+1].T))
            
            # B = delta; Batch calculation needs the average of each delta row
            del_cost_del_bias = np.mean(delta, axis=0)

            # W = a^(L-1) * delta; Batch calculation adds / batch_size
            if i!=0:
                del_cost_del_weight = np.dot(self.__relu__(self.activations[i]).T, delta) / batch_size
            else:
                del_cost_del_weight = np.dot(self.activations[i].T, delta) / batch_size

            # Append each gradient step to respective lists
            grad_biases.append(del_cost_del_bias)
            grad_weights.append(del_cost_del_weight)

        # Reverse lists to order gradients from input to output layer
        grad_biases.reverse()
        grad_weights.reverse()

        # Return the lists with all gradient descent steps
        return grad_biases, grad_weights

    def train(self, t_data, t_labels, learning_rate = 0.71, batch_size=5000, epochs=10, shuffle=False, one_hot=False):
        if not one_hot:
            t_labels = self.__one_hotify__(t_labels)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in range(0, len(t_data), batch_size):
                tb_data = t_data[batch:batch+batch_size]
                tb_labels = t_labels[batch:batch+batch_size]

                output = self.__passForward__(tb_data)
                total_loss += self.__calculate_loss__(tb_labels, output) / batch_size
                grad_biases, grad_weights = self.__backprop__(tb_labels)
                self.__update_b_and_w__(grad_biases, grad_weights, learning_rate)
            print(f"Epoch: {epoch+1}:\n\tLoss {total_loss}")
        print("Finished training.")
            
    def predict(self, t_data, t_labels):
        num_correct = 0
        for data, label in zip(t_data, t_labels):
            output = self.__passForward__(data)
            if label==np.argmax(output):
                num_correct+=1
        return f"{(num_correct / t_data.shape[0]):.2%}"

    ############################# END CORE COMPONENTS OF NEURAL NETWORK #############################

    #############################            HELPER METHODS             #############################

    def __he_vals__(self, arr_shape):
        # He initialization method that works well with the ReLU activation function
        # For more, see https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
        return np.random.randn(*arr_shape) * np.sqrt(2 / arr_shape[0])
    
    def __relu__(self, a):
        return np.maximum(0, a)

    def __softmax__(self, a):
        ea = np.exp(a - np.max(a, axis=-1, keepdims=True))
        return ea / np.sum(ea, axis=-1, keepdims=True)
    
    def __d_relu__(self, a):
        return np.where(a>0, 1, 0)

    def __one_hotify__(self, labels, num_classes=10):
        # Convert labels into a one hot 2D array
        return np.eye(num_classes)[labels]

    def __calculate_loss__(self, labels, preds):
        # Calculate loss using Cross Entropy Loss
        return -np.sum(labels * np.log(preds + 1e-10))
    
    def __update_b_and_w__(self, biases, weights, l_rate):
        # For each array in the lists, update the bias/weights using the formula: b - learning_rate * gradient_bias
        self.biases = [sb - l_rate*b for sb, b in zip(self.biases, biases)]
        self.weights = [sw - l_rate*w for sw, w in zip(self.weights, weights)]
    
    #############################          END HELPER METHODS           #############################
    
    #############################           ACCESSOR METHODS            #############################

    def getWeights(self):
        return self.weights
    
    def getBiases(self):
        return self.biases
    
    #############################         END ACCESSOR METHODS          #############################