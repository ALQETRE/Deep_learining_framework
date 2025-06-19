import numpy as np
import tensorflow as tf
from tqdm import tqdm

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# Load MNIST dataset
#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


train_images = train_images.reshape(train_images.shape[0], -1)/255
test_images = test_images.reshape(test_images.shape[0], -1)/255

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)


print(train_images.shape)  # (60000, 784)
print(train_labels.shape)  # (60000, 10)

## Activation Functions ##

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def softmax(x):
    x = np.clip(x, -500, 500)  # Prevent overflow
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return exps / np.sum(exps, axis=1, keepdims=True)

def d_softmax(x):
    s = softmax(x)
    jacobian = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        s_i = s[i].reshape(-1, 1)
        jacobian[i] = np.diagflat(s_i) - np.dot(s_i, s_i.T)
    return jacobian


ac_funcs = {
    "sigmoid": [sigmoid, d_sigmoid],
    "relu": [relu, d_relu],
    "softmax": [softmax, None]
}

## Loss Functions ##

def mse_loss(predictions, targets):
    return np.mean(np.clip(predictions - targets, -10000, 10000) ** 2)

def d_mse_loss(predictions, targets):
    return 2 * (predictions - targets) / targets.size

def ce_loss(predictions, targets):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.mean(np.sum(targets * np.log(predictions), axis=1))


def d_ce_loss(predictions, targets):
    return (predictions - targets) / targets.shape[0]



def success_func(predictions, targets):
    predicted_classes = np.argmax(predictions, axis=1)
    target_classes = np.argmax(targets, axis=1)
    accuracy = np.mean(predicted_classes == target_classes)
    return accuracy * 100

class Layer:
    def __init__(self, num_inputs: int, num_neurons: int, ac_func= None):
        self.weights = (np.random.rand(num_inputs, num_neurons) - 0.5) * 0.1
        self.biases = np.zeros((1, num_neurons))

        self.using_af = ac_func is not None
        self.ac_func_name = ac_func
        if self.using_af:
            self.activation_function = ac_funcs[ac_func][0]
            self.d_activation_function = ac_funcs[ac_func][1]
        else:
            self.activation_function = None
            self.d_activation_function = None

        self.z = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        z = np.dot(inputs, self.weights) + self.biases
        self.z = z

        if self.using_af:
            optput = self.activation_function(z)
        else:
            optput = z
    
        return optput

    def backward(self, d_output, learning_rate):
        if self.using_af:
            # print("Z:\n", self.z, "\n")
            # print("d_output:\n", d_output, "\n")
            if self.d_activation_function is not None:
                d_output *= self.d_activation_function(self.z)


        batch_num = len(d_output)
        d_weights = np.dot(self.inputs.transpose(), d_output)/batch_num
        d_biases = np.sum(d_output, axis=0, keepdims=True)/batch_num
        d_inputs = np.dot(d_output, self.weights.transpose())

        self.weights -= d_weights * learning_rate
        self.biases -= d_biases * learning_rate

        return d_inputs
    
    def save(self):
        layer_data = [self.weights, self.biases, self.ac_func_name]
        return layer_data
    
    def load(self, layer_data):
        self.weights = layer_data[0]
        self.biases = layer_data[1]

        self.ac_func_name = str(layer_data[2])
        self.using_af = self.ac_func_name != "None"
        if self.using_af:
            self.activation_function = ac_funcs[self.ac_func_name][0]
            self.d_activation_function = ac_funcs[self.ac_func_name][1]
        else:
            self.activation_function = None
            self.d_activation_function = None
    
class Network:
    def __init__(self, layers: list, activation_functions: list, num_inputs: int):
        self.layers = []
        for num_neurons, ac_func in zip(layers, activation_functions):
            self.layers.append(Layer(num_inputs, num_neurons, ac_func))
            num_inputs = num_neurons

        self.layer_num = layers
        self.ac_func = activation_functions
            
    def save(self, name):
        data = []
        for layer in self.layers:
            for layer_data in layer.save():
                data.append(layer_data)
        data.append(self.layer_num)
        data.append(self.ac_func)
        np.savez(f"{name}.npz", *data)

    def load(self, name):
        npz_obj = np.load(f"{name}.npz", allow_pickle= True)
        data = []
        for key in npz_obj.files:
            data.append(npz_obj[key])
        if not np.array_equal(self.layer_num, data[-2]):
            raise EnvironmentError(f"Model architecture missmatch {data[-2]} != {self.layer_num} <- incorrect")
        if not np.array_equal(self.ac_func, data[-1]):
            raise EnvironmentError(f"Model activation functions missmatch {data[-1]} != {self.ac_func} <- incorrect")
        for i, layer in enumerate(self.layers):
            layer.load(data[i*3:i*3+3])

    def save_question(self):
        ans = input("Save (Y/n): ").lower()
        while not (ans == "y" or ans == "n"):
            ans = input("Save (Y/n): ").lower()
        if ans == "y":
            name = input("Model name: ")
            self.save(name)

    def get_batch(self, X, y, batch_size):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            yield X_batch, y_batch
    
    def learn(self, X, y, batch_size, model, learning_rate, loss_function, d_loss_function, num_gens=1, bar= True, update_bar= False):
        if model != "current":
            self.load(model)
        if bar:
            self.pbar = tqdm(total=len(X)*num_gens, desc=f"Learning: (Loss - ---, gen 0/{num_gens})")
        for gen in range(num_gens):
            for X_batch, y_batch in self.get_batch(X, y, batch_size):
                output = X_batch
                for layer in self.layers:
                    output = layer.forward(output)
                loss = loss_function(output, y_batch)
                d_output = d_loss_function(output, y_batch)
                for layer in self.layers[::-1]:
                    d_output = layer.backward(d_output, learning_rate)
                if bar:
                    self.pbar.update(batch_size)
                    self.pbar.desc = f"Learning: (Loss - {round(loss*100, 3)}, gen - {gen+1}/{num_gens})"
                elif update_bar:
                    self.pbar.update(batch_size)
        if bar:
            self.pbar.close()
        
    def test(self, X, y, batch_size, model, loss_function, bar= True, update_bar= False):
        if model != "current":
            self.load(model)
        if bar:
            self.pbar = tqdm(total=len(X), desc="Testing: (Success - --%)")
        total_success = 0
        total_loss = 0
        batches = 0
        for X_batch, y_batch in self.get_batch(X, y, batch_size):
            output = X_batch
            for layer in self.layers:
                output = layer.forward(output)
            total_success += success_func(output, y_batch)
            total_loss += loss_function(output, y_batch)
            batches += 1
            if bar:
                self.pbar.update(batch_size)
                self.pbar.desc = f"Testing: (Success - {total_success/batches}%)"
            elif update_bar:
                self.pbar.update(batch_size)

        if bar:
            self.pbar.close()
        return total_success/batches, total_loss/batches

    def learn_adaptive_lr(self, X, y, X_testing, y_testing, batch_size, model, loss_function, d_loss_function, num_gens=20, lr_slope= 2.5, moving_avg_num= 3):
        if model != "current":
            self.load(model)

        success, loss = self.test(X_testing, y_testing, 16, "current", loss_function, bar= False)

        best_loss = 100000
        best_gen = 0
        lr = 0.5
        self.pbar = tqdm(total=len(X)*num_gens, desc=f"Learning: (Success - {round(success, 3)}%, Learning rate - {round(lr, 4)}, Loss - {round(loss, 5)}, gen 0/{num_gens})")
        moving_avg_list = [2.3]*moving_avg_num
        for gen in range(num_gens):
            
            if loss < best_loss:
                self.save("cache")
                best_loss = loss
                best_gen = gen

            self.learn(X, y, batch_size, "current", lr, loss_function, d_loss_function, num_gens= 1, bar= False, update_bar= True)
            success, loss = self.test(X_testing, y_testing, 16, "current", loss_function, bar= False)

            moving_avg_list.append(loss)
            moving_avg_list.pop(0)

            moving_avg = sum(moving_avg_list)/moving_avg_num
            lr = (moving_avg) * lr_slope

            self.pbar.desc = f"Learning: (Success - {round(success, 3)}%, Learning rate - {round(lr, 4)}, Loss - {round(loss, 5)}, gen - {gen+1}/{num_gens})"
        if loss > best_loss:
            self.load("cache")
            success, loss = self.test(X_testing, y_testing, 16, "current", loss_function, bar= False)
            self.pbar.desc = f"Learning: (Success - {round(success, 3)}%, Learning rate - ---, Loss - {round(loss, 5)}, gen - {best_gen}/{num_gens}, rolled back)"
        self.pbar.close()


network = Network([512, 256, 10], ["relu", "relu", "softmax"], 784)

network.learn_adaptive_lr(train_images, train_labels, test_images, test_labels, 32, "current", ce_loss, d_ce_loss, 10, 1.5, 4)

#network.test(test_images, test_labels, 16, "v1-fashion", success_func)

network.save_question()