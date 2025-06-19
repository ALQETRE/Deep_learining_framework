from deep_learning import *

train_images, train_labels, test_images, test_labels = get_mnist_number_dataset() # Loading dataset
network = Network([512, 256, 10], ["relu", "relu", "softmax"], 784) # Defining network

network.learn_adaptive_lr(train_images, train_labels, test_images, test_labels, 32, "current", "ce", 10, 1.5, 4) # Learning
network.save_question() # Ask's if you want to save the current network