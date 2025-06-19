# Neural Network Framework
### This project enables to easily make and train simple MLPs

## Getting Started
### Step 1. Choosing a dataset:
You will need a so called ***dataset*** this is a big collection of training examples with correct answers.
Also you will need a validation/testing dataset, this will be used to test the network accuracy.

In the code we provide two dataset options:
1. MNIST 28x28 number recoginition
2. MNIST 28x28 fashion items recognition

But feel free to add your own, just use ***ONE HOT ENCODING** for the lables/answers.

### Step 2. Defining + creating network architecture:
Create a variable that your network will be saved in.

Arguments:
1. A list on the layer sizes starting from the first hidden layer
2. A list of activation functions in strings
3. An INT of how many inputs are there
~~~
my_network = Network([32, 16, 10], ["sigmoid", "relu", "softmax"], 784)
~~~
Activation functions options:
1. "relu"
2. "sigmoid"
3. "softmax"
4. None

### Step 3. Training with adaptive lr:
You could train the network with the ```learn``` method, but this is not very efficient, instead use the ```learn_adaptive_lr``` method, this changes the learning rate during training, so you get better and faster results.

Arguments:
1. Your training dataset
2. Your training labels/answers
3. Your testing dataset
4. Your testing labels/answers
5. Batch size
6. String for your model name, if it is set to ```"current"```, then no network will be loaded. (The .nzp will be added automaticly)
7. Your loss function
8. The derivate of your loss function
9. How many generations
10. Make the number higher if the learning rate stalls and the success is not optimal. If it rolls back lowwer this number.
11. How many generations will it look to determine the learning rate
~~~
my_network.learn_adaptive_lr(
  train_images,
  train_labels,
  test_images,
  test_labels,
  32,
  "current",
  ce_loss,
  d_ce_loss,
  num_gens= 10,
  lr_slope= 1.5,
  moving_avg_num= 4
)
~~~
