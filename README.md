# Additive ML

A repository containing past version of the framework written as part of my Bachelor's Thesis. The name Additive ML is a result of the framework's ability to increase the neural network size based on the error trend during training and can change the network size during runtime with minimal overhead.


## ```nlayer``` class
The ```nlayer``` class is the fundamental class which are handled by the ```nnetwork``` class.
The ```nlayer``` class contains array of neurons based which can be upto 3 dimensional,

There are 2 types of ```nlayer``` based on the operation it performs.

### Fully connected layer
The number of neurons is given by x * y * z. 
The neurons are processed as flattened 1D array.

#### input_layers
This is an array which stores the pointer to all the ```nlayer``` instances which are part of this current layer. It is stored in C++ STL vector.
Declaration as 
```cpp
std::vector<nlayer*>;
```

#### weights matrix
The weight matrix stores the weights of the connections to the input_layers of the class.

For computing the forward propagation, all the activations of layers in ```input_layers``` are stacked and then multiplied by this stacked array of activations to get output activation (without the bias and the activation function applied yet.)

**Life cycle of weights matrix**
The weights matrix is stored as flattened STL vector, with row major encoding.

### Convolutional layer
Work in Progress, added types, but not implemented any training method.


### Initialization Technicalities
When a layer is initialized, it has empty weights vector. 2 properties weight_inp and weight_out define the logical dimensions of the 2D weight matrix.

Once a weight has been resized, or an input layer is added to it, the weight matrix is initialized.
The initialization takes place by calling the add_input_layer function, which calls auto_grow_weights. auto_grow_weights calculates the number neurons from it's input layers and output size from its own properties. 


## Compilation

Requires GCC/G++ and is tested on Linux x86_64.

Can be compiled using command with a ```build/``` directory made.

```bash
g++ main.cpp -o build/main -O3 -march=native -mavx && build/main
```
