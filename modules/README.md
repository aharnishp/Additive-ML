# Modules

# ```nlayer``` class
The ```nlayer``` class is the fundamental class which are handled by the ```nnetwork``` class.
The ```nlayer``` class contains array of neurons based which can be upto 3 dimensional,

There are 2 types of ```nlayer``` based on the operation it performs.

## Fully connected layer
The number of neurons is given by x * y * z. 
The neurons are processed as flattened 1D array.

### input_layers
This is an array which stores the pointer to all the ```nlayer``` instances which are part of this current layer. It is stored in C++ STL vector.
Declaration as 
```cpp
std::vector<nlayer*>;
```

### weights matrix
The weight matrix stores the weights of the connections to the input_layers of the class.

For computing the forward propagation, all the activations of layers in ```input_layers``` are stacked and then multiplied by this stacked array of activations to get output activation (without the bias and the activation function applied yet.)

**Life cycle of weights matrix**
The weights matrix is stored as flattened STL vector, with row major encoding.

## Convolutional layer
