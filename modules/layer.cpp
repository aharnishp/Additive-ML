#include<iostream>
#include<vector>

#define telemetry 1
#define fori(i,n) for(int i = 0; i < n; i++)
#define pb push_back


// float is 4 bit 
#define FLOATINGPOINT float

// float is 8 bit 
#define FLOATINGPOINT double



// Acivation Functions
#define Linear_ACTIVATION 0
#define ReLU_ACTIVATION 1
#define Sigmoid_ACTIVATION 2
#define Exponential_ACTIVATION 3


namespace std
{
    
    class nlayer{
    
        public:

        __UINT32_TYPE__ id = 0;

        // shape of this 3D layer
        __UINT16_TYPE__ x;
        __UINT16_TYPE__ y;
        __UINT16_TYPE__ z;

        // vector of pointers storing pointer to input layers
        vector<int*> input_layers(0, nullptr);

        // 3D vector storing weights
        vector<vector<vecctor<float>>> weights;

        // float storing bias
        float bias;

        // int to store activation function of the layer
            // 0 = ReLU
            // 1 = Linear
            // 2 = sigmoid
            // 3 = exponential
        __UINT16_TYPE__ activationFn = 0;


        


        



        


        



    }


} // namespace std


