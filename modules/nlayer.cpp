#include "nlayer.hpp"

#define telemetry 1
#define fori(i,n) for(int i = 0; i < n; i++)
#define pb push_back

// Compile Time Parameters 


// Settings
#define TELEMETRY 1
#define DEFAULT_LAYER_VERSION 1
#define INITIAL_LEARNING_RATE 0.05


#ifdef TELEMETRY
  #include<iostream>
  #define TELEMETRY(x) cout << x << endl;
#else
  #define TELEMETRY(x)
#endif





// Static Defines Below

// cached_run_states
#define un_initiated_cache_state 2
#define initiated_cache_state 1

// Activation Functions
#define Linear_activation_state 0
#define ReLU_activation_state 1
#define Sigmoid_activation_state 2
#define Exponential_activation_state 3


namespace std
{
    
    class nlayer
    {
    
      public:

        def_uint_t id = 0;
        def_uint_t layerVersion = 0;
        

        // shape of this 3D layer
        def_uint_t x;
        def_uint_t y;
        def_uint_t z;

        // vector of pointers storing pointer to input layers
        vector<nlayer*> input_layers(0, nullptr);

        // 3D vector storing weights
        vector<vector<vector<def_float_t>>> weights;

        // float storing bias
        def_float_t bias;

        // int to store activation function of the layer
            // 0 = ReLU
            // 1 = Linear
            // 2 = sigmoid
            // 3 = exponential
        def_uint_t activationFn = 0;
        
        def_float_t learningRate = INITIAL_LEARNING_RATE;

        
        
        // caching activations after activation Fn.
        def_uint_small_t cache_init = un_initiated_cache_state;

        // stores run_id when it this layer's last activation was calculated 
        def_int_t cached_run_id = 0;

        // FIXME:
        // stores the value of last activation
        vector<vector<vector<def_float_t>>> cached_acivation_values;


        // FUTURE:
        // store the variance in error or similar 
        // # Flag indicating if it was being evaluated.
        // ### This can help in case of self loops, when a layer was being evaluated was evaluated again
        //   meaning one of this layer's input_layer has this layer as one of the inputs (called self-loop in a graph).
        //   In this situation, the last cached value of this layer will be returned.

        def_uint_small_t being_evaluated = 0; // when executing
        def_uint_small_t being_corrected = 0; // when training

        def_uint_small_t is_dynamic_layer = 1;
        def_uint_small_t is_input_layer = 1;

        
        void nlayer(def_uint_t x, def_uint_t y, def_uint_t z, def_uint_t activationFn, def_float_t learningRate)
        {
            this->x = x;
            this->y = y;
            this->z = z;
            this->activationFn = activationFn;
            this->learningRate = learningRate;
            this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }
        
        void nlayer(def_uint_t x, def_uint_t y, def_uint_t z, def_uint_t activationFn)
        {
            this->x = x;
            this->y = y;
            this->z = z;
            this->activationFn = activationFn;
            this->learningRate = INITIAL_LEARNING_RATE;
            this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }

        void nlayer(def_uint_t x, def_uint_t activationFn, def_float_t learningRate)
        {
            this->x = x;
            this->y = 1;
            this->z = 1;
            this->activationFn = activationFn;
            this->learningRate = learningRate;
            this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }

        void nlayer(def_uint_t x, def_uint_t y, def_uint_t z, def_uint_t is_input_layer)
        {
            this->x = x;
            this->y = y;
            this->z = z;
            this->activationFn = ReLU_activation_state;
            this->learningRate = INITIAL_LEARNING_RATE;
            this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 0;
        }


        def uint_t get_id()
        {
            return this->id;
        }

        def_uint_t get_layerVersion()
        {
            return this->layerVersion;
        }
        
        def_uint_t add_input_layer(nlayer* input_layer)
        {
            this->input_layers.pb(input_layer);
            return this->input_layers.size();
        }

        // pass the vector of cached values 
        vector<vector<vector<def_float_t>>> get_cached_activation_values()
        {
            return this->cached_acivation_values;
        }

        def_int_t add_input_layer(nlayer * new_layer)  // success = 0, failure = -1
        {
            // check if it is not same as this
            this->input_layers.push_back(new_layer);
        }
        



    }


} // namespace std


