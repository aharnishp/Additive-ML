#include<vector>

// Compile Time Parameters 
#define Low_Memory_Target 1


// Compile Parameter Code
#ifdef Low_Memory_Target
  // float is 4 bytes
  #define def_float_t float
  #define def_uint_small_t __UINT8_TYPE__
  #define def_uint_t __UINT16_TYPE__
  #define def_int_t __INT16_TYPE__
  
#else
  // float is 8 bytes
  #define def_float_t double
  #define def_uint_small_t __UINT16_TYPE__
  #define def_uint_t __UINT32_TYPE__
  #define def_int_t __INT32_TYPE__
#endif


class nlayer{
      public:

        def_uint_t id = 0;
        def_uint_t layerVersion = 0;
        

        // shape of this 3D layer
        def_uint_t x;
        def_uint_t y;
        def_uint_t z;

        // vector of pointers storing pointer to input layers
        std::vector<nlayer *> input_layers;

        // 3D vector storing weights
        std::vector<std::vector<std::vector<def_float_t>>> weights;

        // float storing bias
        def_float_t bias;

        // int to store activation function of the layer
            // 0 = ReLU
            // 1 = Linear
            // 2 = sigmoid
            // 3 = exponential
        def_uint_t activationFn = 0;
        
        def_float_t learningRate;

        
        
        // caching activations after activation Fn.
        def_uint_small_t cache_init;

        // stores run_id when it this layer's last activation was calculated 
        def_int_t cached_run_id = 0;

        // FIXME:
        // stores the value of last activation
        std::vector<std::vector<std::vector<def_float_t>>> cached_acivation_values;


        // FUTURE:
        // store the variance in error or similar 
        // # Flag indicating if it was being evaluated.
        // ### This can help in case of self loops, when a layer was being evaluated was evaluated again
        //   meaning one of this layer's input_layer has this layer as one of the inputs (called self-loop in a graph).
        //   In this situation, the last cached value of this layer will be returned.

        def_uint_small_t being_evaluated; // when executing
        def_uint_small_t being_corrected; // when training

        def_uint_small_t is_dynamic_layer;
        def_uint_small_t is_input_layer;

        
        nlayer(def_uint_t x, def_uint_t y, def_uint_t z, def_uint_t activationFn, def_float_t learningRate);
};