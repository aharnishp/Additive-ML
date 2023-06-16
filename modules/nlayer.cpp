#include "nlayer.hpp"

#include<cstring>
#include<cblas.h>

#define telemetry 1
#define fori(i,n) for(int i = 0; i < n; i++)
#define pb push_back

//// Compile Time Parameters 

// Settings
#define TELEMETRY 1
#define DEFAULT_LAYER_VERSION 1
#define INITIAL_LEARNING_RATE 0.05


#ifdef TELEMETRY
  #include<iostream>
  #define print_telm(x) cout << x << endl;
#else
  #define print_telm(x)
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


// Layer types
#define Fully_Connected_INPUTS 0
#define Convolutional_INPUTS 1

def_float_t get_rand_float(){ srand(time(0)); return ( (float)(rand()) / (float)(RAND_MAX) ); }
def_float_t get_rand_float_seeded(unsigned int seed){ srand(seed); return ( (float)(rand()) / (float)(RAND_MAX) ); }



namespace std
{
    class nlayer
    {

    private:
        // if the layer type is linear fully connected, then weight is 2D dimensional
        // def_float_t** weights = NULL;
        def_float_t* weights_allocator = NULL;
        def_uint_t allocated_weight_x = 0; def_uint_t allocated_weight_y = 0;

    public:

        // general info
        def_uint_t id = 0;
        def_uint_t layer_type = Fully_Connected_INPUTS;
        // def_uint_t layerVersion = 0;

        // shape of this 3D layer
        def_uint_t x;
        def_uint_t y;
        def_uint_t z;


        // vector of pointers storing pointer to input layers
        vector<nlayer*> input_layers;



        def_uint_t weight_x = 1; def_uint_t weight_y = 1;
        // vector<def_float_t> weights;


        // if the layer is convolutional, then it requires multiple 3D filters stored as weights
        // 3D vector storing weights

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
        // if a convolutional layer, then cached values would be 3D,
        // if a normal layer, then cached values would be 1D
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
        def_uint_small_t is_input_layer = 0;


        def_float_t get_weight(def_uint_t px, def_uint_t py){
            if(px < weight_x && px < weight_y){
                // std::cout << "(" << px << " , " << py << ")";
                // std::cout << " off=" << (weight_x * py + px) << " > ";
                return weights_allocator[(weight_x * py + px)*sizeof(def_float_t)];
            }else{
                print_telm("get address not in weight matrix range." << px << ',' << py)
                return -1;
            }
        }

        void set_weight(def_uint_t px, def_uint_t py, def_float_t new_value){
            if(px < weight_x && px < weight_y){
                weights_allocator[(weight_x * py + px)*sizeof(def_float_t)] = new_value;
            }else{
                print_telm("get address not in weight matrix range." << px << ',' << py)
            }
        }
        
        nlayer(){};

        nlayer(def_uint_t x, def_uint_t y, def_uint_t z, def_uint_t activationFn, def_float_t learningRate)
        {
            this->x = x;
            this->y = y;
            this->z = z;
            this->activationFn = activationFn;
            this->learningRate = learningRate;
            this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }

        nlayer(def_uint_t x, def_uint_t y, def_uint_t z, def_uint_t activationFn)
        {
            this->x = x;
            this->y = y;
            this->z = z;
            this->activationFn = activationFn;
            this->learningRate = INITIAL_LEARNING_RATE;
            this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }

        nlayer(def_uint_t x, def_uint_t activationFn, def_float_t learningRate)
        {
            this->x = x;
            this->y = 1;
            this->z = 1;
            this->activationFn = activationFn;
            this->learningRate = learningRate;
            this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }

        def_uint_t init_weight() {

            // check dimensions > 0
            if(this->weight_y < 1 || this->weight_x < 1){ return -1; }

            if(weights_allocator != NULL){
                if(allocated_weight_x != weight_x || allocated_weight_y != weight_y){
                    reset_weights();
                }else{
                    print_telm("Weight already allocated to " << weights_allocator);
                    return -1; // already allocated
                }
            }


            // intialize to 0
            if(this->layer_type == Fully_Connected_INPUTS) {
                allocated_weight_x = weight_x;
                allocated_weight_y = weight_y;
                weights_allocator = new def_float_t[allocated_weight_x * allocated_weight_y];
                memset(weights_allocator, 0, allocated_weight_x * allocated_weight_y);
            }
            return 0;
        }

        def_uint_t get_id() {
            return this->id;
        }

        def_uint_t get_layerVersion() {
            return this->layerVersion;
        }

        // void print_weight_matrix(){

        // }

        // pass the vector of cached values
        vector<vector<vector<def_float_t>>> get_cached_activation_values()
        {
            return this->cached_acivation_values;
        }

        def_int_t add_input_layer(nlayer * new_layer)  // success = 0, failure = -1
        {
            // check if it is not same as this
            this->input_layers.push_back(new_layer);

            print_telm("Added input layer to layer " << this->id << " with id " << new_layer->get_id() << endl);
            return 0;
        }

        vector<vector<vector<def_float_t>>> get_activation(def_int_t run_id){
            if(this->cached_run_id == run_id){
                return cached_acivation_values;
            }
            
        }

        void set_activation_fn(def_uint_t new_activation_fn){
            this->activationFn = new_activation_fn;
        }



        void reset_weights()
        {
            // check this->layer_type
            if(this->layer_type == Fully_Connected_INPUTS) {
                if(weights_allocator==NULL){

                }  // if not already allocated

                // check if dimension of existing weights is correct
                if(allocated_weight_x != weight_x || allocated_weight_y != weight_y){
                    // size allocated is old or incorrect
                    // deallocate weight matrix
                    

                }

                def_uint_t count = 0;
                fori(i,allocated_weight_x){
                    fori(j,allocated_weight_y){
                        set_weight(i,j,get_rand_float_seeded((unsigned int)count));
                        count++;
                    }
                    // weights_allocator[i] = get_rand_float();
                }
            }
        }





    };





} // namespace std
