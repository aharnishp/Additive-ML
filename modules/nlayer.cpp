#include "nlayer.hpp"

#include<cstring>
// #include<cblas.h>

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
        // def_float_t* weights_allocator = NULL;
        // def_uint_t allocated_weight_x = 0; def_uint_t allocated_weight_y = 0;

    public:

        // general info
        def_uint_t id = 0;

        def_uint_small_t layer_type = Fully_Connected_INPUTS;
        // def_uint_t layerVersion = 0;

        // shape of this 3D layer
        def_uint_t x = 1;
        def_uint_t y = 1;
        def_uint_t z = 1;


        // vector of pointers storing pointer to input layers
        vector<nlayer*> input_layers;


        // weights will only be used when not an input layer and layer_type = Fully Connected
        def_uint_t weight_inp = 0; def_uint_t weight_out = 0;
        // NOTE: 2D weights matrix is stored as as 1D flattened vector expected as row major.
        vector<def_float_t> weights; // weights[px][py] = weights[px*weight_inp + py]

        // NOTE: 4D vector of filters is stored as 1D flattened vector expected as row major. Hence all filters must be of same size or need to make new layer for heterogenous sizes.
        def_uint_t num_filters = 0;

        def_uint_t filter_x = 0;
        def_uint_t filter_y = 0;
        def_uint_t filter_z = 0;

        vector<def_float_t> conv_filters;
        // DEPRECTATED: //  vector<vector<vector<vector<def_float_t>>>> conv_filters;

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
        // NOTE: is now 1D flattened vector expected as row major
        vector<def_float_t> cached_acivation_values;


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


        nlayer(){};

        nlayer(def_uint_t x, def_uint_t y, def_uint_t z, def_uint_t activationFn, def_float_t learningRate)
        {
            this->layer_type = Convolutional_INPUTS;
            this->x = x;
            this->y = y;
            this->z = z;
            this->activationFn = activationFn;
            this->learningRate = learningRate;
            // this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }

        // nlayer(def_uint_small_t new_layer_type, def_uint_t x, def_uint_t y, def_uint_t z, def_uint_t activationFn, def_float_t learningRate)
        // {
        //     this->layer_type = new_layer_type;
        //     this->x = x;
        //     this->y = y;
        //     this->z = z;
        //     this->activationFn = activationFn;
        //     this->learningRate = learningRate;
        //     // this->layerVersion = DEFAULT_LAYER_VERSION;
        //     this->is_dynamic_layer = 1;
        // }

        nlayer(def_uint_t x, def_uint_t y, def_uint_t z, def_uint_t activationFn)
        {
            this->layer_type = Convolutional_INPUTS;
            this->x = x;
            this->y = y;
            this->z = z;
            this->activationFn = activationFn;
            this->learningRate = INITIAL_LEARNING_RATE;
            // this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }

        nlayer(def_uint_t x, def_uint_t activationFn, def_float_t learningRate)
        {
            this->x = x;
            this->y = 1;
            this->z = 1;
            this->activationFn = activationFn;
            this->learningRate = learningRate;
            // this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }

        
        def_float_t get_weight_value(def_uint_t px, def_uint_t py){
            if(layer_type==Fully_Connected_INPUTS){
                if(px < weight_inp && py < weight_out){
                    return (weights[px*weight_inp + py]);
                }else{
                    return (-1);
                }
            }else{
                return (-1);
            }
        }

        

        def_uint_t init_weight() {
            // check current layer type
            if(this->layer_type != Fully_Connected_INPUTS){
                return(-1);
            }

            // find the output dimension of every layer in the input_layer array
            def_int_t input_length = 0;

            for(int i = 0; i < input_layers.size(); i++){
                auto this_layer = *(input_layers[i]);
                cout << this_layer.get_id() << endl;

                if(this_layer.layer_type == Fully_Connected_INPUTS){
                    input_length += this_layer.x;
                    print_telm("Added from FCLayer +" << this_layer.x << ".");
                }else if(this_layer.layer_type == Convolutional_INPUTS){
                    print_telm("Added from CNV layer +" << this_layer.x * this_layer.y * this_layer.z);
                    input_length += (this_layer.x * this_layer.y * this_layer.z);
                }else{
                    print_telm("! - Layer type unexpected when calculating input size.");
                }
            }

            print_telm("Total input size is " << input_length);

            // FIXME:
            // check if this conforms to the format
            if(weight_inp > input_length){print_telm("Inputs to this layer lower than before.");}
            weight_inp = input_length;
            weight_out = this->x;

            // TODO:
            // check if current weight matrix contains something already
            // else
            
            weights.clear();
            weights.resize(weight_inp * weight_out);

            for(int i = 0; i < weight_inp * weight_out; i++){
                weights[i] = get_rand_float_seeded(i);
            }


            // therefore the x of weight matrix is input_length before multiplication
            // check if weight_inp <= 
            return 0;
        }

        def_uint_t get_id() {
            return this->id;
        }


        // pass the vector of cached values
        vector<def_float_t> get_cached_activation_values(){
            return this->cached_acivation_values;
        }

        def_int_t add_input_layer(nlayer * new_layer)  // success = 0, failure = -1
        {
            // check if it is not same as this
            this->input_layers.push_back(new_layer);

            print_telm("Added input layer to layer " << this->id << " with id " << new_layer->get_id() << endl);
            return 0;
        }


        void set_activation_fn(def_uint_t new_activation_fn){
            this->activationFn = new_activation_fn;
        }

        vector<def_float_t> get_activation(def_int_t run_id){
            if(this->cached_run_id == run_id){
                return cached_acivation_values;
            }
            // build an array of input activation before calculating itself's activation
            // visit all input_layers and add number of nodes to get the final result

            // the changing perspective code starts below.
            std::nlayer* curr_layer_ptr;
            // vector<nlayer*> visited_layers;

            vector<def_float_t> curr_input_activations; // store the value vector for input_layers of current layer.
            curr_layer_ptr = this;

            // for current layer, check each input layer
            for(int i = 0; i < curr_layer_ptr->input_layers.size(); i++){
                // checking what is type of this layer
                if(curr_layer_ptr->input_layers[i]->layer_type == Fully_Connected_INPUTS){
                    std::cout << "this layer is Fully Connected" << std::endl;
                    def_uint_t this_input_size = curr_layer_ptr->input_layers[i]->x * curr_layer_ptr->input_layers[i]->y * curr_layer_ptr->input_layers[i]->z;

                    // check if this layer has cached values for current run id
                    if(curr_layer_ptr->cached_run_id == run_id){
                        // curr_input_activations.insert(curr_layer_ptr->cached_acivation_values);
                        curr_input_activations.insert(curr_input_activations.end(), curr_layer_ptr->cached_acivation_values.begin(), curr_layer_ptr->cached_acivation_values.end() );
                    }

                }else if(curr_layer_ptr->input_layers[i]->layer_type == Convolutional_INPUTS){
                    std::cout << "this layer is a Convolutional layer" << std::endl;
                    
                }
            }
        }

        // void reset_weights()
        // {
        //     // check this->layer_type
        //     if(this->layer_type == Fully_Connected_INPUTS) {
        //         if(weights_allocator==NULL){

        //         }  // if not already allocated

        //         // check if dimension of existing weights is correct
        //         if(allocated_weight_x != weight_inp || allocated_weight_y != weight_out){
        //             // size allocated is old or incorrect
        //             // deallocate weight matrix
                    

        //         }

        //         def_uint_t count = 0;
        //         fori(i,allocated_weight_x){
        //             fori(j,allocated_weight_y){
        //                 set_weight(i,j,get_rand_float_seeded((unsigned int)count));
        //                 count++;
        //             }
        //             // weights_allocator[i] = get_rand_float();
        //         }
        //     }
        // }





    };





} // namespace std
