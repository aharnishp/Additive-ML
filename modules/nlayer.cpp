#include "nlayer.hpp"

#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(__aarch64__)
    #define USE_OPEN_BLAS 1    // previously = 1
    #include<cblas.h>
#else
    #define USE_OPEN_BLAS 0
#endif


// #define fori(i,n) for(int i = 0; i < n; i++)
// #define pb push_back

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
// #define Linear_activation_state 0
// #define ReLU_activation_state 1
// #define Sigmoid_activation_state 2
// #define Exponential_activation_state 3


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
        // INFO: the number of rows in the weight matrix corresponds to the number of input units, and the number of columns corresponds to the number of output units.
        def_uint_t weight_inp = 0;  
        def_uint_t weight_out = 0;
        // NOTE: 2D weights matrix is stored as as 1D flattened vector expected as row major.
        // vector<def_float_t> weights; // weights[px][py] = weights[px*weight_out + py]
        vector<def_float_t> weights;

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
        activation_fn_t activationFn = ReLU;

        def_float_t learningRate = INITIAL_LEARNING_RATE;



        // caching activations after activation Fn.
        def_uint_small_t cache_init = un_initiated_cache_state;

        // stores run_id when it this layer's last activation was calculated
        def_int_t cached_run_id = 0;
        def_uint_t cached_batch_size = 1;    // store the batch size of the input

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

        nlayer(def_uint_t x, def_uint_t y, def_uint_t z, activation_fn_t activationFn, def_float_t learningRate)
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

        nlayer(def_uint_t x, def_uint_t y, def_uint_t z, activation_fn_t activationFn)
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

        nlayer(def_uint_t x, activation_fn_t activation_fn, def_float_t learningRate)
        {
            this->x = x;
            this->y = 1;
            this->z = 1;
            this->activationFn = activation_fn;
            this->learningRate = learningRate;
            // this->layerVersion = DEFAULT_LAYER_VERSION;
            this->is_dynamic_layer = 1;
        }

        
        def_float_t get_weight_value(def_uint_t px, def_uint_t py){
            if(layer_type==Fully_Connected_INPUTS){
                if(px < weight_inp && py < weight_out){
                    return (weights[px*weight_out + py]);
                }else{
                    return (-1);
                }
            }else{
                return (-1);
            }
        }

        def_uint_small_t init_weight(def_uint_small_t random_values) {
            if(this->layer_type == Fully_Connected_INPUTS){
                this->weights.clear();
                this->weights.reserve(weight_inp * weight_out);

                def_int_t rand_seed = get_rand_float()*1000;
                
                for(int i = 0; i < weight_inp * weight_out; i++){
                    if(random_values){
                        this->weights[i] = get_rand_float_seeded(rand_seed++);
                    }else{
                        this->weights[i] = 0;
                    }
                }
            }
            return 0;
        }

        void apply_activation_fn(std::vector<def_float_t>& input_vector){
            if(this->activationFn == ReLU){
                #ifdef USE_OPEN_BLAS
                    cblas_smax(0.0f, input_vector.data(), 1, input_vector.size());
                #endif
            }
            
        }

        // std::vector<def_float_t> get_activation_Fn_value(std::vector<def_float_t> input){
        //     if(this->activationFn == ReLU){
        //         if(input > 0){
        //             return input;
        //         }else{
        //             return 0;
        //         }
        //     }else if(this->activationFn == Linear){
        //         return input;
        //     }else if(this->activationFn == Sigmoid){
        //         return (1/(1+exp(-input)));
        //     }else if(this->activationFn == Exponential){
        //         return exp(input);
        //     }else{
        //         return input;
        //     }
        // }

        // def_uint_t init_weight() {
        //     // check current layer type
        //     if(this->layer_type != Fully_Connected_INPUTS){
        //         return(-1);
        //     }

        //     // find the output dimension of every layer in the input_layer array
        //     def_int_t input_length = 0;

        //     for(int i = 0; i < input_layers.size(); i++){
        //         auto this_layer = *(input_layers[i]);
        //         cout << this_layer.get_id() << endl;

        //         if(this_layer.layer_type == Fully_Connected_INPUTS){
        //             input_length += this_layer.x;
        //             print_telm("Added from FCLayer +" << this_layer.x << ".");
        //         }else if(this_layer.layer_type == Convolutional_INPUTS){
        //             print_telm("Added from CNV layer +" << this_layer.x * this_layer.y * this_layer.z);
        //             input_length += (this_layer.x * this_layer.y * this_layer.z);
        //         }else{
        //             print_telm("! - Layer type unexpected when calculating input size.");
        //         }
        //     }

        //     print_telm("Total input size is " << input_length);

        //     // FIXME:
        //     // check if this conforms to the format
        //     if(weight_inp > input_length){print_telm("Inputs to this layer lower than before.");}
        //     weight_inp = input_length;
        //     weight_out = this->x;

        //     // TODO:
        //     // check if current weight matrix contains something already
        //     // else
            
        //     weights.clear();
        //     weights.resize(weight_inp * weight_out);

        //     for(int i = 0; i < weight_inp * weight_out; i++){
        //         weights[i] = get_rand_float_seeded(i);
        //     }


        //     // therefore the x of weight matrix is input_length before multiplication
        //     // check if weight_inp <= 
        //     return 0;
        // }

        def_uint_small_t auto_resize_weight(def_uint_t preferred_input_size, def_uint_t preferred_output_size){
            if(this->layer_type == Fully_Connected_INPUTS){
                // weight_out must also equal x * y * z
                if(weight_inp == preferred_input_size && weight_out == preferred_output_size){
                    return 0;   // if already the same size.
                }
                while(weight_inp != preferred_input_size || weight_out != preferred_output_size){
                    // try to resize both dimensions until successfully resized

                }
                return 1;
            }else if(this->layer_type == Convolutional_INPUTS){

            }

            // return 1;   // if had to resize.
            // return 0;   // if already the same size.
            return -1;  // if this size is smaller than preferred.
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


        void set_activation_fn(activation_fn_t new_activation_fn){
            this->activationFn = new_activation_fn;
        }

        vector<def_float_t> get_activation_rec(def_int_t run_id, def_uint_t batch_size){
            // this is memory in efficient, but faster to implement.
            if(this->cached_run_id == run_id || this->is_input_layer){  // check whether to return directly from cache if already calculated.
                if(TELEMETRY) { std::cout << "Result returned from cache. id=" << this->id << " size=" << this->x * this->y * this->z << std::endl;}
                return cached_acivation_values;
            }

            this->being_evaluated = 1;  // lock flag and prevent cyclic evaluations.

            if(this->layer_type == Fully_Connected_INPUTS){
                // confirm if this layer weights are initialised
                if(this->weights.size() != weight_inp * weight_out){
                    // TODO: Make adjust_weight_dimension() to non-destructively handle this. 
                    this->init_weight(1);
                }
                                
                // build an array of input activation before calculating itself's activation
                vector<def_float_t> input_activations;

                // collect activations of all layers and append to vector input_activations
                for(int i = 0; i < this->input_layers.size(); i++){
                    vector<def_float_t> new_activation = this->input_layers[i]->get_activation_rec(run_id, batch_size);
                    input_activations.insert(input_activations.end(), new_activation.begin(), new_activation.end());
                }

                if(input_activations.size() == weight_inp*batch_size){
                    // do the matrix multiplication
                        std::vector<def_float_t> output_vector(weight_out*batch_size);
                    #if USE_OPEN_BLAS

                        // for matrix to matrix multiplication
                        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, weight_out, batch_size, weight_inp, 1.0f, flattened_weights.data(), weight_inp, input_activations[0].data(), weight_inp, 0.0f, output.data(), weight_out);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, weight_out, weight_inp, 1.0f, input_activations.data(), weight_inp, weights.data(), weight_out, 0.0f, output_vector.data(), weight_out);



                        // For vector to matrix multiplication:
                        // cblas_sgemv(CblasRowMajor, CblasTrans, weight_inp, weight_out, 1.0f, weights.data(), weight_out, input_activations.data(), 1, 0.0f, output_vector.data(), 1);

                    #else
                        // TODO: Add a fallback code for other platforms

                        // printing this->weights
                        std::cout << "&this=  \t" << this << std::endl;
                        std::cout << "this->weight_inp=" << this->weight_inp << "\t this->weight_out=" << this->weight_out << std::endl;
                        std::cout << "this->weights.size=" << this->weights.size() << "\t this->weights flattened values=" << std::endl;
                        for(int i = 0; i < this->weights.size(); i++){ 
                            std::cout << this->weights[i] << " ";
                        }

                        for (int batch = 0; batch < batch_size; batch++) {
                            for (int out = 0; out < weight_out; out++) {
                                float result = 0.0f;
                                for (int in = 0; in < weight_inp; in++) {
                                    result += input_activations[batch * weight_inp + in] * this->weights[in * weight_out + out];
                                }
                                output_vector[batch * weight_out + out] = result;
                            }
                        }

                    #endif


                    // apply activation function
                    // output_vector = get_activation_Fn_value(output_vector);
                    apply_activation_fn(output_vector);

                    if(TELEMETRY){
                        std::cout << "Input Values" << std::endl;
                        for(int i = 0; i < input_activations.size(); i++){
                            std::cout << input_activations[i] << " ";
                        }
                        std::cout << std::endl;


                        std::cout << "Output Values" << std::endl;
                        for(int i = 0; i < output_vector.size(); i++){
                            std::cout << output_vector[i] << " ";
                        }
                        std::cout << std::endl;
                    }

                    this->being_evaluated = 0;

                    return output_vector;

                }else{
                    if(TELEMETRY){std::cout << "Weight matrix input size not adjusted for input_activations." << std::endl;}
                    // TODO: Add support for increasing matrix row or columns 
                }

                // if(TELEMETRY) { std::cout << "inputs=" << std::endl; }

                // check if existing dimensions of this->weights matches with size of inputs 
                


                this->being_evaluated = 0;
            }else if(this->layer_type == Convolutional_INPUTS){

                this->being_evaluated = 0;
            }


            // error case
            vector<def_float_t> empty_vector = {-1};
            return empty_vector;
        }

        vector<def_float_t> get_correct_error_rec(def_int_t run_id, def_uint_t batch_size, vector<def_float_t> activation_error){
            if(!(this->being_corrected)){
                this->being_corrected = 1;
            }else{
                return(this->cached_acivation_values);
            }

            if(this->layer_type == Fully_Connected_INPUTS){
                // check if backprop errors are fresh, otherwise, wrong errors will be calculated.
                if(this->cached_run_id == run_id){
                    

                }
            }

            
            
            this->being_corrected = 0;
        }

        // vector<def_float_t> get_activation(def_int_t run_id){
        //     if(this->cached_run_id == run_id){
        //         return cached_acivation_values;
        //     }
            
        //     if(this->layer_type == Fully_Connected_INPUTS){
        //         // build an array of input activation before calculating itself's activation
        //         // visit all input_layers and add number of nodes to get their activations
        //         vector<def_float_t> input_activations;

        //         // check if all inputs of this layer have cached their activations?
        //         for(int i = 0; i < this->input_layers.size(); i++){
        //             // if this->input_layers[i]->cached_run_id is same as run_id, it was calculated in this iteration.
        //             if(this->input_layers[i]->cached_run_id == run_id){
                        
        //             }
        //         }
                

        //         // // if only it were so simple: // collect activations of all layers and append to vector input_activations
        //         // for(int i = 0; i < this->input_layers.size(); i++){
        //         //     vector<def_float_t> new_activation = this->input_layers[i]->get_activation_rec(run_id);
        //         //     input_activations.insert(input_activations.end(), new_activation.begin(), new_activation.end());
        //         // }

        //         // if(TELEMETRY) { std::cout << "inputs=" << std::endl; }

        //         // check if existing dimensions of this->weights matches with size of inputs 



        //         this->being_evaluated = 0;
        //     }else if(this->layer_type == Convolutional_INPUTS){

        //         this->being_evaluated = 0;
        //     }

        // }

        // vector<def_float_t> get_activation(def_int_t run_id){
        //     if(this->cached_run_id == run_id){
        //         return cached_acivation_values;
        //     }
        //     // build an array of input activation before calculating itself's activation
        //     // visit all input_layers and add number of nodes to get their activations

        //     // the changing perspective code starts below.
        //     std::nlayer* curr_layer_ptr;
        //     // vector<nlayer*> visited_layers;

        //     vector<def_float_t> curr_input_activations; // store the value vector for input_layers of current layer.
        //     curr_layer_ptr = this;

        //     // for current layer, check each input layer
        //     for(int i = 0; i < curr_layer_ptr->input_layers.size(); i++){
        //         // checking what is type of this layer
        //         if(curr_layer_ptr->input_layers[i]->layer_type == Fully_Connected_INPUTS){
        //             std::cout << "this layer is Fully Connected" << std::endl;
        //             def_uint_t this_input_size = curr_layer_ptr->input_layers[i]->x * curr_layer_ptr->input_layers[i]->y * curr_layer_ptr->input_layers[i]->z;

        //             // check if this layer has cached values for current run id
        //             if(curr_layer_ptr->cached_run_id == run_id){
        //                 // curr_input_activations.insert(curr_layer_ptr->cached_acivation_values);
        //                 curr_input_activations.insert(curr_input_activations.end(), curr_layer_ptr->cached_acivation_values.begin(), curr_layer_ptr->cached_acivation_values.end() );
        //             }

        //         }else if(curr_layer_ptr->input_layers[i]->layer_type == Convolutional_INPUTS){
        //             std::cout << "this layer is a Convolutional layer" << std::endl;
                    
        //         }
        //     }
        // }

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
