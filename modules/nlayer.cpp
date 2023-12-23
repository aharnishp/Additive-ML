#include "nlayer.hpp"

#include <vector>
#include <cmath>
#include <cstring>
#if defined(__x86_64__) || defined(__aarch64__)
    #define USE_OPEN_BLAS 1    // previously = 1
    #include<cblas.h>
#else
    #define USE_OPEN_BLAS 0
#endif
#define USE_OPEN_BLAS 0


// #define fori(i,n) for(int i = 0; i < n; i++)
// #define pb push_back

//// Compile Time Parameters 
// #define Low_Memory_Target 0


// Compile Parameter Code
#ifdef Low_Memory_Target
  #define def_float_t float
  #define def_uint_small_t __UINT8_TYPE__
  #define def_uint_t __UINT16_TYPE__
  #define def_int_t __INT16_TYPE__
#else
  #define def_float_t double
  #define def_uint_small_t __UINT16_TYPE__
  #define def_uint_t __UINT32_TYPE__
  #define def_int_t __INT32_TYPE__
#endif

// Activation Function Codes
// enum activationFn{
//     ReLU = 0,
//     Linear = 1,
//     Sigmoid = 2,
//     Exponential = 3
// };

typedef enum {
    ReLU = 0,
    Linear = 1,
    Sigmoid = 2,
    Exponential = 3,
    Softmax = 4,
    LReLU = 5
} activation_fn_t;

#define leaky_relu_slope 0.05


// Settings
#define TELEMETRY 0     // 0 is no string, 1 is only errors, 2 is full telemetry
#define DEFAULT_LAYER_VERSION 1
#define INITIAL_LEARNING_RATE 0.05



#if TELEMETRY == 2
  #include<iostream>
  #define print_telm(x) std::cout << x << std::endl;
  #define print_err(x) std::cout << x << std::endl;
#elif TELEMETRY == 1
  #include<iostream>
  #define print_err(x) std::cout << x << std::endl;
  #define print_telm(x)
#else
  #define print_err(x)
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



// namespace std
// {
class nlayer{

private:
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
    std::vector<nlayer*> input_layers;


    // weights will only be used when not an input layer and layer_type = Fully Connected
    // INFO: the number of rows in the weight matrix corresponds to the number of input units, and the number of columns corresponds to the number of output units.
    def_uint_t weight_inp = 0;  
    def_uint_t weight_out = 0;
    // NOTE: 2D weights matrix is stored as as 1D flattened vector expected as row major.
    // vector<def_float_t> weights; // weights[px][py] = weights[px*weight_out + py]
    std::vector<def_float_t> weights;

    // NOTE: 4D vector of filters is stored as 1D flattened vector expected as row major. Hence all filters must be of same size or need to make new layer for heterogenous sizes.
    def_uint_t num_filters = 0;

    def_uint_t filter_x = 0;
    def_uint_t filter_y = 0;
    def_uint_t filter_z = 0;

    std::vector<def_float_t> conv_filters;
    // DEPRECTATED: //  vector<vector<vector<vector<def_float_t>>>> conv_filters;

    // vector of float storing bias
    std::vector<def_float_t> bias;

    // int to store activation function of the layer
        // 0 = ReLU
        // 1 = Linear
        // 2 = sigmoid
        // 3 = exponential
        // 4 = softmax
    activation_fn_t activationFn = ReLU;

    // if learning_rate of current layer is 0, than the layer weights will be immutable
    def_float_t learning_rate = INITIAL_LEARNING_RATE;



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
    std::vector<def_float_t> cached_activation_values;


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

    nlayer(def_uint_t x, def_uint_t y, def_uint_t z, activation_fn_t activationFn, def_float_t learning_rate){
        this->layer_type = Convolutional_INPUTS;
        this->x = x;
        this->y = y;
        this->z = z;
        this->weight_out = x * y * z;
        this->activationFn = activationFn;
        this->learning_rate = learning_rate;
        // this->layerVersion = DEFAULT_LAYER_VERSION;
        this->is_dynamic_layer = 1;
    }


    nlayer(def_uint_t x, def_uint_t y, def_uint_t z, activation_fn_t activationFn) {
        this->layer_type = Convolutional_INPUTS;
        this->x = x;
        this->y = y;
        this->z = z;
        this->weight_out = x * y * z;
        this->activationFn = activationFn;
        this->learning_rate = INITIAL_LEARNING_RATE;
        // this->layerVersion = DEFAULT_LAYER_VERSION;
        this->is_dynamic_layer = 1;
    }

    nlayer(def_uint_t x, activation_fn_t activation_fn, def_float_t learning_rate){
        this->x = x;
        this->y = 1;
        this->z = 1;
        this->weight_out = x;
        this->activationFn = activation_fn;
        this->learning_rate = learning_rate;
        // this->layerVersion = DEFAULT_LAYER_VERSION;
        this->is_dynamic_layer = 1;
    }

    nlayer(def_uint_t x) {
        this->x = x;
        this->y = 1;
        this->z = 1;
        this->weight_out = x;
        this->activationFn = ReLU;
        this->layer_type = Fully_Connected_INPUTS;
        this->learning_rate = INITIAL_LEARNING_RATE;
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
            this->weights.resize(weight_inp * weight_out);

            def_int_t rand_seed = get_rand_float()*1000;

            if(weight_inp == 0 || weight_out == 0){
                
                print_err("Error weight dimensions are unknown.")
                return 1;
            }

            def_uint_small_t has_relu = 0;
            // check if input layers include any ReLU layer
            for(int i = 0; i < this->input_layers.size(); i++){
                if(this->input_layers[i]->activationFn == ReLU || this->input_layers[i]->activationFn == LReLU){
                    has_relu = 1;
                    break;
                }
            }
            // FIXME: Commented He Initialization
            if(random_values && has_relu){
                // He initialization
                // def_float_t std_dev = sqrt(2.0/1); 
                def_float_t std_dev = sqrt(2.0/this->weight_inp); 
                // Fill weights with random values based on the seed and normal distribution
                for (int i = 0; i < weight_inp * weight_out; i++){
                    def_float_t u1 = get_rand_float_seeded(rand_seed);
                    def_float_t u2 = get_rand_float_seeded((rand_seed++) + 1);
                    weights[i] = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
                    weights[i] *= std_dev;
                }
            }else{  // TODO: Improve control flow structure // every other case
                for(int i = 0; i < weight_inp * weight_out; i++){
                    if(random_values){
                        this->weights[i] = get_rand_float_seeded(rand_seed++);
                    }else{
                        this->weights[i] = 0;
                    }
                }
            }

            this->bias.clear();
            this->bias.reserve(weight_out);
            for (int i = 0; i < weight_out; i++) {
                this->bias.push_back(0);
                // this->bias.push_back(get_rand_float_seeded(rand_seed++));
            }
            
        }
        return 0;
    }

    void apply_activation_fn(std::vector<def_float_t>& input_vector, def_uint_t batch_size){
        if(this->activationFn == LReLU){
            for(int i = 0; i < input_vector.size(); i++){
                // std::transform(input_vector.begin(), input_vector.end(), input_vector.begin(), [](def_float_t x) { return x < 0 ? x * leaky_relu_slope : x; });
                if(input_vector[i] < 0){
                    input_vector[i] *= leaky_relu_slope;
                }
            }

        }else if(this->activationFn == ReLU){
            // cblas_d
            #if USE_OPEN_BLAS
            // TODO: Use SIMD here instead
                for(int i = 0; i < input_vector.size(); i++){
                    if(input_vector[i] < 0){
                        input_vector[i] = 0;
                    }
                }
                // cblas_smax(0.0f, input_vector.data(), 1, input_vector.size());
            #else
                for(int i = 0; i < input_vector.size(); i++){
                    if(input_vector[i] < 0){
                        input_vector[i] = 0;
                    }
                }
            #endif
        }else if(this->activationFn == Sigmoid){
            for(int i = 0; i < input_vector.size(); i++){
                input_vector[i] = 1/(1 + exp(-input_vector[i]));
            }
        }else if(this->activationFn == Softmax){
            // perform the softmax batch-wise
            for(int batch = 0; batch < batch_size; batch++){
                def_float_t batch_start = batch * (input_vector.size()/batch_size);
                def_float_t batch_end = (batch + 1) * (input_vector.size()/batch_size); // excluding this index
                def_float_t sum = 0;
                // def_float_t max
                def_float_t max_element = -INFINITY;
                // shift the values by subtracting max of all
                for(int i = batch_start; i < batch_end; i++){ // TODO: potential bottleneck here
                    if(max_element < input_vector[i]){
                        max_element = input_vector[i];
                    }
                }
                for(int i = batch_start; i < batch_end; i++){   // may be parallelised
                    input_vector[i] -= max_element;
                    sum += exp(input_vector[i]);
                }

                for(int i = batch_start; i < batch_end; i++){   // can be parallelised
                    input_vector[i] = exp(input_vector[i])/sum;
                }
            }
        }
    }

    def_int_t size(){
        if(Fully_Connected_INPUTS){
            return this->x * this->y * this->z;
        }else{
            return this->x * this->y * this->z;
        }
    }

    void multiply_activation_derivative_fn(std::vector<def_float_t>& input_vector){
        
        #if USE_OPEN_BLAS
            // TODO: Use SIMD here instead
        #else
            if(this->activationFn == Linear){
                // do nothing
            }else if(this->activationFn == LReLU){
                for(int i = 0; i < input_vector.size(); i++){
                    if(input_vector[i] < 0){
                        input_vector[i] = leaky_relu_slope;
                    }else{
                        input_vector[i] = 1;
                    }
                }
            }else if(this->activationFn == ReLU){
                for(int i = 0; i < input_vector.size(); i++){
                    if(input_vector[i] < 0){
                        input_vector[i] = 0;
                    }
                }
            }else if(this->activationFn == Exponential){    // FIXME:
                std::cout << "Currently not supporting back prop on exponential activation fn" << std::endl;
                // for(int i = 0; i < input_vector.size(); i++){   // SIMD
                    
                // }
            }else if(this->activationFn == Sigmoid){        // FIXME:
                
            }else if(this->activationFn == Softmax){
                
            }
        #endif
    }

    /**
     * @brief Automatically grow weights to match the input and output size of the layer.
     * @return 0 if successful, 1 if failed.
    */
    def_uint_small_t auto_grow_weight(){
        // Calculate current input weight size and output weight size and grow weights accordingly.
        
        if(this->layer_type == Fully_Connected_INPUTS){
            def_uint_t new_weight_inp = 0;

            for(int i = 0; i < this->input_layers.size(); i++){
                new_weight_inp += this->input_layers[i]->x * this->input_layers[i]->y * this->input_layers[i]->z;
            }

            def_uint_t new_weight_out = this->x * this->y * this->z;

            return grow_weights(new_weight_inp, new_weight_out, 1);
        }
        return 1;
    }

    def_uint_t get_id() {
        return this->id;
    }

    // pass the vector of cached values
    std::vector<def_float_t> get_cached_activation_values(){
        return this->cached_activation_values;
    }

    def_int_t add_input_layer(nlayer * new_layer){  // success = 0, failure = -1
        // check if it is not same as this
        this->input_layers.push_back(new_layer);

        this->auto_grow_weight();

        print_telm("Added layer(id=" << new_layer->id << ") to layer(id=" << this->id << ") as input." << endl);
        return 0;
    }


    void set_activation_fn(activation_fn_t new_activation_fn){
        this->activationFn = new_activation_fn;
    }

    def_uint_small_t generate_weight_matrix(def_uint_t new_weight_inp, def_uint_t new_weight_out){
        if(this->layer_type == Fully_Connected_INPUTS){
            this->weight_inp = new_weight_inp;
            this->weight_out = new_weight_out;
            this->init_weight(1);
            return 0;
        }else{
            return 1;
        }
    }

    def_uint_small_t grow_weights(def_uint_t new_weight_inp, def_uint_t new_weight_out, def_uint_small_t random_values){
        if(this->layer_type == Fully_Connected_INPUTS){
            // preserve data and insert additional columns or rows
            def_int_t cols_add = new_weight_inp - this->weight_inp;
            def_int_t rows_add = new_weight_out - this->weight_out;

            if(this->weight_inp == 0 || this->weight_out == 0){     // currently any of the dimension == 0
                if(new_weight_inp == 0 || new_weight_out == 0){     // new atleast one of the weight dimension == 0
                    print_err("Error: Cannot grow weight matrix as one of the dimension is 0.")
                    return 1;
                }else if(new_weight_inp != 0 && new_weight_out != 0){   // new both weight dimensions != 0
                    this->weight_inp = new_weight_inp;
                    this->weight_out = new_weight_out;
                    this->init_weight(1);
                    return 0;
                }
            }
            if(cols_add > 0){
                // add columns
                for(int i = 0; i < this->weight_out; i++){
                    for(int j = 0; j < cols_add; j++){
                        this->weights.insert(this->weights.begin() + (i*this->weight_inp) + this->weight_inp + j, (random_values ? get_rand_float() : 0));
                    }
                }
            }else if(cols_add < 0){
                print_err("Error: Cannot shrink weight cols of matrix. Use shrink matrix instead.");
            }
            if(rows_add > 0){
                // add rows
                for(int i = 0; i < rows_add; i++){
                    for(int j = 0; j < this->weight_inp; j++){
                            this->weights.push_back(random_values ? get_rand_float() : 0);
                    }
                }
            }else if(rows_add < 0){
                print_err("Error: Cannot shrink weight rows of matrix. Use shrink matrix instead.");
            }
            return 0;
        }else{
            return 1;
        }
    }

    std::vector<def_float_t> get_activation_rec(def_int_t run_id, def_uint_t batch_size){
        // this is memory in efficient, but faster to implement.
        if(this->cached_run_id == run_id){  // check whether to return directly from cache if already calculated.
            if(TELEMETRY == 2) { std::cout << "Result returned from cache. id=" << this->id << " size=" << this->x * this->y * this->z << std::endl;}
            return cached_activation_values;
        }else if(this->is_input_layer){
            if(cached_activation_values.size() == 0){
                print_telm("Unable to give input values as input values are not provided.")
                if(this->layer_type == Fully_Connected_INPUTS){
                    std::vector<def_float_t> new_empty(this->x * this->y * this->z, 0);
                    if(TELEMETRY == 2) { std::cout << "Result returned from input_cache. id=" << this->id << " size=" << this->x * this->y * this->z << std::endl;}
                    return new_empty;
                }
            }
            if(TELEMETRY == 2) { std::cout << "Result returned from input_cache. id=" << this->id << " size=" << this->x * this->y * this->z << std::endl;}
            return cached_activation_values;
        }

        this->being_evaluated = 1;  // lock flag and prevent cyclic evaluations.

        if(this->layer_type == Fully_Connected_INPUTS){
            if(weight_inp == 0 || weight_out == 0){
                print_err("Error weight dimensions are unknown.")
            }

            // check if current layer output size has grown
            if(weight_out != this->x * this->y * this->z){
                grow_weights(weight_inp, this->x * this->y * this->z, 1);
            }

            // confirm if this layer weights are actually flattened appropriately
            if(this->weights.size() != weight_inp * weight_out){
                // TODO: Make adjust_weight_dimension() to non-destructively handle this. 
                this->init_weight(1);
            }
                            
            // build an array of input activation before calculating itself's activation
            std::vector<def_float_t> input_activations;

            // collect activations of all layers and append to vector input_activations
            for(int i = 0; i < this->input_layers.size(); i++){
                std::vector<def_float_t> new_activation;
                new_activation = this->input_layers[i]->get_activation_rec(run_id, batch_size);
                input_activations.insert(input_activations.end(), new_activation.begin(), new_activation.end());
            }

            // check if input size is actually supported
            if(weight_inp * batch_size != input_activations.size()){
                print_err("Error: Input size(" << input_activations.size() <<") does not match weight matrix size.")
                def_uint_t new_weight_out_size = this->weight_out;
                if(this->weight_out == 0){
                    new_weight_out_size = this->x * this->y * this->z;
                }
                this->grow_weights(input_activations.size(), new_weight_out_size, 1);
            }

            if(input_activations.size() == weight_inp*batch_size){
                // do the matrix multiplication
                    std::vector<def_float_t> output_vector(weight_out*batch_size);
                #if USE_OPEN_BLAS
                    // TODO:
                #else

                    // printing this->weights
                    if(TELEMETRY == 2) {
                    std::cout << "&this(id)= " << id << "  \t" << this << std::endl;
                    std::cout << "this->weight_inp=" << this->weight_inp << "\t this->weight_out=" << this->weight_out << std::endl;
                    std::cout << "this->weights.size=" << this->weights.size() << "\t this->weights flattened values=" << std::endl;
                        for(int i = 0; i < this->weights.size(); i++){ 
                            std::cout << this->weights[i] << " ";
                        }std::cout << std::endl;
                    }

                    for (int batch = 0; batch < batch_size; batch++) {
                        for (int out = 0; out < weight_out; out++) {
                            def_float_t result = 0.0f;
                            for (int in = 0; in < weight_inp; in++) {
                                result += input_activations[batch * weight_inp + in] * this->weights[in * weight_out + out];
                            }
                            output_vector[batch * weight_out + out] = result;
                        }
                    }

                    // add bias
                    if(TELEMETRY == 2){ if(bias.size() != weight_out){ std::cout << "this->bias uninitialized. this=" << this << std::endl; } }
                    for (int i = 0; i < weight_out*batch_size; i++) {
                        output_vector[i] += this->bias[i%weight_out];   // add bias to all elements in the batch
                    }
                    

                #endif

                // apply activation function
                // output_vector = get_activation_Fn_value(output_vector);
                apply_activation_fn(output_vector, batch_size);

                // store copy of outputs to cache 
                this->cached_run_id = run_id;
                this->cached_activation_values = output_vector;  // confirmed: creates copy
                this->cached_batch_size = batch_size;

                if(TELEMETRY == 2){
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
                if(TELEMETRY == 2){std::cout << "Weight matrix input size not adjusted for input_activations." << std::endl;}
                // TODO: Add support for increasing matrix row or columns 
            }
            // check if existing dimensions of this->weights matches with size of inputs 
            this->being_evaluated = 0;
        }else if(this->layer_type == Convolutional_INPUTS){

            this->being_evaluated = 0;
        }


        // error case
        std::vector<def_float_t> empty_vector = {-1};
        return empty_vector;
    }

    /**
     * @brief Calculate the backprop error for this layer.
     * @param run_id The run_id of the current run.
     * @param batch_size The batch size of the current run.
     * @param activation_error The error of the next layer.\
     * @param learning_rate The learning rate of the current run.
    */
    std::vector<def_float_t> get_correct_error_rec(def_int_t run_id, def_uint_t batch_size, std::vector<def_float_t>& activation_error, def_float_t learning_rate){
        if(!(this->being_corrected)){
            this->being_corrected = 1;
        }else{
            // this is detecting loop
            print_telm("Loop detected in calculating backprop error.");
            return(this->cached_activation_values);
        }

        if(this->is_input_layer == 1){
            this->being_corrected = 0;
            return std::vector<def_float_t>(0,0);
        }

        if(this->layer_type == Fully_Connected_INPUTS){
            // check if backprop errors are fresh, otherwise, wrong errors will be calculated.
            if(this->cached_run_id < run_id){
                if(TELEMETRY) {std::cout << "Uncalculated forward prop cache detected. this=" << this << std::endl;}
                this->get_activation_rec(run_id, batch_size);
            }

            std::vector<def_float_t> error_diff;
            error_diff.reserve(activation_error.size());
            #if USE_OPEN_BLAS    // TODO: Add SIMD instructions
            #else
                // getting difference (error) in current activation and expected activation
                // for(int i = 0; i < activation_error.size(); i++){
                //     error_diff.push_back(this->cached_acivation_values[i] - activation_error[i]);
                // }
                
                // generate a list of last inputs
                std::vector<def_float_t> last_inputs;
                last_inputs.reserve(this->weight_inp);

                this->being_evaluated = 1;

                for(int i = 0; i < this->input_layers.size(); i++){
                    // asking layer for their activation
                    std::vector<def_float_t> new_activation = this->input_layers[i]->get_activation_rec(run_id, batch_size);
                    last_inputs.insert(last_inputs.end(), new_activation.begin(), new_activation.end());

                    // currently taking cached values directly from input_layers
                    // if (this->input_layers[i]->cached_run_id == run_id){
                    //     last_inputs.insert(last_inputs.end(), this->input_layers[i]->cached_acivation_values.begin(), this->input_layers[i]->cached_acivation_values.end());
                    // }else{
                    //     std::vector<def_float_t> new_activation = this->input_layers[i]->get_activation_rec(run_id, batch_size);
                    //     last_inputs.insert(last_inputs.end(), new_activation.begin(), new_activation.end());
                    // }
                }

                this->being_evaluated = 0;



                std::vector<def_float_t> delta_weight;   // the dimensions are same as weights matrix
                delta_weight.reserve(this->weight_inp * this->weight_out);

                def_float_t reci_batch_size = 1.0/batch_size;
                
                // Matrix Multiply to get delta_weights
                def_float_t sum = 0;
                // FIXME: Rewrite code to generate Matrix
                for(int col = 0; col < weight_inp; col++){
                    for(int row = 0; row < weight_out; row++){
                        sum = 0;
                        for(int batch_num = 0; batch_num < batch_size; batch_num++){
                            sum += activation_error[(batch_num*this->weight_out) + row] * last_inputs[(batch_num*this->weight_inp) + col];
                        }

                        delta_weight.push_back(sum * reci_batch_size);
                    }
                }

                if(TELEMETRY == 2){
                    std::cout << "# delta_weight = " << std::endl;
                    for (int i = 0; i < weight_inp; i++) {
                        for (int j = 0; j < weight_out; j++) {
                            std::cout << delta_weight[i * weight_out + j] << " ";
                        }
                        std::cout << std::endl;
                    }
                }

                // calculate for Biases
                std::vector<def_float_t> delta_bias;    // empty vec
                delta_bias.reserve(weight_out);

                // summing all errors for each neurons across the batch
                for (int i = 0; i < weight_out; i++) {
                    def_float_t sum = 0;
                    for (int j = 0; j < batch_size; j++) {
                        sum += activation_error[j * weight_out + i];
                    }
                    delta_bias.push_back(sum * reci_batch_size);
                }

                if(TELEMETRY == 2){
                    std::cout << "# delta_bias = " << std::endl;
                    for (int i = 0; i < weight_out; i++) {
                        std::cout << delta_bias[i] << " ";
                    }
                    std::cout << std::endl;
                }

                std::vector<def_float_t> old_weights = this->weights;

                // update weights
                for(int i = 0; i < weight_out; i++){
                    for(int j = 0; j < weight_inp; j++){
                        this->weights[ i * weight_inp + j ] -= (delta_weight[ i * weight_inp + j ] * learning_rate);
                    }
                }

                // // update bias
                for(int i = 0; i < weight_out; i++){
                    this->bias[i] -= delta_bias[i] * learning_rate;
                }

                // input_dz = (W.T x dZ) * g'(Z)

                // finding error for input layer
                std::vector<def_float_t> input_error;
                input_error.reserve(this->weight_inp * batch_size);

                // find the inverse transformation of weights matrix
                // matrix multiply activation_error and last_inputs
                for(int i = 0; i < this->weight_inp; i++){
                    for(int j = 0; j < this->weight_out; j++){
                        def_float_t sum = 0;
                        for(int k = 0; k < batch_size; k++){
                            sum += activation_error[k*this->weight_out + j] * old_weights[i*this->weight_out + j];
                        }
                        input_error.push_back(sum);
                    }
                }


                // multiply the derivative of activation function with input_error
                multiply_activation_derivative_fn(input_error);

                def_uint_t input_split_ptr = 0;

                // splitting input corrections to their corresponding layers
                for(int i = 0; i < (this->input_layers.size()); i++) {  // for each layer
                    std::vector<def_float_t> this_errors;
                    def_uint_t this_layer_output_size = 0;



                    if(this->input_layers[i]->layer_type == Fully_Connected_INPUTS){
                        this_layer_output_size = this->input_layers[i]->weight_out;

                        this_errors.reserve((this_layer_output_size) * batch_size);
                    }

                    def_uint_t start_inp = input_split_ptr;
                    def_uint_t end_inp = input_split_ptr + this_layer_output_size;

                    // also for each batch data point. basically taking out a slice from a flattened 2D matrix
                    for(int batch = 0; batch < batch_size; batch++){
                        def_int_t start_range = start_inp + batch * this->weight_inp;
                        def_int_t end_range = end_inp + batch * this->weight_inp;

                        if(TELEMETRY == 2){
                            std::cout << "input_error.size()=" << input_error.size() << std::endl;
                            std::cout << "this_errors.size()=" << this_errors.size() << std::endl;
                        }

                        // FIXME: faster method crashes with larger array sizes
                        // pasting behind the array this_errors
                        // this_errors.insert(this_errors.end(),
                        //     input_error.begin() + start_range * sizeof(def_float_t),
                        //     input_error.begin() + (end_range + 1) * sizeof(def_float_t));

                        // inserting manually
                        for(int el = 0; el < input_error.size(); el++){
                            this_errors.push_back(input_error[el]);
                        }

                    }

                    this->input_layers[i]->get_correct_error_rec(run_id, batch_size, this_errors, learning_rate);
                    
                }





            #endif


        }else if(this->layer_type== Convolutional_INPUTS){

        }

        
        this->being_corrected = 0;
        return std::vector<def_float_t>(0,0);
    }
};





// } // namespace std