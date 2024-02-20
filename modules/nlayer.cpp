#include "nlayer.hpp"

#include <vector>
#include <cmath>
#include <cstring>
#if defined(__x86_64__) || defined(__aarch64__)
    #define USE_SIMD 0    // previously = 1
    #include<cblas.h>
    #include<immintrin.h>
#else
    #define USE_SIMD 0
#endif

#define MAE_CALCULATION 1
#define MAE_Split_Min_training 80
#define DEF_MAE_THRESHOLD 0.1

#if MAE_CALCULATION == 1
    #define MAE_HISTORY 1
    #define MAE_HISTORY_SIZE 4
#endif


// #define fori(i,n) for(int i = 0; i < n; i++)
// #define pb push_back

//// Compile Time Parameters 
#define Low_Memory_Target 1

#define weight_row_major 1

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
#define TELE_PROP 0

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
    def_uint_t weight_inp_allocated = 0;
    def_uint_t weight_out_allocated = 0;
    // NOTE: 2D weights matrix is stored as as 1D flattened vector expected as row major.
    // vector<def_float_t> weights; // weights[m][n] = weights[m * weight_inp + n]
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

    // stores run_id when it this layer's last activation was calculated
    def_int_t cached_run_id = 0;
    def_uint_t cached_batch_size = 1;    // store the batch size of the input

    // FIXME: 
    // stores the value of last activation
    // if a convolutional layer, then cached values would be 3D,
    // if a normal layer, then cached values would be 2D flattened to 1D, it would be column major and of size layer.size() * batch_size with adjacent neurons adjacent in single batch
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

    #if MAE_CALCULATION == 1
        std::vector<def_float_t> mae_vec;  // stores the absolute errors accumulated at each node
        std::vector<def_uint_t> mae_count;
        def_float_t max_inp_activ = 0;      // store the max input activation ever given, to normalize layer with activation

        // splitting conditions
        def_float_t mae_threshold = DEF_MAE_THRESHOLD;

        #if MAE_HISTORY == 1
            std::vector<def_float_t> mae_history;
            


        #endif

    #endif


    nlayer(){};

    /**
     * @brief create a convolutional layer with given dimensions, activation function, and learning rate.
     * @param x The number of neurons in the x direction.
     * @param y The number of neurons in the y direction.
     * @param z The number of neurons in the z direction.
     * @param activationFn The activation function of the layer.
     * @param learning_rate The learning rate of the layer.
    */
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


    /**
     * @brief create a convolutional layer with given dimensions and activation function.
     * @param x The number of neurons in the x direction.
     * @param y The number of neurons in the y direction.
     * @param z The number of neurons in the z direction.
     * @param activationFn The activation function of the layer.
    */
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

    /**
     * @brief create a fully connected layer to its input
    */
    nlayer(def_uint_t x, activation_fn_t activation_fn, def_float_t learning_rate){
        this->x = x;
        this->y = 1;
        this->z = 1;
        this->weight_out = x;
        this->activationFn = activation_fn;
        this->learning_rate = learning_rate;
        this->layer_type = Fully_Connected_INPUTS;
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
    
    /**
     * @brief returns the index of the weight matrix in the flattened vector.
     * @param m The row index of the weight matrix. max val = weight_inp 
     * @param n The column index of the weight matrix. max val = weight_out
    */
    inline def_uint_t flat_indx(def_uint_t m, def_uint_t n){
        #if weight_row_major == 1
            return ( (this->weight_inp_allocated * n) + m );    // assuming row major for faster forward prop
        #else
            return ( (this->weight_out_allocated * m) + n );    // assuming column major for faster forward prop
        #endif
    }

    static inline def_uint_t get_default_reserve_size(def_uint_t actual_size){
        return (actual_size * 1.5);
        // return (actual_size + 2);
    }

    // def_float_t get_weight_value(def_uint_t px, def_uint_t py){
    //     if(layer_type==Fully_Connected_INPUTS){
    //         if(px < weight_inp && py < weight_out){
    //             return (weights[px*weight_out + py]);
    //         }else{
    //             return (-1);
    //         }
    //     }else{
    //         return (-1);
    //     }
    // }

    def_float_t get_rand_gaussian(def_uint_t seed){
        static bool generate_cached = false;
        static def_float_t cached_value;

        if (generate_cached) {
            generate_cached = false;
            return cached_value;
        } else {
            def_float_t u1 = get_rand_float_seeded(seed);
            def_float_t u2 = get_rand_float_seeded(seed+1);

            def_float_t z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            def_float_t z1 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);

            // Cache one of the values for the next call
            cached_value = z1;
            generate_cached = true;

            return z0;
        }
    }


    /**
     * @breif fills values according to current weight in and out through random, He, or Xavier Initialization methods
     * @param inp_vec address of the vector to fill values to
     * @param end_fill_indx number of values to append
     * @param num_inp final expected number of inputs
     * @param num_out final expected number of outputs
     * @param init_method 0 = Random from 0 to 1, 1 = Xavier, 2 = He Initialization
    */
    def_uint_small_t fill_weight_value_initialization(std::vector<def_float_t>& inp_vec, def_uint_t start_fill_indx, def_uint_t end_fill_indx, def_uint_t num_inp, def_uint_t num_out, def_uint_small_t init_method){
        if(this->layer_type == Fully_Connected_INPUTS){
            if((num_inp == 0 || num_out == 0) || (start_fill_indx > end_fill_indx)){
                print_err("Error: fill_weight_value_initialization-> one of the dimension is 0 or start_fill_indx > end_fill_indx.");
                return 1;
            }
            def_uint_t seed = time(0);
            if(init_method == 0){
                for(int i = start_fill_indx; i < end_fill_indx; i++){
                    inp_vec[i] = (get_rand_float());
                }
            }else if(init_method == 1){     // Xavier Initialization
                float xavier_variance = 1.0 / (num_inp + num_out);
                float xavier_stddev = std::sqrt(xavier_variance);

                for (int i = start_fill_indx; i < end_fill_indx; i++) {
                    inp_vec[i] = (get_rand_gaussian(seed++) * xavier_stddev);
                }

            }else if(init_method == 2){     // He Initialization
                float he_variance = 2.0 / (num_inp);
                float he_stddev = std::sqrt(he_variance);

                for (int i = start_fill_indx; i < end_fill_indx; i++) {
                    inp_vec[i] = (get_rand_gaussian(seed++) * he_stddev);
                }
            }

            return 0;
        }else{
            return 1;
        }

    }

    /**
     * @breif initialize the weight matrix of size weight_inp * weight_out with random values.
     * @param random_values If 1, then initialize with random values, else initialize all with 0.
     * @param reserve If 1, then reserve more space than required to reduce matrix growing overhead.
    */
    def_uint_small_t init_weight(def_uint_small_t random_values, def_uint_small_t reserve) {
        if(this->layer_type == Fully_Connected_INPUTS){
            std::cout << "DEPRECATED: init weight was called, id=" << this->id << std::endl;

            this->weights.clear();
            // this->weights.resize(weight_inp * weight_out);

            def_uint_small_t has_relu = 0;
            def_uint_small_t is_input = 1;  // check if the only input layer is the network's input layer, no need to make current layer reserve more inputs
            // check if input layers include any ReLU layer
            for(int i = 0; i < this->input_layers.size(); i++){
                if(this->input_layers[i]->is_input_layer == 0){
                    is_input = 0;   // there exists atleast one input layer which is not the network's input layer
                }
                if(this->input_layers[i]->activationFn == ReLU || this->input_layers[i]->activationFn == LReLU){
                    has_relu = 1;
                    // break;
                }
            }


            if(reserve){
                this->weight_inp_allocated = get_default_reserve_size(weight_inp);
                this->weight_out_allocated = (weight_out);
            }else{
                this->weight_inp_allocated = weight_inp;
                this->weight_out_allocated = weight_out;
            }
            this->weights.resize(this->weight_inp_allocated * this->weight_out_allocated);

            def_int_t rand_seed = get_rand_float()*1000;

            if(weight_inp == 0 || weight_out == 0){
                // print_err("Error weight dimensions are unknown.")
                if(TELEMETRY){
                    std::cout << "Error: init_weight-> weight dimensions are unknown, id=" << this->id << std::endl;
                }
                return 1;
            }


            if(random_values){
                if(has_relu){
                    // He initialization
                    // def_float_t std_dev = sqrt(2.0/1); 
                    def_float_t std_dev = sqrt(2.0/this->weight_inp); 
                    // Fill weights with random values based on the seed and normal distribution
                    for (int row = 0; row < weight_inp; row++) {
                        for (int col = 0; col < weight_out; col++){
                            def_float_t u1 = get_rand_float_seeded(rand_seed);
                            def_float_t u2 = get_rand_float_seeded((rand_seed++) + 1);
                            weights[flat_indx(row, col)] = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
                            weights[flat_indx(row, col)] *= std_dev;
                        
                        }
                    }
                    // for (int i = 0; i < weight_inp * weight_out; i++){
                    //     def_float_t u1 = get_rand_float_seeded(rand_seed);
                    //     def_float_t u2 = get_rand_float_seeded((rand_seed++) + 1);
                    //     weights[i] = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
                    //     weights[i] *= std_dev;
                    // }
                }else{
                    for (int row = 0; row < weight_inp; row++) {
                        for (int col = 0; col < weight_out; col++){
                            weights[flat_indx(row, col)] = get_rand_float_seeded(rand_seed++);
                        
                        }
                    }
                }

            }else{
                memset(weights.data(), 0, sizeof(def_float_t) * weight_inp_allocated * weight_out_allocated);
                // for(int i = 0; i < weight_inp_allocated * weight_out_allocated; i++){
                //     this->weights[i] = 0;
                // }
            }

            this->bias.clear();
            this->bias.resize(weight_out);
            for (int i = 0; i < weight_out; i++) {
                // this->bias.push_back(0);
                this->bias[i] = get_rand_float_seeded(rand_seed++);
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
            #if USE_SIMD
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
        
        #if USE_SIMD
            // TODO: Use SIMD here instead
        #else
            if(this->activationFn == Linear){
                // do nothing
            }else if(this->activationFn == LReLU){
                for(int i = 0; i < input_vector.size(); i++){
                    if(input_vector[i] < 0){
                        input_vector[i] *= leaky_relu_slope;
                    }else{
                        input_vector[i] *= 1;
                    }
                }
            }else if(this->activationFn == ReLU){
                for(int i = 0; i < input_vector.size(); i++){
                    if(input_vector[i] < 0){
                        input_vector[i] = 0;
                    }
                }
            }else if(this->activationFn == Exponential){    // TODO:
                std::cout << "Currently not supporting back prop on exponential activation fn" << std::endl;
                // for(int i = 0; i < input_vector.size(); i++){   // SIMD
                    
                // }
            }else if(this->activationFn == Sigmoid){        // TODO:
                std::cout << "Currently not supporting back prop on exponential activation fn" << std::endl;
                
            }else if(this->activationFn == Softmax){
                std::cout << "Currently not supporting back prop on exponential activation fn" << std::endl;
                
            }
        #endif
    }

    /**
     * @brief Automatically resize weights to match the input and output size of the layer.
     * @return 0 if successful, 1 if failed.
    */
    def_uint_small_t auto_grow_weight(){
        // Calculate current input weight size and output weight size and grow weights accordingly.
        this->fix_weights();
        return 0;

        
        if(this->is_input_layer != 1 && this->layer_type == Fully_Connected_INPUTS){
            def_uint_t new_weight_inp = 0;

            for(int i = 0; i < this->input_layers.size(); i++){
                new_weight_inp += this->input_layers[i]->x * this->input_layers[i]->y * this->input_layers[i]->z;
            }

            def_uint_t new_weight_out = this->x * this->y * this->z;

            if(TELEMETRY){
                std::cout << "auto_grow_weights for id=" << this->id << std::endl;
            }

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
            this->init_weight(1,1);
            return 0;
        }else{
            return 1;
        }
    }

    /**
     * @breif calculate the size of expected layer's inputs and outputs, and resize weights matrix with reserve
    */
    def_uint_t fix_weights(def_uint_t new_weight_inp = 0, def_uint_t new_weight_out = 0){
        def_uint_small_t random_weight_init = 1;
        def_uint_t seed1 = 1;


        if(this->layer_type == Fully_Connected_INPUTS){
            // calculate the size of expected layer's inputs and outputs, and resize weights matrix with reserve
            if (new_weight_inp == 0){
                for(int i = 0; i < this->input_layers.size(); i++){
                    new_weight_inp += this->input_layers[i]->size();
                }
            }
            if(new_weight_out == 0){
                new_weight_out = this->size();
            }

            if(TELEMETRY==2){
                std::cout << "fix_weights for id=" << this->id << std::endl;
            }

            // if any of it is 0, then return error
            if(new_weight_inp == 0 || new_weight_out == 0){
                if(TELEMETRY){
                    std::cout << "Error: fix_weights-> one of the dimension is 0, id=" << this->id << " size=(new_wi=" << new_weight_inp << ",new_wo=" << new_weight_out << ")     (old_wi=" << weight_inp << ",old_w=" << weight_out << ")" <<  std::endl;
                }
                return 1;
            }

            // handling bias
            if(new_weight_out > this->bias.size()){
                def_uint_t old_bias_size = this->bias.size();

                bias.resize(new_weight_out);
                if(random_weight_init){
                    for(int n = old_bias_size; n < new_weight_out; n++){
                        bias[n] = get_rand_float_seeded(seed1++);
                    }
                }else{
                    for(int n = old_bias_size; n < new_weight_out; n++){
                        bias[n] = 0;
                    }
                }
                
            }

            // change what is necessary
            if(new_weight_inp > this->weight_inp && this->weight_out_allocated > 0){
                // if already allocated, then just change the weight_inp
                def_uint_t old_weight_inp = this->weight_inp;
                if(new_weight_inp <= this->weight_inp_allocated){
                    this->weight_inp = new_weight_inp;
                }else{
                    // if not allocated, then allocate more
                    def_uint_t old_weight_inp_allocated = this->weight_inp_allocated;
                    this->weight_inp_allocated = get_default_reserve_size(new_weight_inp);
                    this->weights.resize(this->weight_inp_allocated * this->weight_out_allocated);

                    // shift the weights to conform as a flattened matrix
                    // for each yth row from last copying weights from block((y*old_weight_inp_allocated) <= i < ((y+1)*old_weight_inp_allocated) to postion(y*weight_inp_allocated)
                    // for each row from last to first
                    for(int row = weight_out - 1; row >= 0; row--){
                        // for each element as a block to make it vectorizable
                        for(int n = old_weight_inp_allocated - 1; n >= 0; n--){
                            // copying element from old index to new index
                            weights[row*weight_inp_allocated + n] = weights[(row*old_weight_inp_allocated + n)];
                        }
                    }
                    this->weight_inp = new_weight_inp;
                }
                if(random_weight_init){
                    // if random weight init, then initialize the new weights

                    def_uint_small_t init_method = 0;
                    if(this->activationFn == Sigmoid || this->activationFn == Exponential){
                        init_method = 1;
                    }else if(this->activationFn == ReLU || this->activationFn == LReLU){
                        init_method = 2;
                    }

                    if(fill_weight_value_initialization(this->weights, flat_indx(old_weight_inp, 0), flat_indx(this->weight_inp, this->weight_out), old_weight_inp, this->weight_out, init_method)){
                        // LEGACY: was uniform initialization below
                        for(int i = 0; i < this->weight_out; i++){
                            for(int j = old_weight_inp; j < this->weight_inp; j++){
                                this->weights[flat_indx(j, i)] = get_rand_float_seeded(seed1++);
                            }
                        }
                    }

                    // for(int i = 0; i < this->weight_out; i++){
                    //     for(int j = old_weight_inp; j < this->weight_inp; j++){
                    //         this->weights[flat_indx(j, i)] = get_rand_float_seeded(seed1++);
                    //     }
                    // }
                }
            }else if(new_weight_inp > this->weight_inp){ // this is when weight_out is 0, but just for safety
                this->weight_inp = new_weight_inp;
                this->weight_inp_allocated = get_default_reserve_size(new_weight_inp);
            }

            if(TELEMETRY == 2){
                std::cout << "current weight.flattened_size() = " << weights.size() << " size=(new_wi=" << new_weight_inp << ",new_wo=" << new_weight_out << ")     (old_wi=" << weight_inp << ",old_w=" << weight_out << ")" <<  std::endl;
            }

            if(new_weight_out > this->weight_out || (this->weight_out_allocated == 0 &&  this->weight_inp_allocated > 0)){
                def_uint_t old_weight_out = this->weight_out;
                def_uint_t old_weight_out_allocated = this->weight_out_allocated;
                // if already allocated, then just change the weight_out
                if(new_weight_out <= this->weight_out_allocated){
                    this->weight_out = new_weight_out;

                }else{
                    // if not allocated, then allocate more
                    def_uint_t old_weight_out_allocated = this->weight_out_allocated;
                    this->weight_out_allocated = get_default_reserve_size(new_weight_out);
                    
                    this->weights.resize(this->weight_inp_allocated * this->weight_out_allocated);

                    this->weight_out = new_weight_out;
                }
                if(random_weight_init){
                    // if random weight init, then initialize the new weights
                    def_uint_small_t init_method = 0;
                    if(this->activationFn == Sigmoid || this->activationFn == Exponential){
                        init_method = 1;
                    }else if(this->activationFn == ReLU || this->activationFn == LReLU){
                        init_method = 2;
                    }

                    if(fill_weight_value_initialization(this->weights, old_weight_out_allocated * this->weight_inp_allocated, this->weight_out_allocated * this->weight_inp_allocated, this->weight_inp, this->weight_out, init_method)){
                        // LEGACY: was uniform initialization below
                        for(int i = old_weight_out_allocated; i < this->weight_out; i++){
                            for(int j = 0; j < this->weight_inp; j++){
                                this->weights[flat_indx(j, i)] = get_rand_float_seeded(seed1++);
                            }
                        }
                    }

                }
            }
    

        }
        // std::cout << "Weights fixed!" << std::endl;

        return 0;
    }

    /**
     * @breif increase the size of weights matrix.
     * @param new_weight_inp The new number of columns in the weight matrix.
     * @param new_weight_out The new number of rows in the weight matrix.
     * @param random_values If 1, then initialize with random values, else initialize all with 0.
    */
    def_uint_small_t grow_weights(def_uint_t new_weight_inp, def_uint_t new_weight_out, def_uint_small_t randon_values){ //, def_uint_small_t reserve_new){
        fix_weights();
        return 0;


        def_uint_small_t reserve_new = 1;
        if(TELEMETRY){
            std::cout << "growing id=" << this->id << " to size=(new_wi=" << new_weight_inp << ",new_wo=" << new_weight_out << ")     (old_wi=" << weight_inp << ",old_w=" << weight_out << ")" <<  std::endl;
        }
        if(this->layer_type == Fully_Connected_INPUTS) {
            def_int_t cols_add = new_weight_inp - this->weight_inp;
            def_int_t rows_add = new_weight_out - this->weight_out;

            

            if(this->weight_inp == 0 || this->weight_out == 0){     // currently if any of the dimension == 0
                if(new_weight_inp == 0 || new_weight_out == 0){     // new atleast one of the weight dimension == 0
                    print_err("Error: Cannot grow weight matrix as one of the dimension is 0.")
                    return 1;
                }else if(new_weight_inp != 0 && new_weight_out != 0){   // new both weight dimensions != 0
                    this->weight_inp = new_weight_inp;
                    this->weight_out = new_weight_out;
                    this->init_weight(1,1);
                    return 0;
                }
            }

            // assuming Row Major Storing Matrix
            // handling number of rows first
            if(rows_add > 0){
                if(new_weight_out <= weight_out_allocated){
                    // if new_weight_out is less than allocated, then just increase the weight_out
                    this->weight_out = new_weight_out;
                }else{
                    // if new_weight_out is more than allocated, then increase the weight_out_allocated
                    if(reserve_new){
                        this->weight_out_allocated = get_default_reserve_size(new_weight_out);
                    }else{
                        this->weight_out_allocated = new_weight_out;
                    }
                    this->weight_out = new_weight_out;
                    this->weights.resize(this->weight_inp_allocated * this->weight_out_allocated);
                }
            }else if(cols_add < 0){
                print_err("Error: Cannot shrink weight rows of matrix. Use shrink matrix instead.");
            }

            // handling number of columns
            if(cols_add > 0){
                if(new_weight_inp <= weight_inp_allocated){
                    // if new_weight_inp is less than allocated, then just increase the weight_inp
                    this->weight_inp = new_weight_inp;
                }else{
                    def_uint_t delta_col_alloc = new_weight_inp - this->weight_inp_allocated;

                    def_uint_t old_weight_inp_allocated = this->weight_inp_allocated;
                    def_uint_t old_weight_inp = this->weight_inp;

                    if(reserve_new){
                        this->weight_inp_allocated = get_default_reserve_size(new_weight_inp);
                    }else{
                        this->weight_inp_allocated = new_weight_inp;
                    }

                    this->weight_inp = new_weight_inp;
                    this->weights.resize(this->weight_inp_allocated * this->weight_out_allocated);

                    // the weight matrix is now resized, now shifting weights to actual position as it is flattened
                    // for each yth row from last copying weights from block((y*old_weight_inp_allocated) <= i < ((y+1)*old_weight_inp_allocated) to postion(y*weight_inp_allocated)
                    // for each row from last to first
                    for(int row = weight_out - 1; row >= 0; row--){
                        // for each element in the block
                        for(int n = old_weight_inp - 1; n >= 0; n--){
                            // copying element from old index to new index
                            weights[row*weight_inp_allocated + n] = weights[(row*old_weight_inp_allocated + n)];

                        }
                        // std::copy(weights.data)
                    }



                    // if new_weight_inp is more than allocated, then increase the weight_inp_allocated
                    // if(reserve_new){
                    //     this->weight_inp_allocated = get_default_reserve_size(new_weight_inp);
                    // }else{
                    //     this->weight_inp_allocated = new_weight_inp;
                    // }
                    // this->weight_inp = new_weight_inp;
                    // this->weights.resize(this->weight_inp_allocated * this->weight_out_allocated);
                }

            }
        }
    }


    // // // DEPRECATED: uses old weight matrix format
    // def_uint_small_t grow_weights(def_uint_t new_weight_inp, def_uint_t new_weight_out, def_uint_small_t random_values){
    //     if(this->layer_type == Fully_Connected_INPUTS){
    //         // preserve data and insert additional columns or rows
    //         def_int_t cols_add = new_weight_inp - this->weight_inp;
    //         def_int_t rows_add = new_weight_out - this->weight_out;

    //         if(this->weight_inp == 0 || this->weight_out == 0){     // currently any of the dimension == 0
    //             if(new_weight_inp == 0 || new_weight_out == 0){     // new atleast one of the weight dimension == 0
    //                 print_err("Error: Cannot grow weight matrix as one of the dimension is 0.")
    //                 return 1;
    //             }else if(new_weight_inp != 0 && new_weight_out != 0){   // new both weight dimensions != 0
    //                 this->weight_inp = new_weight_inp;
    //                 this->weight_out = new_weight_out;
    //                 this->init_weight(1,1);
    //                 return 0;
    //             }
    //         }
    //         if(cols_add > 0){
    //             // add columns
    //             if(random_values){
    //                 for(int i = 0; i < this->weight_out; i++){
    //                     for(int j = 0; j < cols_add; j++){
    //                         this->weights.insert(this->weights.begin() + (i*this->weight_inp) + this->weight_inp + j, get_rand_float());
    //                     }
    //                 }
    //             }else{
    //                 for(int i = 0; i < this->weight_out; i++){
    //                     for(int j = 0; j < cols_add; j++){
    //                         this->weights.insert(this->weights.begin() + (i*this->weight_inp) + this->weight_inp + j, 0);
    //                     }
    //                 }
    //             }
    //         }else if(cols_add < 0){
    //             print_err("Error: Cannot shrink weight cols of matrix. Use shrink matrix instead.");
    //         }
    //         if(rows_add > 0){
    //             // add rows
    //             for(int i = 0; i < rows_add; i++){
    //                 for(int j = 0; j < this->weight_inp; j++){
    //                         this->weights.push_back(random_values ? get_rand_float() : 0);
    //                 }
    //             }
    //         }else if(rows_add < 0){
    //             print_err("Error: Cannot shrink weight rows of matrix. Use shrink matrix instead.");
    //         }
    //         this->weight_inp = new_weight_inp;
    //         this->weight_out = new_weight_out;
    //         return 0;
    //     }else{
    //         return 1;
    //     }
    // }

    #if USE_SIMD
    /**
     * @brief Multiply matrices in place, using AVX
     * @param A The first matrix
     * @param B The second matrix
     * @param output_vector The output vector
     * @param weight_inp The number of columns in A and rows in B
     * @param weight_out The number of rows in A and columns in B
     * @param batch_size The number of batches
    */
    void matrix_multiply(const def_float_t* A, const def_float_t* B, std::vector<def_float_t> output_vector, def_uint_t weight_inp, def_uint_t weight_out, def_uint_t batch_size){
        // TODO: make this setting global
        #define AVX_Size 256

        // check if weights are indeed stored as floats
        if(sizeof(def_float_t) == sizeof(float)){
            static def_uint_small_t AVX_num_floats = AVX_Size/sizeof(float);

            // clean the output vector, and initialize of size weight_out*batch_size with 0
            output_vector.resize(weight_out*batch_size);
            std::memset(output_vector.data(), 0, sizeof(def_float_t) * weight_out * batch_size);

            // for each batch
            for(int batch = 0; batch < batch_size; batch++){
                for(int m = 0; m < weight_out; m++){
                    __m256 sum = _mm256_setzero_ps();
                    int n = 0;
                    for(; n < weight_inp; n += AVX_num_floats){
                        __m256 a = _mm256_load_ps(A + batch*weight_inp + n);
                        __m256 b = _mm256_load_ps(B + m*weight_inp + n);
                        sum = _mm256_fmadd_ps(a, b, sum);
                    }
                    // if n is not a multiple of 8, then handle the remaining elements, given n is already greater than weight_inp using scalar operations
                    if(n < weight_inp){
                        def_float_t temp_sum[AVX_num_floats];
                        // initialize with 0
                        std::memset(temp_sum, 0, sizeof(def_float_t) * AVX_num_floats);
                        for(int i = 0; i < AVX_num_floats; i++){
                            if(n + i < weight_inp){
                                temp_sum[i] = A[batch*weight_inp + n + i] * B[m*weight_inp + n + i];
                            }
                        }
                        sum = _mm256_add_ps(sum, _mm256_load_ps(temp_sum));
                    }


                    output_vector[batch * weight_out + m] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
                }

            }
        }
    }
    #endif

    /**
     * @brief function for assisting debugging
    */
    void print_weights(){
        // takes care of reserved weights.
        std::cout << "Weight id=" << this->id << " weights.size()=" << this->weights.size() << " = (" << this->weight_inp <<  "x" << this->weight_inp<< "), allocated=(" << this->weight_inp_allocated << "," << this->weight_out_allocated << ")" << std::endl;
        for(int i = 0; i < this->weight_out; i++){
            for(int j = 0; j < this->weight_out; j++){
                std::cout << this->weights[flat_indx(i,j)] << " ";
            }
            std::cout << std::endl;
        }
    }

    /**
     * @brief a function to change neural network density as needed based on errors in current layer
    */
    def_int_t adjust_arch(){
        #if MAE_CALCULATION == 1
            if(this->is_dynamic_layer){
                // find the node with most error
                def_float_t error_nodes = 0;
                for(int indx = 0; indx < mae_count.size(); indx++){
                    if(max_inp_activ == 0){
                        max_inp_activ = 0.1;
                    }
                    def_float_t this_err = (mae_vec[indx]/mae_count[indx])/max_inp_activ;
                    // count the number of nodes with error more than threshold
                    if(this_err > mae_threshold && mae_count[indx] > MAE_Split_Min_training){
                        error_nodes++;
                        // half the current error values to not cross next run
                        mae_vec[indx] *= 0.5;
                    }
                }
                // increase the number of neurons in current layer
                if(this->layer_type == Fully_Connected_INPUTS){
                    if(y == 1 && z == 1){
                        if(error_nodes > 0){
                            if(TELEMETRY){
                                std::cout << "layer id=" << this->id << " resizing from " << this->size() << " to " << this->size() + error_nodes << std::endl;
                            }
                        }
                        this->x += error_nodes;
                        fix_weights();
                    }else{
                        print_err("cannot add neurons, as layer id=" << this->id << "is 2D (x,y,z)=("<<x<<","<<y<<","<<z<<"). Error nodes = " << error_nodes );
                    }
                }
                return error_nodes;
            }
            return 0;
        #endif
    }


    /**
     * @brief returns activation values of this layer's neurons.
    */
    std::vector<def_float_t> get_activation_rec(def_int_t run_id, def_uint_t batch_size){
        // this is memory inefficient, but faster to implement.
        if(this->cached_run_id == run_id || this->being_evaluated == 1){  // check whether to return directly from cache if already calculated.
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
            // if(weight_out != this->size(){
            //     grow_weights(weight_inp, this->size(), 1);
            // }

            // confirm if this layer weights are actually flattened appropriately
                            
            // build an array of input activation before calculating itself's activation
            std::vector<def_float_t> input_activations;

            // assuming the input_activations are in column_major (such that activations of single batch are contiguous)
            // collect activations of all layers and append to vector input_activations batch-wise
            
            // for(int i = 0; i < this->input_layers.size(); i++){
            //     std::vector<def_float_t> new_activation;
            //     new_activation = this->input_layers[i]->get_activation_rec(run_id, batch_size);
            //     input_activations.insert(input_activations.end(), new_activation.begin(), new_activation.end());
            // }

            // keep activations ready for each input layer to reduce memory overhead of recursion
            for(int indx = 0; indx < this->input_layers.size(); indx++){
                nlayer * this_layer = this->input_layers[indx];

                if(this_layer->cached_run_id != run_id || this_layer->cached_batch_size != batch_size){
                    this->input_layers[indx]->get_activation_rec(run_id, batch_size);
                }
            }
            
            // traverse layers for each batch
            for(int batch = 0; batch < batch_size; batch++){
                for(int indx = 0; indx < this->input_layers.size(); indx++){
                    nlayer * this_layer = this->input_layers[indx];
                    // take the cached values directly
                    if(this_layer->cached_run_id == run_id && this_layer->cached_batch_size == batch_size){
                        auto start_indx = this_layer->cached_activation_values.begin() + (batch * this_layer->size());
                        auto end_indx = (this_layer->cached_activation_values.begin() + ((batch + 1) * this_layer->size()));
                        input_activations.insert(input_activations.end(), start_indx, end_indx);
                    }else{
                        std::vector<def_float_t> new_activation;
                        new_activation = this->input_layers[indx]->get_activation_rec(run_id, batch_size);
                        auto start_indx = this_layer->cached_activation_values.begin() + (batch * this_layer->size());
                        auto end_indx = (this_layer->cached_activation_values.begin() + ((batch + 1) * this_layer->size()));
                        input_activations.insert(input_activations.end(), start_indx, end_indx);

                    }
                }    
            }

            this->fix_weights((input_activations.size()/batch_size), this->size());

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
                std::vector<def_float_t> output_vector;
                output_vector.resize(weight_out*batch_size);

                // printing this->weights
                if(TELEMETRY == 2) {
                    this->print_weights();
                }

                #if USE_SIMD
                matrix_multiply(this->weights.data(), input_activations.data(), output_vector, weight_inp, weight_out, batch_size);
                #else
                #if weight_row_major == 1
                    // FIXME: Make sure the output_vector is column major ( the activations of single batch sample are contiguous )
                    // performing matrix multiplication
                    for(int batch = 0; batch < batch_size; batch++){
                        for(int m = 0; m < weight_out; m++){
                            def_float_t result = 0;
                            for(int n = 0; n < weight_inp; n++){
                                result += input_activations[batch*weight_inp + n] * this->weights[flat_indx(n,m)]; //  previously: + 0*(m*weight_inp + n)
                            }
                            output_vector[batch * weight_out + m] = result;
                        }
                    }                    
                    // output_vector is column major, such that the activations of single batch sample are contiguous
                #else
                    // for (int batch = 0; batch < batch_size; batch++) {
                    //     for (int out = 0; out < weight_out; out++) {
                    //         def_float_t result = 0.0f;
                    //         for (int in = 0; in < weight_inp; in++) {
                    //             #if weight_row_major == 1
                    //                 result += input_activations[batch * weight_inp + in] * this->weights[flat_indx(in, out)];
                    //             #else
                    //                 result += input_activations[batch * weight_inp + in] * this->weights[in * weight_out + out];
                    //             #endif
                    //         }
                    //         output_vector[batch * weight_out + out] = result;
                    //     }
                    // }
                #endif
                #endif


                if(TELEMETRY == 2){ if(bias.size() != weight_out){ std::cout << "this->bias uninitialized. this=" << this << std::endl; } }
                
                // add bias to result
                // adding assuming column major
                for(int batch = 0; batch < batch_size; batch++){
                    for(int i = 0; i < weight_out; i++){
                        output_vector[batch * weight_out + i] += this->bias[i];
                    }
                }
                // for (int i = 0; i < weight_out*batch_size; i++) {
                //     output_vector[i] += this->bias[i%weight_out];   // add bias to all elements in the batch
                // }
                

                // apply activation function
                // output_vector = get_activation_Fn_value(output_vector);
                apply_activation_fn(output_vector, batch_size);

                // store copy of outputs to cache 
                this->cached_run_id = run_id;
                this->cached_activation_values = output_vector;  // confirmed: creates copy
                this->cached_batch_size = batch_size;

                #if TELE_PROP==1
                    std::cout << "FORWARD ACTIVATIONS of id=" << this->id << "  weight_out=" << this->weight_out << "  batch_size=" << batch_size << "  activations(row-major):" << std::endl;
                    // if(cached_activation_values.size() > 10){}
                    for(int i = 0; (i < cached_activation_values.size() && i < 12); i++){
                        std::cout << cached_activation_values[i] << "  ";
                    }
                    if(cached_activation_values.size() >= 12){
                        std::cout << "....";
                    }
                    std::cout << std::endl;
                #endif

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

    // def_uint_small_t retrieve_previous_inputs_col_major(std::vector<def_float_t> &collected_vec, def_uint_t batch_size){
    //     // assuming cached activation values are column major
    //     for(int batch = 0; batch < batch_size; batch++){
    //         for(int lindx = 0; lindx < this->input_layers.size(); lindx++){
    //             // append this layer's size worth of activations to 
    //         }
    //     }
    //     return 0;
    // }


    /**
     * @breif retrieve previous activations from corresponding input layers as batch-major
     * @param collected_vec reference to empty output vec batch-major flattened vec
     * @param batch_size uint of number of batches
    */
    def_uint_small_t retrieve_previous_inputs_batch_major(std::vector<def_float_t> &collected_vec, def_uint_t batch_size){
        // assuming cached activation values are column major
        for(int lindx = 0; lindx < this->input_layers.size(); lindx++){
            nlayer * this_layer = this->input_layers[lindx];
            if(this_layer->cached_batch_size != batch_size){
                print_err("Error: previous inputs batch size incorrect. id=" << this->id);
                return 1;
            }
            def_uint_t layer_size = this_layer->size();
            for(int n = 0; n < layer_size; n++){
                for(int batch = 0; batch < batch_size; batch++){                    
                    collected_vec.push_back(this_layer->cached_activation_values[(batch*layer_size)+n]);
                }
            }
        }
        return 0;
    }


    /**
     * @brief generate delta_weights in row-major form (same as weights) but without reserved spaces
     * @param delta_weights address of vec to store delta_weights
     * @param last_inputs address of vec with last_input values as batch-major
     * @param error_diff address of vec with error_diff values as batch-major
     * @param batch_size number of batches
    */
    def_uint_small_t calc_delta_weight(std::vector<def_float_t> &delta_weights, std::vector<def_float_t> &last_input, std::vector<def_float_t> &error_diff, def_uint_t batch_size){
        def_float_t sum = 0;
        // assuming error_diff and last-inputs as batch-major 
        for(int drow = 0; drow < weight_out; drow++){
            for(int acol = 0; acol < weight_inp; acol++){
                sum = 0;
                for(int batch = 0; batch < batch_size; batch++){
                    sum += error_diff[batch_size*drow + batch]*last_input[batch_size*acol + batch];
                }
                delta_weights[weight_inp * drow + acol] = sum;
            }
        }

        return 0;
    }

    #if MAE_CALCULATION == 1
    void print_accumulated_mae_normalized(){
        for(int n = 0; n < mae_count.size(); n++){
            if(mae_count.size() != 0){
                std::cout << ((mae_vec[n])/mae_count[n]) << "    ";
            }else{
                std::cout << 0 << "    ";
            }
        }
    }
    #endif

    /**
     * @brief Calculate the backprop error for this layer.
     * @param run_id The run_id of the current run.
     * @param batch_size The batch size of the current run.
     * @param activation_error The error of the next layer.
     * @param learning_rate The learning rate of the current run.
    */
    std::vector<def_float_t> get_correct_error_rec(def_int_t run_id, def_uint_t batch_size, std::vector<def_float_t>& activation_error, def_float_t learning_rate){
        if(!(this->being_corrected)){
            this->being_corrected = 1;
        }else{
            // this is detecting loop
            print_telm("Loop detected in calculating backprop error.");
            return(activation_error);
        }

        if(this->is_input_layer == 1){
            this->being_corrected = 0;
            return std::vector<def_float_t>(0,0);
        }

        #if TELE_PROP == 1
            std::cout << "BACKWARD PROPAGATION of id=" << this->id << "  weight_out=" << this->weight_out << "  batch_size=" << batch_size << "  errors(batch-major):" << std::endl;
            for(int i = 0; (i < activation_error.size() && i < 12); i++){
                std::cout << activation_error[i] << "  ";
            }
            if(activation_error.size() >= 12){
                std::cout << "....";
            }
            std::cout << std::endl;
        #endif

        if(this->layer_type == Fully_Connected_INPUTS){
            // check if forward prop caches errors are fresh, otherwise, wrong errors will be calculated.
            if(this->cached_run_id < run_id){
                if(TELEMETRY) {std::cout << "Uncalculated forward prop cache detected. this=" << this << std::endl;}
                this->get_activation_rec(run_id, batch_size);    // recalculating forward prop
            }

            // check if forward prop batch size is same as backprop batch size
            if(this->cached_batch_size != batch_size){
                print_err("Error: Forward prop batch size does not match backprop batch size.")
                this->being_corrected = 0;
                return activation_error;
            }


            // activation_error is the error in current layer // dZ_0 = A_l_0 - Y_ground-truth
            // std::vector<def_float_t> error_diff;    // dZ_0 = A_l_0 - Y_ground-truth
            // error_diff.reserve(activation_error.size());
            #if USE_SIMD    // TODO: Add SIMD instructions
            
            #else
                // getting difference (error) in current activation and expected activation
                // for(int i = 0; i < activation_error.size(); i++){
                //     error_diff.push_back(this->cached_acivation_values[i] - activation_error[i]);
                // }
                
                // notation: A_l_-1
                // generate a list of last inputs
                std::vector<def_float_t> last_inputs;   // A_l_-1
                last_inputs.reserve(this->weight_inp * batch_size);
                
                this->being_evaluated = 1;


                // expecting the cached_activation_values to be column major
                // DONE: rewrite such that last_inputs are in batch-major flattened format. 
                if(retrieve_previous_inputs_batch_major(last_inputs,batch_size) == 1){
                    // returned 1 means error
                    this->being_corrected = 0;
                    return activation_error;
                }


                // LEGACY: 
                // for(int i = 0; i < this->input_layers.size(); i++){
                //     // asking this layer's input_layers for their activation
                //     if(this->input_layers[i]->cached_batch_size == batch_size){
                //         // std::vector<def_float_t> new_activation = this->input_layers[i]->get_activation_rec(run_id, batch_size);
                //         last_inputs.insert(last_inputs.end(), this->input_layers[i]->cached_activation_values.begin(), this->input_layers[i]->cached_activation_values.end());
                //     }else{
                //         print_err("Error: Input layer batch size does not match backprop batch size.")
                //         this->being_corrected = 0;
                //         return activation_error;
                //     }
                // }

                this->being_evaluated = 0;



                std::vector<def_float_t> delta_weight;   // the dimensions are same as weights matrix
                delta_weight.resize(this->weight_inp * this->weight_out);

                def_float_t reci_batch_size = 1.0/batch_size;
                
                def_float_t sum = 0;

                // DONE: Rewrite code to generate delta_weights Matrix
                def_uint_small_t dW_status = calc_delta_weight(delta_weight, last_inputs, activation_error, batch_size);
                // // assumes that activation_error & last_inputs are in batch_major form
                // for(int drow = 0; drow < this->weight_out; drow++){
                //     for(int acol = 0; acol < this->weight_inp; acol++){
                //         sum = 0;
                //         for(int batch_num = 0; batch_num < batch_size; batch_num++){
                //             sum += activation_error[batch_size*drow + batch_num] * last_inputs[batch_size*acol + batch_num];
                //         }
                //         delta_weight[weight_inp*drow + acol] = sum;
                //     }
                // }


                // LEGACY: is not in current batch-major format
                // Matrix Multiply to get delta_weights
                // def_float_t sum = 0;
                // for(int col = 0; col < weight_inp; col++){
                //     for(int row = 0; row < weight_out; row++){
                //         sum = 0;
                //         for(int batch_num = 0; batch_num < batch_size; batch_num++){
                //             sum += activation_error[(batch_num*this->weight_out) + row] * last_inputs[(batch_num*this->weight_inp) + col];
                //         }

                //         delta_weight.push_back(sum * reci_batch_size);
                //     }
                // }

                if(TELEMETRY == 2){
                    std::cout << "# delta_weight = " << std::endl;
                    for (int i = 0; i < weight_inp; i++) {
                        for (int j = 0; j < weight_out; j++) {
                            std::cout << delta_weight[j * weight_inp + i] << " ";
                        }
                        std::cout << std::endl;
                    }
                }

                // calculate for Biases
                std::vector<def_float_t> delta_bias;    // empty vec
                delta_bias.resize(this->size());

                // DONE:
                // summing all errors for each neurons across the batch
                // assuming that activation_error are batch-major

                for(int n = 0; n < this->size(); n++){
                    def_float_t bias_error_sum = 0;
                    for(int batch = 0; batch < batch_size; batch++){
                        bias_error_sum += activation_error[n*batch_size + batch];
                    }
                    delta_bias[n] = bias_error_sum * reci_batch_size;
                }

                

                // for (int i = 0; i < weight_out; i++) {
                //     def_float_t sum = 0;
                //     for (int j = 0; j < batch_size; j++) {
                //         sum += activation_error[j * weight_out + i];
                //     }
                //     delta_bias.push_back(sum * reci_batch_size);
                // }

                if(TELEMETRY == 2){
                    std::cout << "# delta_bias = " << std::endl;
                    for (int i = 0; i < weight_out; i++) {
                        std::cout << delta_bias[i] << " ";
                    }
                    std::cout << std::endl;
                }

                std::vector<def_float_t> old_weights = this->weights;


                // #if MAE_CALCULATION == 1
                //     std::vector<def_float_t> reci_node_age(1,this->size());

                //     if(mae_count.size() == this->size()){
                //         for(int n = 0; n < mae_count.size(); n++){
                //             if(mae_count[n] != 0){
                //                 reci_node_age[n] = 1.0/mae_count[n];
                //             }else{
                //                 reci_node_age[n] = 1;
                //             }
                //         }
                //     }else{
                //         print_err("warning!")
                //     }

                //     // DONE: Verify if this is in correct form
                //     // update weights
                //     for(int i = 0; i < weight_inp; i++){
                //         for(int j = 0; j < weight_out; j++){
                //             this->weights[ flat_indx(i,j) ] += (delta_weight[ weight_inp * j + i ] * learning_rate * reci_node_age[j]);
                //         }
                //     }

                //     // // update bias
                //     for(int i = 0; i < weight_out; i++){
                //         this->bias[i] -= delta_bias[i] * learning_rate * reci_node_age[i];
                //     }
                // #else
                    // update weights
                    for(int i = 0; i < weight_inp; i++){
                        for(int j = 0; j < weight_out; j++){
                            this->weights[ flat_indx(i,j) ] += (delta_weight[ weight_inp * j + i ] * learning_rate);
                        }
                    }

                    // // update bias
                    for(int i = 0; i < weight_out; i++){
                        this->bias[i] -= delta_bias[i] * learning_rate;
                    }
                // #endif

                // input_dz = (W.T x dZ) * g'(Z)

                // finding error for input layer
                std::vector<def_float_t> input_error;
                input_error.reserve(this->weight_inp * batch_size);


                def_float_t inp_error_sum;
                // FIXME: Make sure that the calculated input_errors are batch-major for fast access, and vectorizability
                for(int m = 0; m < weight_inp; m++){
                    for(int batch = 0; batch < batch_size; batch++){
                        inp_error_sum = 0;
                        for(int n = 0; n < weight_out; n++){
                            inp_error_sum += old_weights[weight_inp_allocated*n + m] * activation_error[batch_size*n + batch];
                        }
                        input_error.push_back(inp_error_sum);
                    }
                }
                // Broken below:
                // for(int drow = 0; drow < weight_out; drow++){
                //     for(int batch = 0; batch < batch_size; batch++){
                //         inp_error_sum = 0;
                //         for(int wi = 0; wi < weight_inp; wi++){
                //             inp_error_sum += old_weights[];
                //         }
                //         input_error.push_back(inp_error_sum);
                //     }
                // }
                // find the inverse transformation of weights matrix
                // matrix multiply activation_error and last_inputs
                // for(int i = 0; i < this->weight_inp; i++){
                //     for(int j = 0; j < this->weight_out; j++){
                //         def_float_t sum = 0;
                //         for(int k = 0; k < batch_size; k++){
                //             sum += activation_error[k*this->weight_out + j] * old_weights[flat_indx(i,j)];
                //         }
                //         input_error.push_back(sum);
                //     }
                // }
                #if MAE_CALCULATION == 1
                // calculate MAE across past batches
                    def_uint_t my_size = this->size();
                    if(mae_count.size() < my_size){
                        // initialize new size
                        for(int n = mae_count.size(); n < my_size; n++){
                            mae_count.push_back(0);
                            mae_vec.push_back(0.0);
                        }
                    }else if(mae_count.size() > my_size){
                        print_err("mae shows layer shrink, TODO: ");
                    }
                    // NOTE: Bottleneck 
                    for(int n = 0; n < last_inputs.size(); n++){
                        if(last_inputs[n] > max_inp_activ){
                            max_inp_activ= last_inputs[n];
                        }
                    }

                    // add errors and 1 to mae_count to all nodes
                    for(int n = 0; n < my_size; n++){   // nth 
                        mae_count[n] += batch_size;
                        // mae_vec += abs()
                        for(int batch = 0; batch < batch_size; batch++){
                            mae_vec[n] += abs(activation_error[batch_size*n + batch]);
                        }
                    }

                    if(TELEMETRY == 2){
                        print_accumulated_mae_normalized();
                    }

                #endif

                // multiply the derivative of activation function with input_error
                multiply_activation_derivative_fn(input_error);

                def_uint_t inp_split_count = 0; // number of neurons errors given to

                // splitting input corrections to their corresponding layers, assuming input errors are in batch-major ordering.
                for(int lindx = 0; lindx < this->input_layers.size(); lindx++){
                    nlayer * this_layer = this->input_layers[lindx];

                    std::vector<def_float_t> this_errors;
                    this_errors.reserve(batch_size*this_layer->size());


                    // slice values from input_errors
                    def_uint_t slice_start = batch_size*inp_split_count;
                    def_uint_t slice_end = batch_size*(inp_split_count+this_layer->size());
                    inp_split_count += this_layer->size();

                    for(int n = slice_start; n < this_layer->size(); n++){
                        for(int batch = 0; batch < batch_size; batch++){
                            this_errors.push_back(input_error[batch_size*n + batch]);
                        }
                        // this_errors.insert(input_error)

                    }
                    this->input_layers[lindx]->get_correct_error_rec(run_id, batch_size, this_errors, learning_rate);
                }

                this->adjust_arch();

                // LEGACY: below code assumes column-major instead of batch-major
                // for(int i = 0; i < (this->input_layers.size()); i++) {  // for each layer
                //     std::vector<def_float_t> this_errors;
                //     def_uint_t this_layer_output_size = 0;



                //     if(this->input_layers[i]->layer_type == Fully_Connected_INPUTS){
                //         this_layer_output_size = this->input_layers[i]->weight_out;

                //         this_errors.reserve((this_layer_output_size) * batch_size);
                //     }

                //     def_uint_t start_inp = input_split_ptr;
                //     def_uint_t end_inp = input_split_ptr + this_layer_output_size;

                //     // also for each batch data point. basically taking out a slice from a flattened 2D matrix
                //     for(int batch = 0; batch < batch_size; batch++){
                //         def_int_t start_range = start_inp + batch * this->weight_inp;
                //         def_int_t end_range = end_inp + batch * this->weight_inp;

                //         if(TELEMETRY == 2){
                //             std::cout << "input_error.size()=" << input_error.size() << std::endl;
                //             std::cout << "this_errors.size()=" << this_errors.size() << std::endl;
                //         }

                //         // FIXME: faster method crashes with larger array sizes
                //         // pasting behind the array this_errors
                //         // this_errors.insert(this_errors.end(),
                //         //     input_error.begin() + start_range * sizeof(def_float_t),
                //         //     input_error.begin() + (end_range + 1) * sizeof(def_float_t));

                //         // inserting manually
                //         for(int el = 0; el < input_error.size(); el++){
                //             this_errors.push_back(input_error[el]);
                //         }

                //     }

                //     this->input_layers[i]->get_correct_error_rec(run_id, batch_size, this_errors, learning_rate);
                    
                // }





            #endif


        }else if(this->layer_type== Convolutional_INPUTS){

        }

        
        this->being_corrected = 0;
        return std::vector<def_float_t>(0,0);
    }
};





// } // namespace std
