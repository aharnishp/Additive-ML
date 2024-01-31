#ifndef NNETWORK_HPP    // FIXME: Check all import states
#define NNETWORK_HPP
#include "nnetwork.hpp"
#endif
#include "nlayer.hpp"

#include <vector>
#include <algorithm>
#include <string>
#include <set>

#include <iostream>
#include <fstream>
// #ifdef __unix__
//     #include <fstream>
// #endif

// #define def_float_t double


using namespace std;
class nnetwork{
    private:
    def_int_t run_id = 0;

    void print1D(std::vector<def_float_t> vec){
        for(int i = 0; i < vec.size(); i++){
            std::cout << vec[i] << " ";
        }std::cout << std::endl;
    }


    // void append_float_to_char_vector(def_float_t val, vector<char> &output){
    //     // append float to char vector

    //     // using reinterpret_cast
    //     output.insert(output.end(), reinterpret_cast<char*>(&val), reinterpret_cast<char*>(&val) + sizeof(def_float_t));
    //     // int size_of_def_float_t = sizeof(def_float_t);

    //     // // using memcpy
    //     // char *val_char = new char[size_of_def_float_t];
    //     // memcpy(val_char, &val, size_of_def_float_t);
    //     // for(int i = 0; i < size_of_def_float_t; i++){
    //     //     output.push_back(val_char[i]);
    //     // }
    // }

    public:

    /**
     * @brief returns a char array including all properties of the nlayer (including weights)
     * @param inp_layer pointer to the nlayer
    */
    vector<char> get_nlayer_export(nlayer * inp_layer){
        // generate a string including all properties of the nlayer

        int size_of_def_uint_t = sizeof(def_uint_t);
        int size_of_def_float_t = sizeof(def_float_t);

        vector<char> output;
        
        // id
        output.push_back(inp_layer->id);
        output.push_back(',');
        
        // layer type
        output.push_back(inp_layer->layer_type);
        output.push_back(',');

        if(inp_layer->layer_type == 0){
            // input layer
            output.push_back(inp_layer->x);
                output.push_back(',');
            output.push_back(inp_layer->y);
                output.push_back(',');
            output.push_back(inp_layer->z);
                output.push_back(',');
        }

        // number of input layers
        def_uint_t num_inp_layers = inp_layer->input_layers.size();
        output.insert(output.end(), reinterpret_cast<char*>(&num_inp_layers), reinterpret_cast<char*>(&num_inp_layers) + size_of_def_uint_t);
        output.push_back(',');

        output.push_back('{'); // input_layers
            for(int i = 0; i < inp_layer->input_layers.size(); i++){
                // store as little endian into size of def_uint_t
                for(int j = 0; j < size_of_def_uint_t; j++){
                    output.push_back((inp_layer->input_layers[i]->id >> (j*8)) & 0xFF);
                }

                if(i != inp_layer->input_layers.size()-1)
                    output.push_back(',');
            }

        output.push_back('}');
        output.push_back(',');

        // store activation function
        output.push_back(inp_layer->activationFn);

        output.push_back(',');

        def_float_t this_learning_rate = inp_layer->learning_rate;

        // append learning rate as float
        output.insert(output.end(), reinterpret_cast<char*>(&this_learning_rate), reinterpret_cast<char*>(&this_learning_rate) + size_of_def_float_t);
        // for(int i = 0; i < size_of_def_float_t; i++){
        //     // store as little endian
        //     // copy each byte in host order
        //     output.insert(output.end(), reinterpret_cast<char*>(&this_learning_rate) + i, reinterpret_cast<char*>(&this_learning_rate) + i + 1);
        // }
        output.push_back(',');

        // store weight_inp
        output.insert(output.end(), reinterpret_cast<char*>(&inp_layer->weight_inp), reinterpret_cast<char*>(&inp_layer->weight_inp) + size_of_def_uint_t);
        output.push_back(',');

        // store weight_out
        output.insert(output.end(), reinterpret_cast<char*>(&inp_layer->weight_out), reinterpret_cast<char*>(&inp_layer->weight_out) + size_of_def_uint_t);
        output.push_back(',');

        // store weights
        output.push_back('{');  // not including seperating commas, as faster for batch copy
            for(int i = 0; i < inp_layer->weights.size(); i++){
                //  FIXME:
                output.insert(output.end(), size_of_def_float_t, inp_layer->weights[i]);
            }
            
            // use batch copy to quickly copy all weights

            // output.insert(output.end(), reinterpret_cast<char*>(inp_layer->weights.data()), reinterpret_cast<char*>(inp_layer->weights.data() + inp_layer->weights.size()*sizeof(def_float_t)));
        output.push_back('}');

        return output;
    }

    /**
     * @brief fills in layer at the end of given input array
     * @param inp_layer pointer to the nlayer
     * @param output char vector to be filled
    */
    void export_nlayer_to_array(nlayer * inp_layer, std::vector<char> &output){
        // generate a string including all properties of the nlayer

        int size_of_def_uint_t = sizeof(def_uint_t);
        int size_of_def_float_t = sizeof(def_float_t);

        // vector<char> output;
        
        // id
        output.push_back(inp_layer->id);
        output.push_back(',');
        
        // layer type
        output.push_back(inp_layer->layer_type);
        output.push_back(',');

        if(inp_layer->layer_type == 0){
            // input layer
            output.push_back(inp_layer->x);
                output.push_back(',');
            output.push_back(inp_layer->y);
                output.push_back(',');
            output.push_back(inp_layer->z);
                output.push_back(',');
        }

        // number of input layers
        def_uint_t num_inp_layers = inp_layer->input_layers.size();
        output.insert(output.end(), reinterpret_cast<char*>(&num_inp_layers), reinterpret_cast<char*>(&num_inp_layers) + size_of_def_uint_t);
        output.push_back(',');

        output.push_back('{'); // input_layers
            for(int i = 0; i < inp_layer->input_layers.size(); i++){
                // store as little endian into size of def_uint_t
                for(int j = 0; j < size_of_def_uint_t; j++){
                    output.push_back((inp_layer->input_layers[i]->id >> (j*8)) & 0xFF);
                }

                if(i != inp_layer->input_layers.size()-1)
                    output.push_back(',');
            }

        output.push_back('}');
        output.push_back(',');

        // store activation function
        output.push_back(inp_layer->activationFn);

        output.push_back(',');

        def_float_t this_learning_rate = inp_layer->learning_rate;

        // append learning rate as float
        output.insert(output.end(), reinterpret_cast<char*>(&this_learning_rate), reinterpret_cast<char*>(&this_learning_rate) + size_of_def_float_t);
        // for(int i = 0; i < size_of_def_float_t; i++){
        //     // store as little endian
        //     // copy each byte in host order
        //     output.insert(output.end(), reinterpret_cast<char*>(&this_learning_rate) + i, reinterpret_cast<char*>(&this_learning_rate) + i + 1);
        // }
        output.push_back(',');

        // store weight_inp
        output.insert(output.end(), reinterpret_cast<char*>(&inp_layer->weight_inp), reinterpret_cast<char*>(&inp_layer->weight_inp) + size_of_def_uint_t);
        output.push_back(',');

        // store weight_out
        output.insert(output.end(), reinterpret_cast<char*>(&inp_layer->weight_out), reinterpret_cast<char*>(&inp_layer->weight_out) + size_of_def_uint_t);
        output.push_back(',');

        // store weights
        output.push_back('{');  // not including seperating commas, as faster for batch copy
            for(int i = 0; i < inp_layer->weights.size(); i++){
                // FIXME: check if this actually works
                output.insert(output.end(), size_of_def_float_t, inp_layer->weights[i]);
            //     output.insert(output.end(), reinterpret_cast<char*>(inp_layer->weights[i].data()), reinterpret_cast<char*>(inp_layer->weights[i].data() + inp_layer->weights[i].size()*size_of_def_float_t));
            }
            
            // use batch copy to quickly copy all weights
            // output.insert(output.end(), reinterpret_cast<char*>(inp_layer->weights.data()), reinterpret_cast<char*>(inp_layer->weights.data() + inp_layer->weights.size()*sizeof(def_float_t)));
        output.push_back('}');

        // return output;
    }

    public:
    nlayer *input_layer;
    nlayer *output_layer;

    std::vector<nlayer *> input_layers;
    std::vector<nlayer *> output_layers;
    
    def_uint_t next_assigning_id = 0;

    def_float_t default_learning_rate = 0.01;
    
    nnetwork(){
        this->input_layer = NULL;
        this->output_layer = NULL;
    }

    nnetwork(def_uint_t input_layer_size, def_uint_t output_layer_size){
        this->input_layer = new nlayer(input_layer_size);
        this->output_layer = new nlayer(output_layer_size);

        input_layer->id=0;
        output_layer->id=1;
        next_assigning_id = 2;

        input_layer->is_input_layer = 1;
        output_layer->add_input_layer(input_layer);
        // output_layer->init_weight(1);
    }

    nnetwork(def_uint_t input_layer_size, def_uint_t output_layer_size, def_float_t learning_rate){
        this->input_layer = new nlayer(input_layer_size);
        this->output_layer = new nlayer(output_layer_size);
        this->default_learning_rate = learning_rate;

        input_layer->id=0;
        output_layer->id=1;
        next_assigning_id = 2;
        

        input_layer->is_input_layer = 1;
        output_layer->add_input_layer(input_layer);
        // output_layer->init_weight(1);
    }

    nnetwork(nlayer *input_layer, nlayer *output_layer){
        this->input_layer = input_layer;
        this->output_layer = output_layer;
    }


    def_int_t get_run_id(){
        return this->run_id;
    }

    /**
     * @brief adds a new fully connected layer just before the output layer
     * @param layer_size size of the new layer (assumming fully connected)
     * @param activation_function activation function of the new layer
     * @param learning_rate learning rate of the new layer
    */
    def_uint_t add_layer_between_output(def_uint_t layer_size, activation_fn_t activation_function, def_float_t learning_rate){
        /*
        adds a new layer at the end of the network.
        returns the id of the newly added layer.
        */
        nlayer *new_layer = new nlayer(layer_size, activation_function, learning_rate);
        
        new_layer->id = next_assigning_id;
        next_assigning_id++;

        new_layer->input_layers = this->output_layer->input_layers;
        this->output_layer->input_layers =  {new_layer};
        new_layer->auto_grow_weight();
        output_layer->weights.clear();
        output_layer->weight_inp = new_layer->size();
        output_layer->weight_out = output_layer->size();
        return new_layer->id;
    }

    /**
     * @brief adds a new layer just before and connected to the output layer of network.
     * @param layer_size size of the new layer (assumming fully connected)
     * @param activation_function activation function of the new layer
     * @param learning_rate learning rate of the new layer
    */
    def_uint_small_t add_new_layer_at_last(def_uint_t layer_size, activation_fn_t activation_function, def_float_t learning_rate){
        /*
        adds a new layer at the end of the network.
        returns the id of the newly added layer.
        */
        nlayer *new_layer = new nlayer(layer_size, activation_function, learning_rate);
        
        new_layer->id = next_assigning_id;
        next_assigning_id++;

        // new_layer->input_layers = this->output_layer->input_layers);
        // new_layer->init_weight(1);
        // this->output_layer = new_layer;
        new_layer->add_input_layer(this->output_layer);
        new_layer->init_weight(1,1);
        this->output_layer = new_layer;
        return new_layer->id;
    }
    
    /**
     * @brief given the input_values, returns the predicted value of the network.
     * @param input_values flattened 1D vector of the 2D array formed by input_layer.size * batch_size
     * @param batch_size
    */
    std::vector<def_float_t> forward_prop(std::vector<def_float_t>& input_values, def_uint_t batch_size) {
        /*
        returns the predicted value of the network for the given input value.
        (returned value is flattened 1D array of output_layer.size * batch_size
        */
        run_id++;

        // check if there is only a single input
        if(this->input_layers.size() > 1){
            // distribute inputs to each input layer
            int input_size = input_values.size() / batch_size;
            for(int inp_layer_indx = 0; inp_layer_indx < this->input_layers.size(); inp_layer_indx++){
                std::vector<def_float_t> this_input;
                for(int i = 0; i < batch_size; i++){
                    // for(int j = 0; j < input_size; j++){
                    //     this_input.push_back(input_values[i*input_size + j]);
                    // }
                    this_input.insert(this_input.end(), input_values.begin() + i*input_size, input_values.begin() + (i+1)*input_size);
                }
                this->input_layers[inp_layer_indx]->cached_activation_values = this_input;
                this->input_layers[inp_layer_indx]->cached_batch_size = batch_size;

            }
        }else{
            this->input_layer->cached_activation_values = input_values;
            this->input_layer->cached_batch_size = batch_size;
        }

        // check number of output layers in the network
        if(this->output_layers.size() > 1){
            // combine outputs from each output layer, after getting activation for current run_id
            std::vector<def_float_t> output_values;
            for(int out_layer_indx = 0; out_layer_indx < this->output_layers.size(); out_layer_indx++){
                std::vector<def_float_t> this_output = this->output_layers[out_layer_indx]->get_activation_rec(run_id,batch_size);
                output_values.insert(output_values.end(), this_output.begin(), this_output.end());
            }
            return output_values;
        }else if(this->output_layers.size() == 0){
            return (this->output_layer->get_activation_rec(run_id,batch_size));
        }

        // return this->output_layer->cached_activation_values;        
    }

    /**
     * @brief given the input_values, returns the predicted value of the network.
     * @param input_values flattened 1D vector of the 2D array formed by input_layer.size * batch_size
     * @param batch_size
    */

    /**
     * @brief calculates output layer error and recursively corrects the error in the previous layers till the input layer.
     * @param input_values flattened 1D vector of the 2D array formed by input_layer.size * batch_size
     * @param expected_values flattened 1D vector of the 2D array formed by output_layer.size * batch_size
     * @param batch_size
    */
    std::vector<def_float_t> backward_prop(std::vector<def_float_t>& input_values, std::vector<def_float_t>& expected_values, def_uint_t batch_size) {

        run_id++;
        this->input_layer->cached_activation_values = input_values;
        this->input_layer->cached_batch_size = batch_size;

        this->output_layer->get_activation_rec(run_id,batch_size);

        if(TELEMETRY == 2){
            std::cout << "input_values.size()= " << input_values.size() << std::endl;
            std::cout << "expected_values.size()= " << expected_values.size() << std::endl;
        }

        if(input_values.size() != this->input_layer->x * this->input_layer->y * this->input_layer->z * batch_size){
            if(TELEMETRY){ 
                std::cout << "ERROR: input_values dimension not correct. input_layer.size()*batch_size = " << this->input_layer->x * this->input_layer->y * this->input_layer->z * batch_size <<
                " but input_values.size() = " << input_values.size() << std::endl; 
            }
            return {0};
        }
        if(expected_values.size() != this->output_layer->x * this->output_layer->y * this->output_layer->z * batch_size){
            if(TELEMETRY){
                std::cout << "ERROR: expected_values dimension not correct. output_layer.size()*batch_size = " << this->output_layer->x * this->output_layer->y * this->output_layer->z * batch_size <<
                " but input_values.size() = " << expected_values.size() << std::endl; 
            }
            return {0};

        }

        std::vector<def_float_t> error_in_prediction;
        // for(int i = 0; i < expected_values.size(); i++){
        //     error_in_prediction.push_back(expected_values[i] - this->output_layer->cached_activation_values[i]);
        // }
        // std::transform(this->output_layer->cached_activation_values.begin(), this->output_layer->cached_activation_values.end(), expected_values.begin(), std::back_inserter(error_in_prediction), std::minus<def_float_t>());

        // FIXME: Make sure the calculations are batch-major (errors across multiple batches are contiguous per neuron)
        for(int i = 0; i < expected_values.size(); i++){
            error_in_prediction.push_back(this->output_layer->cached_activation_values[i] - expected_values[i]);
            // error_in_prediction.push_back(expected_values[i] - this->output_layer->cached_activation_values[i]);
        }
        
        // print error_in_prediction
        if(TELEMETRY == 2){
            std::cout << "prediction error: ";
            print1D(error_in_prediction);
        }

        if(this->output_layer != NULL){
            std::vector<def_float_t> error_in_input = this->output_layer->get_correct_error_rec(run_id, batch_size, error_in_prediction, this->default_learning_rate);
        }else{
            std::cout << "output_layer pointer is NULL" << std::endl;
        }

        return error_in_prediction;
    }








    /**
     * @brief Prints the network architecture in BFS from output node
    */
    void print_architecture(){
        std::cout << "printing nnetwork architecture" << std::endl;
        std::set<nlayer*> visited;
        std::vector<nlayer*> unvisited;
        unvisited.push_back(this->output_layer);
        while(unvisited.size()){
            nlayer* this_layer = unvisited[unvisited.size()-1];
            unvisited.pop_back();   // remove from last
            visited.insert(this_layer);   // insert this to visted
                       
            std::cout << "# layer-id=" << this_layer->id << " \tlayer-size=" << this_layer->size() << " \tinput_layers:" << std::endl;
            std::cout << "    printing weights(" << this_layer->weight_inp << " x " << this_layer->weight_out << ")" << std::endl;


            for(int i = 0; i < this_layer->input_layers.size(); i++){
                std::cout << "    -layer"  << this_layer->input_layers[i]->id << std::endl;
                if(visited.find(this_layer->input_layers[i]) == visited.end() && std::find(unvisited.begin(), unvisited.end(), this_layer->input_layers[i]) == unvisited.end()){
                    unvisited.push_back(this_layer->input_layers[i]);
                }
            }
            visited.insert(this_layer);
        }

    }

    private:
    void insert_def_float_t(std::vector<char> &input_arr, def_float_t input_val){
        size_t float_size = sizeof(def_float_t);
        char* float_bytes = reinterpret_cast<char*>(&input_val);
        
        for(size_t i = 0; i < float_size; i++){
            input_arr.push_back(float_bytes[i]);
        }
    }

    void insert_def_int_t(std::vector<char> &input_arr, def_int_t input_val) {
        size_t int_size = sizeof(def_int_t);
        char* int_bytes = reinterpret_cast<char*>(&input_val);

        for (size_t i = 0; i < int_size; i++) {
            input_arr.push_back(int_bytes[i]);
        }
    }

    void insert_def_uint_t(std::vector<char> &input_arr, def_uint_t input_val) {
        size_t uint_size = sizeof(def_uint_t);
        char* uint_bytes = reinterpret_cast<char*>(&input_val);

        for (size_t i = 0; i < uint_size; i++) {
            input_arr.push_back(uint_bytes[i]);
        }
}


    public:


    //   ______                       _
    //  |  ____|                     | |
    //  | |__  __  ___ __   ___  _ __| |_
    //  |  __| \ \/ / '_ \ / _ \| '__| __|
    //  | |____ >  <| |_) | (_) | |  | |_
    //  |______/_/\_\ .__/ \___/|_|   \__|
    //              | |
    //              |_|
    //   _          ______ _ _
    //  | |        |  ____(_) |
    //  | |_ ___   | |__   _| | ___
    //  | __/ _ \  |  __| | | |/ _ \ 
    //  | || (_) | | |    | | |  __/
    //   \__\___/  |_|    |_|_|\___|

    /**
     * @brief export all the layers of the network to a file
     * @param filepath string of path to the file to be exported
    */
    def_uint_small_t export_nnetwork_to_file(string filepath){
        string File_Header;

        if(TELEMETRY){
            std::cout << "exporting nnetwork to " << filepath << std::endl;
        }

        //      _  _ ____ ____ ___  ____ ____
        //      |__| |___ |__| |  \ |___ |__/
        //      |  | |___ |  | |__/ |___ |  \ 
        //      insert header to the export buffer
        
        // store network file to buffer
        std::vector<char> export_buffer = {'A','N','N',','};    // init with the file magic number
        // TODO: Add header binaries

        // export the data version in single byte
        export_buffer.push_back((char)DEFAULT_LAYER_VERSION);

        // insert the size of def_uint_small_t    
        export_buffer.push_back((char)sizeof(def_uint_small_t));
    
        // insert the size of def_int_t
        export_buffer.push_back((char)sizeof(def_int_t));

        // insert the size of def_float_t
        export_buffer.push_back((char)sizeof(def_float_t)); 

        // export byte representing little endian byte ordering
        #if defined(__x86_64__) || defined(__aarch64__) || defined(__arm__)
            export_buffer.push_back((char)1);
        #else
            print_err("Unable to detect architecture.")
        #endif
         
        // check if the ids of output & input are unique
        if(this->input_layer->id==this->output_layer->id){
            this->output_layer->id++;
        }

        // export the id of output layer
        insert_def_uint_t(export_buffer, this->output_layer->id);
        //export the id of input layer
        insert_def_uint_t(export_buffer, this->input_layer->id);


        //    ___ ____ ____ _  _ ____ ____ ____ ____
        //     |  |__/ |__| |  | |___ |__/ [__  |___
        //     |  |  \ |  |  \/  |___ |  \ ___] |___
        //    traverse each layer recursively in BFS to store pointers to all layers in visited array.

        // recursively visit all layers, and add to to_be_printed, while making sure, there is no matching id, in that case, add to the highest one.
        // std::set<def_uint_t> visited_ids;
        def_uint_t max_id = 0;

        std::vector<nlayer*> unvisited;
        std::vector<nlayer *> visited;

        unvisited.push_back(this->output_layer);
        max_id = output_layer->id;

        // open fstream file

        while(unvisited.size()){
            nlayer* this_layer = unvisited[unvisited.size()-1]; // get current layer
            unvisited.pop_back();   // remove self from unvisited
            visited.push_back(this_layer);  // add self to visited
            if(max_id < this_layer->id){
                max_id = this_layer->id + 1;
            }
            if(TELEMETRY){
                std::cout << "exporting layer id=" << this_layer->id << std::endl;
            }
            for(int i = 0; i < this_layer->input_layers.size(); i++){
                // checking if the input layers have unique id's or not (and actually different but have same id).
                for(int j = 0; j < visited.size(); j++){
                    // keeping max_id as large as possible
                    if(max_id < this_layer->id){
                        max_id = this_layer->id + 1;
                    }
                    if(visited[j]->id==this_layer->input_layers[i]->id && visited[j] != this_layer->input_layers[i]){
                        if(TELEMETRY){
                            std::cout << "repeated id detected of a layer, assigned new id=" << max_id << std::endl;;
                        }
                        // assign new id to this "j" duplicate layer
                        this_layer->input_layers[j]->id = max_id;
                        max_id++;
                    }
                }

                // Adding this layer's input layers to unvisited queue
                if(std::find(visited.begin(), visited.end(), this_layer->input_layers[i]) == visited.end()){
                    unvisited.push_back(this_layer->input_layers[i]);
                }
            }

            // insert this_layer to export_buffer
            vector<char> this_export = get_nlayer_export(this_layer);
            export_buffer.insert(export_buffer.end(), this_export.begin(), this_export.end());
        }

        // store the export buffer to file filepath
        std::ofstream file(filepath);
        if(file.is_open()){
            file.write((char*)export_buffer.data(), export_buffer.size());
            file.close();
        }else{
            std::cout << "could not open file " << filepath << std::endl;
            return 1;
        }

        return 0;
    }
};


