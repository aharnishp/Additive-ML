#ifndef NNETWORK_HPP    // FIXME: Check all import states
#define NNETWORK_HPP
#include "nnetwork.hpp"
#endif
#include "nlayer.hpp"

#include <vector>
#include <algorithm>

// #define def_float_t double


using namespace std;
class nnetwork{
    private:
    def_int_t run_id = 0;

    public:
    nlayer *input_layer;
    nlayer *output_layer;
    
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
        new_layer->init_weight(1);
        this->output_layer = new_layer;
        return new_layer->id;
    }
    
    /**
     * @brief given the input_values, returns the predicted value of the network.
     * @param input_values flattened 1D vector of the 2D array formed by input_layer.size * batch_size
     * @param batch_size
    */
    std::vector<def_float_t> forward_prop(std::vector<def_float_t> input_values, def_uint_t batch_size) {
        /*
        returns the predicted value of the network for the given input value.
        (returned value is flattened 1D array of output_layer.size * batch_size
        */
        run_id++;

        this->input_layer->cached_activation_values = input_values;
        this->input_layer->cached_batch_size = batch_size;

        return (this->output_layer->get_activation_rec(run_id,batch_size));

        // return this->output_layer->cached_activation_values;        
    }

    /**
     * @brief calculates output layer error and recursively corrects the error in the previous layers till the input layer.
     * @param input_values flattened 1D vector of the 2D array formed by input_layer.size * batch_size
     * @param expected_values flattened 1D vector of the 2D array formed by output_layer.size * batch_size
     * @param batch_size
    */
    std::vector<def_float_t> backward_prop(std::vector<def_float_t> input_values, std::vector<def_float_t> expected_values, def_uint_t batch_size) {

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
        std::transform(this->output_layer->cached_activation_values.begin(), this->output_layer->cached_activation_values.end(), expected_values.begin(), std::back_inserter(error_in_prediction), std::minus<def_float_t>());

        // cout << "this" << this << endl;
        // cout << "this->output_layer->id=" << this->output_layer->id << endl;

        // cout << "run_id=" << run_id << endl;
        // cout << "batch_size=" << batch_size << endl;
        // cout << "error_in_prediction.size()=" << error_in_prediction.size() << endl;
        
        if(this->output_layer != NULL){
            std::vector<def_float_t> error_in_input = this->output_layer->get_correct_error_rec(run_id, batch_size, error_in_prediction, this->default_learning_rate);
        }else{
            cout << "output_layer pointer is NULL" << endl;
        }

        return error_in_prediction;
    }
};


