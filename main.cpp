// g++ main.cpp -o main -Lmodules/

#include<iostream>
#include<fstream>
#include<sstream>
#include<set>

#include "modules/nlayer.hpp"
#include "modules/nnetwork.hpp"

// temporary testing only
#define fori(i,n) for(int i = 0; i < n; i++)
#define pb push_back

#define train_data_sample_limit 43000
#define learning_rate_def 1.0/4.0
// 0.015625

#define epoch_count 1

#define train_batch_size_def 1
#define test_batch_size_def 1

int main(){
    nnetwork net(2,2,learning_rate_def);

    net.output_layer->activationFn=LReLU;
    // net.output_layer->init_weight(1,1);
    net.output_layer->fix_weights();

    net.output_layer->print_weights();

    std::cout << "net.output_layer->weights.size() = " << net.output_layer->weights.size() << std::endl;

    net.output_layer->is_dynamic_layer = 0; // should be 1
    // net.add_new_layer_at_last(2,LReLU,learning_rate_def);
    // net.output_layer->is_dynamic_layer = 0;

    // nlayer newl(3,ReLU,0.05);
    // newl.id=4;
    // newl.add_input_layer(net.input_layer);
    // net.output_layer->input_layers.clear();
    // net.output_layer->add_input_layer(&newl);
    // newl.auto_grow_weight();

    net.print_architecture();

    // net.output_layer->auto_grow_weight();
    
    // net.output_layer->auto_grow_weight(2,2)


    
    std::cout << "net.output_layer->weights.size() = " << net.output_layer->weights.size() << std::endl;

    // for(int i = 0; i< 100; i++){
    //     net.add_new_layer_at_last(2,ReLU,learning_rate_def);
    // }

    vector<def_float_t> input = {1,5,5,1};
    vector<def_float_t> output = {0,1,1,0};
    vector<def_float_t> expected_output = {1,5,5,1};

    // output = net.forward_prop(input,2);

    // // give solution and check for errors
    // std::vector<def_float_t> new_weight = {1.0,0,0,0,1.0,0};
    // std::vector<def_float_t> new_bias = {0,0};
    // net.output_layer->weights = new_weight;
    // net.output_layer->bias = new_bias;
    // net.output_layer->input_layers[0]->weights = new_weight;
    // net.output_layer->input_layers[0]->bias = new_bias;

    // give close solution and check divergence or convergence
    // std::vector<def_float_t> new_weight = {1.0,0,0,0,1.0,0};
    std::vector<def_float_t> new_weight = {0.99,0,0,0,0.99,0};
    std::vector<def_float_t> new_bias = {0,0};
    net.output_layer->weights = new_weight;
    net.output_layer->bias = new_bias;
    // net.output_layer->input_layers[0]->weights = new_weight;
    // net.output_layer->input_layers[0]->bias = new_bias;

    std::cout << "run_id =" << net.get_run_id() << std::endl;
    std::cout << "output before learning =" << std::endl;
    fori(i, output.size()){
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;


    for(int i = 0; i < 4; i++){
        std::cout << "run_id = " << net.get_run_id() << std::endl;
        net.output_layer->print_weights();
        net.backward_prop(input,expected_output, 2);

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "#### Printing output layer weights" << std::endl;
        net.output_layer->print_weights();
        std::cout << std::endl;
        std::cout << std::endl;
    }

    output = net.forward_prop(input,2);
    std::cout << "run_id =" << net.get_run_id() << std::endl;

    std::cout << "output after learning =" << std::endl;
    fori(i, output.size()){
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // std::cout << "Network created" << std::endl;


}