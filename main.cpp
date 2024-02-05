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
#define learning_rate_def 0.015625*8

#define epoch_count 1

#define train_batch_size_def 1
#define test_batch_size_def 1

int main(){
    nnetwork net(2,2);

    net.output_layer->activationFn=ReLU;
    // net.output_layer->init_weight(1,1);
    net.output_layer->fix_weights();

    net.output_layer->print_weights();

    std::cout << "net.output_layer->weights.size() = " << net.output_layer->weights.size() << std::endl;

    net.add_new_layer_at_last(3,ReLU,learning_rate_def);

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

    vector<def_float_t> input = {1,2};
    vector<def_float_t> output = {1,2};

    output = net.forward_prop(input,1);

    std::cout << "output before learning =" << output[0] << "," << output[1] << std::endl;

    for(int i = 0; i < 10; i++){
        net.backward_prop(input,output, 1);
    }

    output = net.forward_prop(input,1);

    std::cout << "output after learning =" << output[0] << "," << output[1] << std::endl;

    // std::cout << "Network created" << std::endl;


}