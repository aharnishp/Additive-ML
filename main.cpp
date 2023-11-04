// g++ main.cpp -o main -Lmodules/

#include<iostream>

#include "modules/nlayer.cpp"

// temporary testing only
#define fori(i,n) for(int i = 0; i < n; i++)
#define pb push_back

int main(){

    std::cout << "Main Executed!" << std::endl;

    std::nlayer layer1;
    std::nlayer layer2;
    // std::nlayer layer3;
    // std::nlayer cnvla4(3,2,1,Linear_activation_state, 0.06);

    layer1.id=1;
    layer2.id=2;


    // set up as input layer
    layer1.is_input_layer=1;
    std::vector<float> input_values = {1,2,3};

    layer1.cached_acivation_values = input_values;
    layer1.cached_batch_size = 1;

    // layer3.id=3;

    layer2.add_input_layer(&layer1);
    // layer1.add_input_layer(&layer3);
    // layer1.add_input_layer(&cnvla4);


    // for(int i = 0; i < 2; i++){
    //     std::cout << "Printing input layers of layer 1:" << std::endl;
    //     std::cout << ((std::nlayer *) (layer1.input_layers[i]))->id << std::endl;
    // }


    
    std::cout << "layer1 size = " << layer2.input_layers.size() << std::endl;

    layer2.weight_inp = 3;
    layer2.weight_out = 4;

    layer2.init_weight(1);
    
    layer2.weights = {
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 0.10, 0.11, 0.12
    };

    // print all flattened weight values:
    std::cout << "Printing Flattened Weights" << std::endl;
    fori(i, layer2.weights.size()){
        std::cout << layer2.weights[i] << std::endl;
    }

    // print all weight values:
    std::cout << "Printing Weights" << std::endl;
    fori(i, layer2.weight_inp){
        fori(j,layer2.weight_out){
            std::cout << layer2.get_weight_value(i,j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Compiling version 6" << std::endl;

    std::vector<float> output;
    std::cout << "layer2.weights.size() = " << layer2.weights.size() << std::endl;
    std::cout << "&layer2=\t" << &layer2 << std::endl;
    output = layer2.get_activation_rec(1,1);

    // // print the outputs
    // fori(i, output.size()){
    //     std::cout << output[i] << std::endl;
    // }
    
    // std::cout <<  << std::endl;
    // std::cout << "initing weights = " << layer1.init_weight() << std::endl;
    // // std::cout << "random float: " << get_rand_float() << std::endl;
    // std::cout << "Printing BEFORE" << std::endl;
    // // print weight matrix
    // fori(i, layer1.weight_inp){
    //     fori(j,layer1.weight_out){
    //         std::cout << layer1.get_weight_value(i,j) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // // layer1.reset_weights();
    // // std::cout << "Printing AFTER" << std::endl;
    // // // print weight matrix
    // // fori(i, layer1.weight_inp){
    // //     fori(j,layer1.weight_out){
    // //         std::cout << layer1.get_weight(i,j) << " ";
    // //     }
    // //     std::cout << std::endl;
    // // }

    return 0;
}
