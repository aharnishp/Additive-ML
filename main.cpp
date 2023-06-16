// g++ main.cpp -o main -Lmodules/

#include<iostream>

#include "modules/nlayer.cpp"


int main(){

    std::cout << "Main Executed!" << std::endl;

    std::nlayer layer1;
    std::nlayer layer2;
    std::nlayer layer3;

    layer1.id=1;
    layer2.id=2;
    layer3.id=3;

    layer1.add_input_layer(&layer2);
    layer1.add_input_layer(&layer3);


    for(int i = 0; i < 2; i++){
        std::cout << "Printing input layers of layer 1:" << std::endl;
        std::cout << ((std::nlayer *) (layer1.input_layers[i]))->id << std::endl;
    }

    layer1.weight_x = 3;
    layer1.weight_y = 4;

    std::cout << "initing weights = " << layer1.init_weight() << std::endl;

    layer1.set_weight(1,2,4.0f);

    // std::cout << "random float: " << get_rand_float() << std::endl;

    std::cout << "Printing BEFORE" << std::endl;
    // print weight matrix
    fori(i, layer1.weight_x){
        fori(j,layer1.weight_y){
            std::cout << layer1.get_weight(i,j) << " ";
        }
        std::cout << std::endl;
    }

    layer1.reset_weights();

    std::cout << "Printing AFTER" << std::endl;
    // print weight matrix
    fori(i, layer1.weight_x){
        fori(j,layer1.weight_y){
            std::cout << layer1.get_weight(i,j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}