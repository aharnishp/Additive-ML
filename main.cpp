// g++ main.cpp -o main -Lmodules/

#include<iostream>
#include<fstream>
#include<sstream>

#include "modules/nlayer.hpp"
#include "modules/nnetwork.hpp"

// temporary testing only
#define fori(i,n) for(int i = 0; i < n; i++)
#define pb push_back

void print1D(std::vector<def_float_t> vec){
    fori(i, vec.size()){
        std::cout << vec[i] << " ";
    }std::cout << std::endl;
}

int main(){

    nnetwork mnist1(784, 10, 0.001);
    mnist1.output_layer->activationFn=Sigmoid;
    // mnist1.add_new_layer_at_last(10, ReLU, 0.01);

    std::vector<def_float_t> input_values(784, 0.1);

    // read the input values from file
    std::ifstream input_file("dataset/mnist-train.csv");
    std::string line;
    std::vector<std::string> input_lines;

    // check if file has opened
    if(!input_file.is_open()){
        cout << "error: couldn't open the input_file." << endl;
        return -1;
    }
    


    // read every 8 lines and make a batch of it
    def_uint_t train_batch_size = 8;
    def_uint_t train_line_count = 0;

    std::vector<def_float_t> training_batch;
    std::vector<def_float_t> labels;

    // read headers once
    std::getline(input_file, line);
    training_batch.reserve(784);

    int batch_num = 0;


    while(std::getline(input_file, line)){
        // parse the line
        std::stringstream ss(line);
        std::string token;
        
        // first element is the label
        std::getline(ss, token, ',');
        // onehot encode the labels
        std::vector<def_float_t> oneh(10,0);
        int lab_num = stoi(token);
        if(lab_num < 0 || lab_num > 9){
            cout << "error: label number is not in range 0-9" << endl;
            return -1;
        }
        oneh[stoi(token)] = 1;
        
        // insert the onehot vector to labels
        labels.insert(labels.end(), oneh.begin(),oneh.end());

        // labels.pb(std::stof(token));

        def_int_t token_num = 0;

        
        // count the number of li
        // rest of the elements are the input values
        // std::vector<def_float_t> input_values;
        while(std::getline(ss, token, ',')){
            // std::cout << (token_num++) << ",\t" << train_line_count << std::endl;
            training_batch.pb(std::stof(token));
        }

        std::cout << lab_num << "\t" << batch_num++ << std::endl;
        training_batch.reserve(training_batch.size()+784);
        // training_batch.resize(training_batch.size()+784);

        train_line_count++;

        if(train_line_count == train_batch_size){
            cout << "training_batch.size() = " << training_batch.size() << endl;
            // train the network
            mnist1.backward_prop(training_batch, labels, train_line_count);
            // reset the training batch
            training_batch.clear();
            labels.clear();

            

            train_line_count = 0;
        }
    }

    
    // actual value is 4
    input_values = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,220,179,6,0,0,0,0,0,0,0,0,9,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,247,17,0,0,0,0,0,0,0,0,27,202,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,242,155,0,0,0,0,0,0,0,0,27,254,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,160,207,6,0,0,0,0,0,0,0,27,254,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127,254,21,0,0,0,0,0,0,0,20,239,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,254,21,0,0,0,0,0,0,0,0,195,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,70,254,21,0,0,0,0,0,0,0,0,195,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,56,251,21,0,0,0,0,0,0,0,0,195,227,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,153,5,0,0,0,0,0,0,0,120,240,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,251,40,0,0,0,0,0,0,0,94,255,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,184,0,0,0,0,0,0,0,19,245,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,169,0,0,0,0,0,0,0,3,199,182,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,154,205,4,0,0,26,72,128,203,208,254,254,131,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,254,129,113,186,245,251,189,75,56,136,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,216,233,233,159,104,52,0,0,0,38,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,206,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,186,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,209,101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    std::vector<def_float_t> predictions = mnist1.forward_prop(input_values, 1);
    print1D(predictions);
    
    return 0;
}


// int main(){

//     std::cout << "Main Executed!" << std::endl;

//     std::nlayer layer1;
//     std::nlayer layer2(4, ReLU, 0.01);    
//     std::nlayer layer3(3, ReLU, 0.01);

//     // std::nlayer layer3;
//     // std::nlayer cnvla4(3,2,1,Linear_activation_state, 0.06);

//     layer1.id=1;
//     layer2.id=2;


//     // set up as input layer
//     layer1.is_input_layer=1;
//     std::vector<float> input_values = {0.1,0.2,1,1,0.2,0.1};

//     layer1.cached_activation_values = input_values;
//     layer1.cached_batch_size = 2;

//     // layer3.id=3;

//     layer2.add_input_layer(&layer3);
//     layer3.add_input_layer(&layer1);
//     // layer1.add_input_layer(&layer3);
//     // layer1.add_input_layer(&cnvla4);


//     // for(int i = 0; i < 2; i++){
//     //     std::cout << "Printing input layers of layer 1:" << std::endl;
//     //     std::cout << ((std::nlayer *) (layer1.input_layers[i]))->id << std::endl;
//     // }


    
//     std::cout << "layer1 size = " << layer2.input_layers.size() << std::endl;

//     layer2.weight_inp = 3;
//     layer2.weight_out = 4;

//     layer3.weight_inp =3;
//     layer3.weight_out =3;


//     layer2.init_weight(1);
//     layer3.init_weight(1);
    
//     layer2.weights = {
//         0.1, 0.2, 0.3, 0.4,
//         0.5, 0.6, 0.7, 0.8,
//         0.9, 0.10, 0.11, 0.12
//     };

//     layer2.bias = {0.1,0.2,0.3,0.4};

//     // // print all flattened weight values:
//     // std::cout << "Printing Flattened Weights" << std::endl;
//     // fori(i, layer2.weights.size()){
//     //     std::cout << layer2.weights[i] << std::endl;
//     // }

//     // print all weight values:
//     std::cout << "Printing Weights" << std::endl;
//     fori(i, layer2.weight_inp){
//         fori(j,layer2.weight_out){
//             std::cout << layer2.get_weight_value(i,j) << " ";
//         }
//         std::cout << std::endl;
//     }

//     std::cout << "Compiling version 7" << std::endl;

//     std::vector<float> output;
//     std::cout << "layer2.weights.size() = " << layer2.weights.size() << std::endl;
//     std::cout << "&layer2=\t" << &layer2 << std::endl;

//     output = layer2.get_activation_rec(1,2);


//     std::cout << std::endl << std::endl << "######### Back Proping ##########" << std::endl;

//     std::vector<def_float_t> expected_vec = {0,2,0,4,5,0,3,0};
//     fori(iter,1000){

//         output = layer2.get_activation_rec(iter+1,2);

//         // calculate the new error which is the difference of expected and predicted value
//         std::vector<def_float_t> error_vec;
//         fori(i,expected_vec.size()){
//             error_vec.push_back((output[i] - expected_vec[i]));
//         }

//         // print error_vec
//         std::cout << "Error vec = ";
//         fori(i, error_vec.size()){
//             std::cout << error_vec[i] << " ";
//         }std::cout << std::endl;

//         layer2.get_correct_error_rec(iter+1,2,error_vec, 0.0001);
//     }

//     // layer2.get_correct_error_rec(1,1,error_vec, 0.05);
//     // fori(i,10){
//     //     layer2.get_correct_error_rec(i+2,1,error_vec, 0.05);
//     // }

//     output = layer2.get_activation_rec(1005,2);

//     // // print all weight values:
//     // std::cout << "Printing Weights after 1 correction" << std::endl;
//     // fori(i, layer2.weight_inp){
//     //     fori(j,layer2.weight_out){
//     //         std::cout << layer2.get_weight_value(i,j) << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }



//     // output = layer2.get_activation_rec(2,1);
//     // fori(i,output.size()){
//     //     std::cout << output[i] << " ";
//     // }std::cout << std::endl;

//     // def_int_t run_id = 3;
//     // fori(i, 99){
//     //     layer2.get_correct_error_rec(run_id++,1,expected_vec, 0.05);
//     // }
//     // // layer2.get_correct_error_rec(run_id++,1,expected_vec, 0.05);


//     // output = layer2.get_activation_rec(run_id++,1);












//     // // print all weight values:
//     // std::cout << "Printing Weights after 100 correction" << std::endl;
//     // fori(i, layer2.weight_inp){
//     //     fori(j,layer2.weight_out){
//     //         std::cout << layer2.get_weight_value(i,j) << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }



//     // // print the outputs
//     // fori(i, output.size()){
//     //     std::cout << output[i] << std::endl;
//     // }
    
//     // std::cout <<  << std::endl;
//     // std::cout << "initing weights = " << layer1.init_weight() << std::endl;
//     // // std::cout << "random float: " << get_rand_float() << std::endl;
//     // std::cout << "Printing BEFORE" << std::endl;
//     // // print weight matrix
//     // fori(i, layer1.weight_inp){
//     //     fori(j,layer1.weight_out){
//     //         std::cout << layer1.get_weight_value(i,j) << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }
//     // // layer1.reset_weights();
//     // // std::cout << "Printing AFTER" << std::endl;
//     // // // print weight matrix
//     // // fori(i, layer1.weight_inp){
//     // //     fori(j,layer1.weight_out){
//     // //         std::cout << layer1.get_weight(i,j) << " ";
//     // //     }
//     // //     std::cout << std::endl;
//     // // }

//     return 0;
// }
