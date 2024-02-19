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


#define train_batch_size_def 16
#define test_batch_size_def 1

// FIXME: this is for testing only
#define train_lines_limit 42000 // 11000

#define epochs_count 1
#define TELEMETRY_PROGRESS 1
#define TELEMETRY_TRAIN_DETAILS 0




#define learning_rate 0.015625
#define mnist_a_b_hidden_size 64

void print1D(std::vector<def_float_t> vec){
    fori(i, vec.size()){
        std::cout << vec[i] << " ";
    }std::cout << std::endl;
}

void print_architecture(nnetwork nn){
    std::cout << "printing nnetwork architecture" << std::endl;
    std::cout << "Output size" << nn.output_layer->size() << std::endl;
    // std::cout << 
    std::set<nlayer*> visited;
    std::vector<nlayer*> unvisited;
    unvisited.push_back(nn.output_layer);
    while(unvisited.size()){
        nlayer* this_layer = unvisited[unvisited.size()-1];
        unvisited.pop_back();   // remove from last
        std::cout << "# layer-id=" << this_layer->id << " \tlayer-size=" << this_layer->size() << " \tinput_layers:" << std::endl;
        std::cout << "    printing weights(" << this_layer->weight_inp << " x " << this_layer->weight_out << ")" << std::endl;


        for(int i = 0; i < this_layer->input_layers.size(); i++){
            std::cout << "    -layer"  << this_layer->input_layers[i]->id << std::endl;
            if(visited.find(this_layer->input_layers[i]) == visited.end()){
                unvisited.push_back(this_layer->input_layers[i]);
            }
        }
        visited.insert(this_layer);

        // std::cout << "layer-" << std::endl;
    }

}

/**
 * @brief onehot encode the given number in class if fitting in range.
*/
std::vector<def_float_t> onehot_encode_unknown(def_int_t num, def_int_t min, def_int_t max){
    std::vector<def_float_t> oneh(max-min+1+1, 0);
    if(num < min || num > max){
        oneh[oneh.size()-1] = 1;    // last element is for the unknown class
        return oneh;
    }
    oneh[num-min] = 1;
    return oneh;
}

std::vector<def_float_t> onehot_encode_ignorant(def_int_t num, def_int_t min, def_int_t max){
    std::vector<def_float_t> oneh(max-min+1, 0);
    if(num < min || num > max){
        return oneh;
    }
    oneh[num-min] = 1;
    return oneh;
}

// std::vector<def_float_t> parse_mnist_data_normalized(std::string line){
//     std::vector<def_float_t> input_values;
//     std::stringstream ss(line);
//     std::string token;
//     while(std::getline(ss, token, ',')){
//         input_values.pb(std::stof(token)/255.0);
//     }
//     return input_values;
// }



/**
 * @brief get the index of the max value out of the input vector
*/
def_int_t get_max_class(std::vector<def_float_t> vec){
    def_int_t max_index = 0;
    def_float_t max_value = vec[0];
    for(int i = 1; i < vec.size(); i++){
        if(vec[i] > max_value){
            max_value = vec[i];
            max_index = i;
        }
    }
    return max_index;
}


int main(){
    nnetwork mna(784,7,learning_rate);
    mna.add_layer_between_output(mnist_a_b_hidden_size,LReLU,learning_rate);
    mna.output_layer->is_dynamic_layer = 0;
    mna.output_layer->input_layers[0]->is_dynamic_layer = 0;

    nnetwork mnb(784,7,learning_rate);
    mnb.add_layer_between_output(mnist_a_b_hidden_size,LReLU,learning_rate);
    mnb.output_layer->is_dynamic_layer = 0;
    mnb.output_layer->input_layers[0]->is_dynamic_layer = 0;

    std::cout << "### ARCHITECTURE ###" << std::endl;
    print_architecture(mna);
    std::cout << "" << std::endl;

    // training the minst a and b in given epochs
    fori(epochs,epochs_count){
        // read the input values from file
        std::ifstream input_file("dataset/mnist-train.csv");
        std::string line;
        std::vector<std::string> input_lines;
        
        //check if the file opened
        if(!input_file.is_open()){
            std::cout << "error: couldn't open the input_file." << std::endl;
            return -1;
        }

        // declare train_batch_size
        def_uint_t train_batch_size = train_batch_size_def;
        def_uint_t train_line_count = 0;

        std::vector<def_float_t> training_batch;
        std::vector<def_float_t> labelsA;
        std::vector<def_float_t> labelsB;

        // read headers once
        std::getline(input_file, line);

        // reserve the memory for the training batch
        training_batch.reserve(784);

        int train_iter = 0;

        while(std::getline(input_file, line)){
            std::stringstream ss(line);
            std::string token;

            // first element is the label
            std::getline(ss, token, ',');
            // onehot encode the labels
            std::vector<def_float_t> onehA = onehot_encode_unknown(stoi(token), 0, 5);
            std::vector<def_float_t> onehB = onehot_encode_unknown(stoi(token), 4, 9);

            // insert the onehot vector to labels
            labelsA.insert(labelsA.end(), onehA.begin(),onehA.end());
            labelsB.insert(labelsB.end(), onehB.begin(),onehB.end());

            // rest of the elements are the input values
            // std::vector<def_float_t> input_values = parse_mnist_data_normalized(line);
            std::vector<def_float_t> input_values;
            while(std::getline(ss, token, ',')){
                input_values.pb(std::stof(token)/255.0);
            }

            training_batch.insert(training_batch.end(), input_values.begin(), input_values.end());
            
            train_iter++;
            if(TELEMETRY_PROGRESS){
                std::cout << "train_iter=" << train_iter << std::endl;
            }

            if(train_iter > train_lines_limit){
                break;
            }


            train_line_count++;

            if(train_line_count == train_batch_size){
                if(TELEMETRY_TRAIN_DETAILS){
                    std::cout << "training_batch.size() = " << training_batch.size() << std::endl;
                }
                // train the networks
                mna.backward_prop(training_batch, labelsA, train_line_count, 0);
                mnb.backward_prop(training_batch, labelsB, train_line_count, 0);

                // reset the training batchs
                training_batch.clear(); labelsA.clear(); labelsB.clear();

                train_line_count = 0;
            }
        }

        input_file.close();
    }


    // // testing the mnist a and b
    // std::string line;
    // std::ifstream test_file("dataset/mnist-test.csv");
    // std::vector<def_float_t> test_batch;
    // std::vector<def_float_t> test_labels;
    // std::getline(test_file, line);
    // test_batch.reserve(784);
    // def_uint_t test_batch_size = test_batch_size_def; // train_test_batch_size
    // def_uint_t test_line_count = 0;
    // def_int_t train_iter = 0;
    //
    // def_int_t total_correct = 0;
    //
    // while(std::getline(test_file, line)){
    //     // parse the line
    //     std::stringstream ss(line);
    //     std::string token;
    //
    //     // first element is the label
    //     std::getline(ss, token, ',');
    //     // onehot encode the labels
    //     std::vector<def_float_t> onehA = onehot_encode_unknown(stoi(token), 0, 5);
    //     std::vector<def_float_t> onehB = onehot_encode_unknown(stoi(token), 4, 9);
    //
    //     // insert the onehot vector to labels
    //     test_labels.insert(test_labels.end(), onehA.begin(),onehA.end());
    //     test_labels.insert(test_labels.end(), onehB.begin(),onehB.end());
    //
    //     // rest of the elements are the input values
    //     // std::vector<def_float_t> input_values = parse_mnist_data_normalized(line);
    //     std::vector<def_float_t> input_values;
    //     while(std::getline(ss, token, ',')){
    //         input_values.pb(std::stof(token)/255.0);
    //     }
    //
    //     test_batch.insert(test_batch.end(), input_values.begin(), input_values.end());
    //
    //     train_iter++;
    //     if(TELEMETRY_TRAIN_PROGRESS){
    //         std::cout << "train_iter=" << train_iter << std::endl;
    //     }
    
    // training new network with this as it's input
    nnetwork mnc(14, 10, learning_rate);
    // mnc.add_layer_between_output(14,LReLU,learning_rate);
    mnc.output_layer->weights = {
        1,0,0,0,0,0,0,0,0,0,
        0,1,0,0,0,0,0,0,0,0,
        0,0,1,0,0,0,0,0,0,0,
        0,0,0,1,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,1,0,0,0,
        0,0,0,0,0,0,0,1,0,0,
        0,0,0,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,0,0
    };
    fori(i,mnc.output_layer->weights.size()){
        mnc.output_layer->weights[i] += 0.05;
    }
    // mnc.output_layer->weights = {
    //     1,0,0,0,0,0,0,0,0,0,0,0,0,0,
    //     0,1,0,0,0,0,0,0,0,0,0,0,0,0,
    //     0,0,1,0,0,0,0,0,0,0,0,0,0,0,
    //     0,0,0,1,0,0,0,0,0,0,0,0,0,0,
    //     0,0,0,0,1,0,0,1,0,0,0,0,0,0,
    //     0,0,0,0,0,1,0,0,1,0,0,0,0,0,
    //     0,0,0,0,0,0,0,0,0,1,0,0,0,0,
    //     0,0,0,0,0,0,0,0,0,0,1,0,0,0,
    //     0,0,0,0,0,0,0,0,0,0,0,1,0,0,
    //     0,0,0,0,0,0,0,0,0,0,0,0,1,0
    // };
    mnc.output_layer->activationFn=Softmax;

    std::cout << "### ARCHITECTURE ###" << std::endl;
    print_architecture(mnc);
    std::cout << "" << std::endl;

    // training mnc with input from mna and mnb
    // FIXME: remove 0* to enable training
    std::cout << "Not training the combined model" << std::endl;
    fori(epochs,epochs_count){
        // read the input values from file
        std::ifstream input_file("dataset/mnist-train.csv");
        std::string line;
        std::vector<std::string> input_lines;
        
        //check if the file opened
        if(!input_file.is_open()){
            std::cout << "error: couldn't open the input_file." << std::endl;
            return -1;
        }

        // declare train_batch_size
        def_uint_t train_batch_size = train_batch_size_def;
        def_uint_t train_line_count = 0;

        std::vector<def_float_t> input_to_AB;
        std::vector<def_float_t> labelsA;
        std::vector<def_float_t> labelsB;
        std::vector<def_float_t> labelsC;

        std::vector<def_float_t> intermediate_batch;

        // read headers once
        std::getline(input_file, line);

        // reserve the memory for the training batch
        input_to_AB.reserve(784);
        intermediate_batch.reserve(12);

        int train_iter = 0;

        while(std::getline(input_file, line)){
            std::stringstream ss(line);
            std::string token;

            // first element is the label
            std::getline(ss, token, ',');
            // onehot encode the labels
            std::vector<def_float_t> onehA = onehot_encode_unknown(stoi(token), 0, 5);
            std::vector<def_float_t> onehB = onehot_encode_unknown(stoi(token), 4, 9);
            std::vector<def_float_t> onehC = onehot_encode_ignorant(stoi(token), 0, 9);


            // insert the onehot vector to labels
            labelsA.insert(labelsA.end(), onehA.begin(),onehA.end());
            labelsB.insert(labelsB.end(), onehB.begin(),onehB.end());
            labelsC.insert(labelsC.end(), onehC.begin(),onehC.end());

            // rest of the elements are the input values
            // std::vector<def_float_t> input_values = parse_mnist_data_normalized(line);
            std::vector<def_float_t> input_values;
            while(std::getline(ss, token, ',')){
                input_values.pb(std::stof(token)/255.0);
            }

            input_to_AB.insert(input_to_AB.end(), input_values.begin(), input_values.end());
            
            // this is slower, but is required to interleave the results for batch_size > 1
            std::vector<def_float_t> input_from_A = mna.forward_prop(input_to_AB, 1);
            std::vector<def_float_t> input_from_B = mnb.forward_prop(input_to_AB, 1);

            input_to_AB.clear();
            labelsA.clear();
            labelsB.clear();


            intermediate_batch.insert(intermediate_batch.end(), input_from_A.begin(), input_from_A.end());
            intermediate_batch.insert(intermediate_batch.end(), input_from_B.begin(), input_from_B.end());
            
            train_iter++;
            if(TELEMETRY_PROGRESS){
                std::cout << "train_iter=" << train_iter << std::endl;
            }

            if(train_iter > train_lines_limit){
                break;
            }

            train_line_count++;

            if(train_line_count == train_batch_size){
                if(TELEMETRY_TRAIN_DETAILS){
                    std::cout << "intermediate_batch.size() = " << intermediate_batch.size() << std::endl;
                }
                // train the networks
                mnc.backward_prop(intermediate_batch, labelsC, train_line_count);

                // reset the training batchs
                input_to_AB.clear(); labelsA.clear(); labelsB.clear(); labelsC.clear(); intermediate_batch.clear();

                train_line_count = 0;
            }
            
        }

        input_file.close();
    }



    // test the combined networks

    std::string line;
    std::ifstream test_file("dataset/mnist-test.csv");
    std::vector<def_float_t> test_batch;
    std::vector<def_float_t> test_labels;
    std::getline(test_file, line);
    test_batch.reserve(784);
    def_uint_t test_batch_size = test_batch_size_def; // train_test_batch_size
    def_uint_t test_line_count = 0;
    def_int_t test_iter = 0;

    def_int_t total_correct = 0;

    while(std::getline(test_file, line)){
        // parse the line
        std::stringstream ss(line);
        std::string token;

        // first element is the label
        std::getline(ss, token, ',');
        // onehot encode the labels
        // std::vector<def_float_t> onehA = onehot_encode_unknown(stoi(token), 0, 5);
        // std::vector<def_float_t> onehB = onehot_encode_unknown(stoi(token), 4, 9);
        std::vector<def_float_t> onehC = onehot_encode_ignorant(stoi(token), 0, 9);

        // insert the onehot vector to labels
        test_labels.insert(test_labels.end(), onehC.begin(),onehC.end());

        // rest of the elements are the input values
        // std::vector<def_float_t> input_values = parse_mnist_data_normalized(line);
        std::vector<def_float_t> input_AB;
        while(std::getline(ss, token, ',')){
            input_AB.pb(std::stof(token)/255.0);
        }
        
        std::vector<def_float_t> input_from_A = mna.forward_prop(input_AB, 1);
        std::vector<def_float_t> input_from_B = mnb.forward_prop(input_AB, 1);

        input_AB.clear();

        test_batch.insert(test_batch.end(), input_from_A.begin(), input_from_A.end());
        test_batch.insert(test_batch.end(), input_from_B.begin(), input_from_B.end());

        test_iter++;
        
        if(TELEMETRY_PROGRESS){
            std::cout << "test_iter=" << test_iter << std::endl;
        }

        test_line_count++;
        
        if(test_line_count == test_batch_size){
            if(TELEMETRY_TRAIN_DETAILS){
                std::cout << "test_batch.size() = " << test_batch.size() << std::endl;
            }
            // train the networks
            std::vector<def_float_t> predictions = mnc.forward_prop(test_batch, test_batch_size);
            // reset the training batchs
            test_batch.clear(); 
            
            // compare the results
                def_uint_t correct_count = 0;
                fori(i, test_batch_size){
                    def_int_t batch_start = i*10;
                    def_int_t batch_end = batch_start + 10;
                    def_int_t max_index = 0;
                    def_float_t max_value = predictions[batch_start];
                    for(int j = batch_start+1; j < batch_end; j++){
                        if(predictions[j] > max_value){
                            max_value = predictions[j];
                            max_index = j;
                        }
                    }
                    std::cout << "pred=" << max_index << " \t corr=" << test_labels[max_index] << std::endl;
                    if(test_labels[max_index] == 1){
                        correct_count++;
                    }
                }
                total_correct += correct_count;
            
            test_labels.clear();

            test_line_count = 0;
        }
        
    }
    std::cout << "Total Combined Accuracy = " << (total_correct*100.0)/test_iter << std::endl;

    std::cout << "mnc output weights:" << std::endl;
    print1D(mnc.output_layer->weights);
    // print2D
    fori(i, mnc.output_layer->weights.size()/10){
        fori(j, 10){
            std::cout << mnc.output_layer->weights[i*10+j] << " ";
        }std::cout << std::endl;
    }
    return 0;
}


int main1(){
    nnetwork mnist1(784, 10, 0.015625);
    mnist1.output_layer->activationFn=Softmax;
    mnist1.add_layer_between_output(64,LReLU,0.015625);
    // mnist1.add_layer_between_output(32,LReLU,0.015625*2);
    // mnist1.add_layer_between_output(16,LReLU,0.015625*2);

    // mnist1.add_layer_between_output(16,ReLU,0.015625/2);

    std::cout << "### ARCHITECTURE ###" << std::endl;
    print_architecture(mnist1);
    std::cout << "" << std::endl;


    std::cout << "output_layer->x = " << mnist1.output_layer->x << std::endl;
    std::cout << "hidden->x = " << mnist1.output_layer->input_layers[0]->x << std::endl;
    std::cout << "input_layer->x = " << mnist1.input_layer->x << std::endl;


    if(1){
        std::vector<def_float_t> input_values(784, 0.1);

        fori(epoch, 1){
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
            def_uint_t train_batch_size = train_batch_size_def;
            def_uint_t train_line_count = 0;

            std::vector<def_float_t> training_batch;
            std::vector<def_float_t> labels;

            // read headers once
            std::getline(input_file, line);
            training_batch.reserve(784);

            int train_iter = 0;


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

                // print1D(oneh);
                // std::cout << oneh << std::endl;
                
                // insert the onehot vector to labels
                labels.insert(labels.end(), oneh.begin(),oneh.end());

                // labels.pb(std::stof(token));

                def_int_t token_num = 0;

                
                // count the number of li
                // rest of the elements are the input values
                // std::vector<def_float_t> input_values;
                while(std::getline(ss, token, ',')){
                    // std::cout << (token_num++) << ",\t" << train_line_count << std::endl;
                    training_batch.pb(std::stof(token)/255.0);
                }

                std::cout << lab_num << "\t" << train_iter++ << std::endl;
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

            // close the file
            input_file.close();
        }

        
        // // actual value is 4
        // input_values = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,220,179,6,0,0,0,0,0,0,0,0,9,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,247,17,0,0,0,0,0,0,0,0,27,202,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,242,155,0,0,0,0,0,0,0,0,27,254,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,160,207,6,0,0,0,0,0,0,0,27,254,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127,254,21,0,0,0,0,0,0,0,20,239,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,254,21,0,0,0,0,0,0,0,0,195,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,70,254,21,0,0,0,0,0,0,0,0,195,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,56,251,21,0,0,0,0,0,0,0,0,195,227,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,153,5,0,0,0,0,0,0,0,120,240,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,251,40,0,0,0,0,0,0,0,94,255,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,184,0,0,0,0,0,0,0,19,245,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,169,0,0,0,0,0,0,0,3,199,182,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,154,205,4,0,0,26,72,128,203,208,254,254,131,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,254,129,113,186,245,251,189,75,56,136,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,216,233,233,159,104,52,0,0,0,38,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,206,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,186,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,209,101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        // for(int i = 0; i < input_values.size(); i++){
        //     input_values[i] /= 255.0;
        // }

        // std::cout << "RUNID=" << mnist1.get_run_id() << std::endl;
        
        // std::vector<def_float_t> predictions = mnist1.forward_prop(input_values, 1);
        // std::cout << "Prediction matrix for 4" << std::endl;
        // print1D(predictions);
        
        // std::cout << "RUNID=" << mnist1.get_run_id() << std::endl;

        // // actual value is 5
        // input_values = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,149,156,179,254,254,201,119,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,147,241,253,253,254,253,253,253,253,245,160,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,253,253,180,174,175,174,174,174,174,223,247,145,6,0,0,0,0,0,0,0,0,0,0,0,0,7,197,254,253,165,2,0,0,0,0,0,0,12,102,184,16,0,0,0,0,0,0,0,0,0,0,0,0,152,253,254,162,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,235,254,158,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,74,250,253,15,0,0,0,16,20,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,199,253,253,0,0,25,130,235,254,247,145,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,253,253,177,100,219,240,253,253,254,253,253,125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,193,253,253,254,253,253,200,155,155,238,253,229,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,249,254,241,150,30,0,0,0,215,254,254,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,39,30,0,0,0,0,0,214,253,234,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,241,253,183,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,201,253,253,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,114,254,253,154,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,254,255,241,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,118,235,253,249,103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,81,0,102,211,253,253,253,135,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,79,243,234,254,253,253,216,117,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,245,253,254,207,126,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        // for(int i = 0; i < input_values.size(); i++){
        //     input_values[i] /= 255.0;
        // }        
        
        // predictions = mnist1.forward_prop(input_values, 1);
        // std::cout << "Prediction matrix for 5" << std::endl;
        // print1D(predictions);

        // std::cout << "RUNID=" << mnist1.get_run_id() << std::endl;



        std::string line;

        // read from mnist-test.csv
        std::ifstream test_file("dataset/mnist-test.csv");
        std::vector<def_float_t> test_batch;
        std::vector<def_float_t> test_labels;
        std::getline(test_file, line);
        test_batch.reserve(784);
        def_uint_t test_batch_size = test_batch_size_def; // train_test_batch_size
        def_uint_t test_line_count = 0;
        def_int_t train_iter = 0;

        def_int_t total_correct = 0;

        // mnist1.output_layer->bias[1] = -0.1;


        while(std::getline(test_file, line)){
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

            // print1D(oneh);
            // std::cout << oneh << std::endl;
            
            // insert the onehot vector to labels
            test_labels.insert(test_labels.end(), oneh.begin(),oneh.end());

            // labels.pb(std::stof(token));

            def_int_t token_num = 0;

            
            // count the number of li
            // rest of the elements are the input values
            // std::vector<def_float_t> input_values;
            while(std::getline(ss, token, ',')){
                // std::cout << (token_num++) << ",\t" << train_line_count << std::endl;
                test_batch.pb(std::stof(token)/255.0);
            }

            std::cout << lab_num << "\t" << train_iter++ << std::endl;
            test_batch.reserve(test_batch.size()+784);
            // training_batch.resize(training_batch.size()+784);

            test_line_count++;

            // check accuracy after every 8 lines
            if(test_line_count == test_batch_size){
                std::vector<def_float_t> predictions = mnist1.forward_prop(test_batch, test_batch_size);
                test_batch.clear();

                // compare the results
                def_uint_t correct_count = 0;
                fori(i, test_batch_size){
                    def_int_t batch_start = i*10;
                    def_int_t batch_end = batch_start + 10;
                    def_int_t max_index = 0;
                    def_float_t max_value = predictions[batch_start];
                    for(int j = batch_start+1; j < batch_end; j++){
                        if(predictions[j] > max_value){
                            max_value = predictions[j];
                            max_index = j;
                        }
                    }
                    std::cout << "pred=" << max_index << " \t corr=" << test_labels[max_index] << std::endl;
                    if(test_labels[max_index] == 1){
                        correct_count++;
                    }
                }
                                
                total_correct += correct_count;
                std::cout << "Accuracy = " << (correct_count*100.0)/test_batch_size << std::endl;
                test_labels.clear();

                // if(train_iter > 100){
                //     break;
                // }
                test_line_count = 0;
                
            }
        }
        // print the total accuarcy on the test set
        std::cout << "Total Accuracy = " << (total_correct*100.0)/train_iter << std::endl;

        
    }

    // print_architecture(mnist1);
    // std::cout << "hidden layer weights" << std::endl;
    // print1D(mnist1.output_layer->input_layers[0]->weights);
    // std::cout << "hidden layer activations" << std::endl;
    // print1D(mnist1.output_layer->input_layers[0]->cached_activation_values);
    // std::cout << "printing output layer weights" << std::endl;
    // print1D(mnist1.output_layer->weights);
    // std::cout << "printing output layer biases" << std::endl;
    // // std::cout << "input cached" << std::endl;
    // // print1D(mnist1.output_layer->input_layers[0]->input_layers[0]->cached_activation_values);
    // print1D(mnist1.output_layer->bias);
    // std::vector<def_float_t> myvec = {0.1,0.2};
    // mnist1.output_layer->apply_activation_fn(myvec);

    
    return 0;
}