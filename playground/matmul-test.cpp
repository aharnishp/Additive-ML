#include <vector>
#include <iostream>

using namespace std;

#define def_float_t float
#define def_int_t int
#define def_uint_t uint
#define def_uint_small_t char

#define weight_inp_allocated 6
#define weight_out_allocated 4

def_uint_t flat_indx(int i, int j){
    return ( weight_inp_allocated * j + i );
} 




int main (){
    int weight_inp = 4;
    int weight_out = 3;

    vector<def_float_t> weights;

    weights = {
        1,0,0,0,0,0,
        0,1,0,0,0,0,
        0,0,1,3,0,0,
        0,0,0,0,0,0
    };


    int batch_size = 1;

    std::vector<def_float_t> activation_error;
    activation_error = {
        -1,-2,-2
    };

    std::vector<def_float_t> old_weights = weights;
    std::vector<def_float_t> input_error;

    for(int i = 0; i < weight_inp; i++){  
        for(int j = 0; j < weight_out; j++){
            def_float_t sum = 0;
            for(int k = 0; k < batch_size; k++){
                sum += activation_error[k*weight_out + j] * old_weights[flat_indx(i,j)];
            }
            input_error.push_back(sum);
        }
    }

    // expected dimensions of errors of input layers = weight_inp * batch_size

    for(int i = 0; i < input_error.size(); i++){
        std::cout << input_error[i] << " ";
    }
    std::cout << std::endl;

   // takes care of reserved weights.
    std::cout << "Weight weights.size()=" << weights.size() << " = (" << weight_inp <<  "x" << weight_inp<< "), allocated=(" <<  weight_inp_allocated << "," << weight_out_allocated << ")" << std::endl;
    for(int i = 0; i < weight_out; i++){
        for(int j = 0; j < weight_out; j++){
            std::cout << weights[flat_indx(i,j)] << " ";
        }
        std::cout << std::endl;
    }
   
    

    return 0;
}