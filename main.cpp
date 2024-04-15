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
#define learning_rate_def 5.0/2048.0
// 0.015625

#define epoch_count 1

#define train_batch_size_def 1
#define test_batch_size_def 1

int main(){
    nnetwork net_base(2,2,learning_rate_def);
    net_base.output_layer->activationFn=LReLU;
    net_base.output_layer->fix_weights();
    net_base.add_layer_between_output(2, LReLU, 0.01);

    net_base.export_nnetwork_to_file("identity-main.ann");



    nnetwork net(2,2,learning_rate_def);

    // import network from file
    net.import_nnetwork_from_file("identity-main.ann");

    net.print_architecture();

    return 0;
}