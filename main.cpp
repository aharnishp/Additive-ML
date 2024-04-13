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
    nnetwork net(2,2,learning_rate_def);

    // import network from file
    net.import_nnetwork_from_file("identity-main.ann");
    

    return 0;
}