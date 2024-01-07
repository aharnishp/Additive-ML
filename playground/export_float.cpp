#include<iostream>
#include<fstream>
#include<vector>

int main (){
    std::ofstream out("floats.dat", std::ios::binary);
    std::vector<float> floats = {1.0, 2.0, 3.0, 4.0, 5.0};
    // out.write(reinterpret_cast<char*>(floats.data()), floats.size()*sizeof(float));

    std::vector<char> outputs;

    for (auto f : floats){
        outputs.insert(outputs.end(), reinterpret_cast<char*>(&f), reinterpret_cast<char*>(&f) + sizeof(float));
        outputs.push_back(',');
    }


    // for(int i = 0; i < floats.size(); i++){
    //     for(int b = 0; b < sizeof(float); b++){
    //         char new_char;
    //         std::memcpy(&new_char, floats.data() + i*sizeof(float) + b, 1);
    //     }
    //     outputs.push_back(',');
    // }

    out.write(outputs.data(), outputs.size());
    out.close();


    return 0;
}
