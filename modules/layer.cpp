#include<iostream>
#include<vector>

#define telemetry 1
#define fori(i,n) for(int i = 0; i < n; i++)
#define pb push_back

namespace std
{
    
    class nlayer{
    
        public:

        __UINT32_TYPE__ id = 0;

        // shape of this 3D layer
        __UINT16_TYPE__ x;
        __UINT16_TYPE__ y;
        __UINT16_TYPE__ z;

        // vector of pointers storing pointer to input layers
        vector<int*> input_layers(0, nullptr);

        // 3D vector storing weights
        vector<vector<vecctor<float>>> weights;

        // float storing bias
        float bias;

        



        


        



    }


} // namespace std


