#pragma once
#include <cmath>

static void IC(double x, double y, double time, double output[], double samples[], int sampleSize){
    output[0] = 0; //-samples[1]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y);
    output[1] = 0; //samples[1]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y);
};