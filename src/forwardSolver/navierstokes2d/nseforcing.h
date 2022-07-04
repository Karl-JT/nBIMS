#pragma once

//Periodic Boundary Case, KL expansion;
static void forcing(double x, double y, double time, double output[], double samples[], int sampleSize){
    output[0] = 100*samples[0]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y)*exp(time);//-samples[1]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)*exp(time)-samples[2]/2.*std::cos(4.*M_PI*x)*std::sin(4.*M_PI*y)*exp(time)-samples[3]/2.*std::sin(4.*M_PI*x)*std::cos(4.*M_PI*y)*exp(time);
    output[1] = -100*samples[0]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)*exp(time);//+samples[1]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y)*exp(time)+samples[2]/2.*std::sin(4.*M_PI*x)*std::cos(4.*M_PI*y)*exp(time)+samples[3]/2.*std::cos(4.*M_PI*x)*std::sin(4.*M_PI*y)*exp(time);
};

// static void forcing(double x, double y, double time, double output[], double samples[], int sampleSize){
// 	output[0] = -samples[0]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y)*exp(time) - 2.*samples[0]*pow(2.*M_PI, 2)*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y)*(exp(time)-1.0) - samples[0]*samples[0]*M_PI*std::sin(4.0*M_PI*x)*pow(exp(time)-1, 2.); //samples[0]*
//     output[1] = samples[0]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)*exp(time) + 2.*samples[0]*pow(2.*M_PI, 2)*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)*(exp(time)-1.0) - samples[0]*samples[0]*M_PI*std::sin(4.0*M_PI*y)*pow(exp(time)-1, 2.); //samples[0]*
// }

// static void forcing(double x, double y, double time, double output[], double samples[], int sampleSize){
// 	output[0] = samples[0]*pow(std::sin(M_PI*x),2)*std::sin(M_PI*y)*std::cos(M_PI*y)*exp(time)-2.*samples[0]*pow(M_PI, 2)*std::cos(2.*M_PI*x)*std::sin(M_PI*y)*std::cos(M_PI*y)*(exp(time)-1.0)+samples[0]*samples[0]*M_PI*pow(std::sin(M_PI*x), 3)*std::cos(M_PI*x)*pow(std::sin(M_PI*y), 2)*pow(exp(time)-1, 2.);
//     output[1] = -samples[0]*pow(std::sin(M_PI*y),2)*std::sin(M_PI*x)*std::cos(M_PI*x)*exp(time)+2.*samples[0]*pow(M_PI, 2)*std::cos(2.*M_PI*y)*std::sin(M_PI*x)*std::cos(M_PI*x)*(exp(time)-1.0)+samples[0]*samples[0]*M_PI*pow(std::sin(M_PI*y), 3)*std::cos(M_PI*y)*pow(std::sin(M_PI*x), 2)*pow(exp(time)-1, 2.);
// }


//Dirichlet Boundary Case, Polynomial Forcing
/*
static void forcing(double x, double y, double time, double output[], double samples[], int sampleSize){
	output[0] = samples[0]*exp(time)*1000*(pow(x,2)-4*pow(x,3)+6*pow(x,4)-4*pow(x,5)+pow(x,6))*(5*pow(y,4)-8*pow(y,3)+3*pow(y,2))+samples[0]*(exp(time)-1)*(-1000*(2-24*x+72*pow(x,2)-80*pow(x,3)+30*pow(x,4))*(5*pow(y,4)-8*pow(y,3)+3*pow(y,2))-1000*(pow(x,2)-4*pow(x,3)+6*pow(x,4)-4*pow(x,5)+pow(x,6))*(60*pow(y,2)-48*y+6)) \
    +samples[0]*samples[0]*pow(exp(time)-1,2)*(1000*(pow(x,2)-4*pow(x,3)+6*pow(x,4)-4*pow(x,5)+pow(x,6))*(5*pow(y,4)-8*pow(y,3)+3*pow(y,2))*1000*(2*x-12*pow(x,2)+24*pow(x,3)-20*pow(x,4)+6*pow(x,5))*(5*pow(y,4)-8*pow(y,3)+3*y*y)+(-1000)*(2*x-12*pow(x,2)+24*pow(x,3)-20*pow(x,4)+6*pow(x,5))*(pow(y,3)-2*pow(y,4)+pow(y,5))*(1000*(pow(x,2)-4*pow(x,3)+6*pow(x,4)-4*pow(x,5)+pow(x,6))*(20*pow(y,3)-24*pow(y,2)+6*y)));
    output[1] = samples[0]*exp(time)*(-1000)*(2*x-12*pow(x,2)+24*pow(x,3)-20*pow(x,4)+6*pow(x,5))*(pow(y,3)-2*pow(y,4)+pow(y,5))+samples[0]*(exp(time)-1)*(1000*(120*pow(x,3)-240*pow(x,2)+144*x-24)*(pow(y,3)-2*pow(y,4)+pow(y,5))+1000*(6*pow(x,5)-20*pow(x,4)+24*pow(x,3)-12*pow(x,2)+2*x)*(6*y-24*pow(y,2)+20*pow(y,3))) \
    +samples[0]*samples[0]*pow(exp(time)-1,2)*(1000*(pow(x,2)-4*pow(x,3)+6*pow(x,4)-4*pow(x,5)+pow(x,6))*(5*pow(y,4)-8*pow(y,3)+3*pow(y,2))*(-1000)*(2-24*x+72*pow(x,2)-80*pow(x,3)+30*pow(x,4))*(pow(y,3)-2*pow(y,4)+pow(y,5))+(-1000)*(2*x-12*pow(x,2)+24*pow(x,3)-20*pow(x,4)+6*pow(x,5))*(pow(y,3)-2*pow(y,4)+pow(y,5))*(-1000)*(2*x-12*pow(x,2)+24*pow(x,3)-20*pow(x,4)+6*pow(x,5))*(3*pow(y,2)-8*pow(y,3)+5*pow(y,4)));
}
*/

// static void forcing(double x, double y, double time, double output[], double samples[], int sampleSize){
// 	output[0] = 100*samples[0]*std::sin(2.*M_PI*x)*std::sin(2.*M_PI*y); 
// 	output[1] = 100*samples[0]*std::sin(2.*M_PI*x)*std::sin(2.*M_PI*y);     	
// }

// static void ref(double x, double y, double output[]){
//     output[0] = 0.8*std::sin(2.*M_PI*y)*(exp(1.0)-1.0);
// 	output[1] = 0.8*std::sin(2.*M_PI*x)*(exp(1.0)-1.0);
// }
