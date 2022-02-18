#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <adolc/adolc.h>
#include <adolc/taping.h>

int main(){
	double* xp = new double[1];
	double yp = 0.0;
	adouble* x = new adouble[1];
	adouble y = 1;
	xp[0] = 0.5;
	
	trace_on(1);
	x[0] <<= xp[0];
	y = x[0] + 5;
	y = x[0]*y;
	trace_off;

	double* g = new double[1];
	gradient(1, 1, xp, g);

	cout << g[0];

	return 0;
}