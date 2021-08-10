#ifndef dnsDataInterpreter_H
#define dnsDataInterpreter_H

#include <vector>

using namespace std;

class dnsDataInterpreter{
public:
	vector<double> y;
	vector<double> U;

	dnsDataInterpreter(vector<double> yCoordinate, double Re_tau);
	vector<double> dnsDataInterpret();
};

#include "dnsDataInterpreter.C"
#endif
