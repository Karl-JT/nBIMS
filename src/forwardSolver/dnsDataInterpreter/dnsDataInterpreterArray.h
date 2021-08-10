#pragma once

using namespace std;

class dnsDataInterpreter{
public:
	double* yDirection;
	double* uDNS;
	double* U;

	dnsDataInterpreter(double yCoordinate[], double Re_tau, int m);
	void dnsDataInterpret();
};
