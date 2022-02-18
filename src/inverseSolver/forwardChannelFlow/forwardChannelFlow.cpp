#include <fstream>
#include <sstream>
#include <random>

#include <dataIO.h>
#include <turbSolver.h>


int main(int argc, char **argv){
	double ref[10] = {0., 0., 0., 0., -0., 0., 0., 0., -0., -0.};

	forwardSolver channelFlow(4);
	channelFlow.updateParameter(ref, 10);
	channelFlow.soluFwd();

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 0.001);

	string filename = "./refObs.txt";

	double velocity[channelFlow.m];
	for (int i = 0; i < channelFlow.m; i++){
		velocity[i] = channelFlow.xVelocity[i].value() + distribution(generator);
	}

	write2txt(channelFlow.yCoordinate.get(), velocity, channelFlow.m, filename);

	double x1[channelFlow.m];
	double x2[channelFlow.m];

	txt2read(x1, x2, channelFlow.m, filename);
	for (int i = 0; i < channelFlow.m; ++i){
		std::cout << x1[i] << " ";
	}
	std::cout << std::endl;

	for (int i = 0; i < channelFlow.m; ++i){
		std::cout << x2[i] << " ";
	}

	return 0;
}
