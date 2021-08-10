#pragma once

class sampler{
public:
	sampler(){};
	~sampler(){};

	virtual void sampleProposal();
	virtual void getAlpha();
};