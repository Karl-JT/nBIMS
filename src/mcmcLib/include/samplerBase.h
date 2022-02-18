#pragma once

class samplerBase{
public:
	samplerBase(){};
	virtual~samplerBase(){};

	virtual void sampleProposal(){};
	virtual void getAlpha(){};
};