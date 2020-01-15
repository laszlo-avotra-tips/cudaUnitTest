#pragma once

#include <vector>

bool addWithCuda(int* c, const int* a, const int* b, unsigned int size);
bool resetCuda();
bool addTwoVectors(int* c, const int* a, const int* b, unsigned int size);
bool cudaRescale(unsigned short* data, unsigned int size, 
	float* wholeSamples = 0,
	float* fractionalSamples = 0,
	char* errorMsg = 0,
	unsigned int linesPerFrame = 592, 
	unsigned int recordLength = 4864, 
	unsigned int rescalingDataLength = 8192);