#pragma once

#include <vector>

bool addWithCuda(int* c, const int* a, const int* b, unsigned int size);
bool resetCuda();
bool addTwoVectors(int* c, const int* a, const int* b, size_t size);
