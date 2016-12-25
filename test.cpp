#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include "practice.h"

using namespace std;

double *buffer, *A, *B, *C, *C1;
int N=128;

// Alignment must be power of 2 (1,2,4,8,16...)
void* aligned_malloc(size_t size, size_t alignment) {
	uintptr_t r=(uintptr_t)malloc(size + --alignment + sizeof(uintptr_t));
	uintptr_t t=r + sizeof(uintptr_t);
	uintptr_t o=(t + alignment) & ~(uintptr_t)alignment;
	if(!r) return NULL;
	((uintptr_t*)o)[-1]=r;
	return (void*)o;
}

void aligned_free(void* p) {
	if(!p) return;
	free((void*)(((uintptr_t*)p)[-1]));
}

/**
* @return true if implementation is correct
*/
bool correct() {
	double eps=1e-5;
	for(int i=0; i<N; i++) {
		if(abs(C[i]-C1[i])>eps) {
			cout<<i<<" "<<" "<<C[i]<<" "<<C1[i]<<endl;
			return false;
		}
	}
	return true;
}

void test() {
	if(correct()) {
		cout<<"Correct!"<<endl;
	} else{
		cout<<"Wrong!"<<endl;
	}
}

int main() {
	buffer=(double*)aligned_malloc(4 * N * N * sizeof(double), 16);
	for(int i=0; i<4*N*N; i++) {
		buffer[i]=0;
	}
	A=buffer + 0;
	B=A + N*N;
	C=B + N*N;
	C1=C + N*N;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dist(-100, 100);
	for(int i=0; i <2*N*N; i++) {
		buffer[i]=dist(gen);
	}

	double a=naive(N, A, B, C);
	double b=transpose_optimized(N, A, B, C1);

	test();
	cout<<a/b<<endl;

	getchar();

}
