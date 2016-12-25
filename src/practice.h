#include <chrono>

using namespace std::chrono;

#define BLOCK_SIZE 32
#define min(a,b) (((a)<(b))?(a):(b))

/**
* @brief transpose transform matrix M with dimension n
*/
void transpose(int n, double* M) {
	double tmp;
	for(int i=0; i < n; i++) {
		for(int j=i + 1; j < n; j++) {
			tmp=M[i*n+j];
			M[i*n+j]=M[i+j*n];
			M[i+j*n]=tmp;
		}
	}
}


void unroll_SIMD_4(int n, double *A, double *B, double *C, int i, int j) {
	/* Compute C(i,j), C(i,j+1), C(i,j+2), C(i,j+3) */
	__m128d  cij=_mm_setzero_pd();
	__m128d  cij1=_mm_setzero_pd();
	__m128d  cij2=_mm_setzero_pd();
	__m128d  cij3=_mm_setzero_pd();

	double *ptrb0=B+j*n;
	double *ptrb1=B+(j+1)*n;
	double *ptrb2=B+(j+2)*n;
	double *ptrb3=B+(j+3)*n;

	for(int k=0; k < n; k+=2) {
		__m128d  a=_mm_load_pd(A+i*n+k);
		cij=_mm_add_pd(cij, _mm_mul_pd(a, _mm_load_pd(ptrb0)));
		ptrb0+=2;
		cij1=_mm_add_pd(cij1, _mm_mul_pd(a, _mm_load_pd(ptrb1)));
		ptrb1+=2;
		cij2=_mm_add_pd(cij2, _mm_mul_pd(a, _mm_load_pd(ptrb2)));
		ptrb2+=2;
		cij3=_mm_add_pd(cij3, _mm_mul_pd(a, _mm_load_pd(ptrb3)));
		ptrb3+=2;
	}
	cij=_mm_hadd_pd(cij, cij);
	_mm_store_sd(C+i*n+j, cij);
	cij1=_mm_hadd_pd(cij1, cij1);
	_mm_store_sd(C+i*n+j+1, cij1);
	cij2=_mm_hadd_pd(cij2, cij2);
	_mm_store_sd(C+i*n+j+2, cij2);
	cij3=_mm_hadd_pd(cij3, cij3);
	_mm_store_sd(C+i*n+j+3, cij3);
}


void unroll_SIMD_8(int n, double *A, double *B, double *C, int i, int j) {
	/* Compute C(i,j), C(i,j+1), C(i,j+2), C(i,j+3) */
	__m128d  cij=_mm_setzero_pd();
	__m128d  cij1=_mm_setzero_pd();
	__m128d  cij2=_mm_setzero_pd();
	__m128d  cij3=_mm_setzero_pd();
	__m128d  cij4=_mm_setzero_pd();
	__m128d  cij5=_mm_setzero_pd();
	__m128d  cij6=_mm_setzero_pd();
	__m128d  cij7=_mm_setzero_pd();

	double *ptrb0=B+j*n;
	double *ptrb1=B+(j+1)*n;
	double *ptrb2=B+(j+2)*n;
	double *ptrb3=B+(j+3)*n;
	double *ptrb4=B+(j+4)*n;
	double *ptrb5=B+(j+5)*n;
	double *ptrb6=B+(j+6)*n;
	double *ptrb7=B+(j+7)*n;

	for(int k=0; k < n; k+=2) {
		__m128d  a=_mm_load_pd(A+i*n+k);
		cij=_mm_add_pd(cij, _mm_mul_pd(a, _mm_load_pd(ptrb0)));
		ptrb0+=2;
		cij1=_mm_add_pd(cij1, _mm_mul_pd(a, _mm_load_pd(ptrb1)));
		ptrb1+=2;
		cij2=_mm_add_pd(cij2, _mm_mul_pd(a, _mm_load_pd(ptrb2)));
		ptrb2+=2;
		cij3=_mm_add_pd(cij3, _mm_mul_pd(a, _mm_load_pd(ptrb3)));
		ptrb3+=2;
		cij4=_mm_add_pd(cij4, _mm_mul_pd(a, _mm_load_pd(ptrb4)));
		ptrb4+=2;
		cij5=_mm_add_pd(cij5, _mm_mul_pd(a, _mm_load_pd(ptrb5)));
		ptrb5+=2;
		cij6=_mm_add_pd(cij6, _mm_mul_pd(a, _mm_load_pd(ptrb6)));
		ptrb6+=2;
		cij7=_mm_add_pd(cij7, _mm_mul_pd(a, _mm_load_pd(ptrb7)));
		ptrb7+=2;
	}
	cij=_mm_hadd_pd(cij, cij);
	_mm_store_sd(C+i*n+j, cij);
	cij1=_mm_hadd_pd(cij1, cij1);
	_mm_store_sd(C+i*n+j+1, cij1);
	cij2=_mm_hadd_pd(cij2, cij2);
	_mm_store_sd(C+i*n+j+2, cij2);
	cij3=_mm_hadd_pd(cij3, cij3);
	_mm_store_sd(C+i*n+j+3, cij3);
	cij4=_mm_hadd_pd(cij4, cij4);
	_mm_store_sd(C+i*n+j+4, cij4);
	cij5=_mm_hadd_pd(cij5, cij5);
	_mm_store_sd(C+i*n+j+5, cij5);
	cij6=_mm_hadd_pd(cij6, cij6);
	_mm_store_sd(C+i*n+j+6, cij6);
	cij7=_mm_hadd_pd(cij7, cij7);
	_mm_store_sd(C+i*n+j+7, cij7);
}


double transpose_optimized(int n, double* A, double* B, double* C){
	high_resolution_clock::time_point t1=high_resolution_clock::now();
	transpose(n, B);
	/* For each row i of A */
	for(int i=0; i < n; i++)
		/* For each column j of B */
		for(int j=0; j < n; j+=8)
		{
			unroll_SIMD_8(n, A, B, C, i, j);
		}
	transpose(n, B);
	high_resolution_clock::time_point t2=high_resolution_clock::now();
	auto duration=duration_cast<milliseconds>(t2 - t1).count();
	return duration;
}


/**
* @brief naive performs C = A * B with naive 3 nested loops, n is matrix dimension
* @return execution time in milli seconds
*/
double naive_transpose(int n, double* A, double* B, double* C){
	high_resolution_clock::time_point t1=high_resolution_clock::now();
	transpose(n, B);
	/* For each row i of A */
	for(int i=0; i < n; ++i)
		/* For each column j of B */
		for(int j=0; j < n; ++j)
		{
			/* Compute C(i,j) */
			__m128d  cij=_mm_setzero_pd();
			for(int k=0; k < n; k+=2)
				cij=_mm_add_pd(cij, _mm_mul_pd(_mm_load_pd(A+i*n+k), _mm_load_pd(B+j*n+k)));
			cij=_mm_hadd_pd(cij, cij);
			_mm_store_sd(C+i*n+j, cij);
		}
	transpose(n, B);
	high_resolution_clock::time_point t2=high_resolution_clock::now();
	auto duration=duration_cast<milliseconds>(t2 - t1).count();
	return duration;
}


/**
* @brief naive performs C = A * B with naive 3 nested loops, n is matrix dimension
* @return execution time in milli seconds
*/
double naive(int n, double* A, double* B, double* C){
	high_resolution_clock::time_point t1=high_resolution_clock::now();

	/* For each row i of A */
	for(int i=0; i < n; ++i)
		/* For each column j of B */
		for(int j=0; j < n; ++j)
		{
			/* Compute C(i,j) */
			C[i*n+j];
			for(int k=0; k < n; k++)
				C[i*n+j]+=A[i*n+k] * B[k*n+j];

		}
	high_resolution_clock::time_point t2=high_resolution_clock::now();
	auto duration=duration_cast<milliseconds>(t2 - t1).count();
	return duration;
}


/**
* @brief do_block perform block operation on a sub matrix
*/
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C)
{
	for(int i=0; i < M; ++i)
		for(int j=0; j < N; ++j)
		{
			/* Compute C(i,j) */
			double cij=C[i*lda+j];
			for(int k=0; k < K; ++k)
				cij+=A[i*lda+k] * B[k*lda+j];
			C[i*lda+j]=cij;
		}
}

/**
* @brief naive performs C = A * B with blocks, n is matrix dimension
* @return execution time in milli seconds
*/
double blocked(int lda, double* A, double* B, double* C)
{
	high_resolution_clock::time_point t1=high_resolution_clock::now();
	/* For each block-row of A */
	for(int i=0; i < lda; i+=BLOCK_SIZE)
		/* For each block-column of B */
		for(int j=0; j < lda; j+=BLOCK_SIZE)
			/* Accumulate block dgemms into block of C */
			for(int k=0; k < lda; k+=BLOCK_SIZE)
			{
				/* Correct block dimensions if block "goes off edge of" the matrix */
				int M=min(BLOCK_SIZE, lda-i);
				int N=min(BLOCK_SIZE, lda-j);
				int K=min(BLOCK_SIZE, lda-k);

				/* Perform individual block dgemm */
				do_block(lda, M, N, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
			}
	high_resolution_clock::time_point t2=high_resolution_clock::now();
	auto duration=duration_cast<milliseconds>(t2 - t1).count();
	return duration;
}


static void unroll_8_blocked(double* A, double* B, double* C, int lda, int i, int j, int K) {

	double *a, *c;
	c=&C[i+j*lda];
	/* Compute C(i,j) */
	double cij=*c++;
	/* Compute C(i+1,j) */
	double cij1=*c++;
	/* Compute C(i+2,j) */
	double cij2=*c++;
	/* Compute C(i+3,j) */
	double cij3=*c++;
	/* Compute C(i+4,j) */
	double cij4=*c++;
	/* Compute C(i+5,j) */
	double cij5=*c++;
	/* Compute C(i+6,j) */
	double cij6=*c++;
	/* Compute C(i+7,j) */
	double cij7=*c++;

	for(int k=0; k < K; k++) {
		double b=B[k+j*lda];
		a=&A[i+k*lda];
		cij+=(*a++)*b;
		cij1+=(*a++)*b;
		cij2+=(*a++)*b;
		cij3+=(*a++)*b;
		cij4+=(*a++)*b;
		cij5+=(*a++)*b;
		cij6+=(*a++)*b;
		cij7+=(*a++)*b;

	}
	c=&C[i+j*lda];
	*c++=cij;
	*c++=cij1;
	*c++=cij2;
	*c++=cij3;
	*c++=cij4;
	*c++=cij5;
	*c++=cij6;
	*c++=cij7;
}


static void do_block_optimized(int lda, int M, int N, int K, double* A, double* B, double* C)
{
	/* For each row i of A */
	for(int i=0; i < M; i+=8)
		/* For each column j of B */
		for(int j=0; j < N; ++j)
		{
			// step size for i is 8 hence unroll_8
			unroll_8_blocked(A, B, C, lda, i, j, K);
		}
}

/* This routine performs a dgemm operation
*  C := C + A * B
* where A, B, and C are lda-by-lda matrices stored in column-major format.
* On exit, A and B maintain their input values. */
double block_optimized(int lda, double* A, double* B, double* C)
{
	high_resolution_clock::time_point t1=high_resolution_clock::now();


	/* For each block-row of A */
	for(int i=0; i < lda; i+=BLOCK_SIZE)
		/* For each block-column of B */
		for(int j=0; j < lda; j+=BLOCK_SIZE)
			/* Accumulate block dgemms into block of C */
			for(int k=0; k < lda; k+=BLOCK_SIZE)
			{
				/* Correct block dimensions if block "goes off edge of" the matrix */
				int M=min(BLOCK_SIZE, lda-i);
				int N=min(BLOCK_SIZE, lda-j);
				int K=min(BLOCK_SIZE, lda-k);

				/* Perform individual block dgemm */
				do_block_optimized(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
			}
	high_resolution_clock::time_point t2=high_resolution_clock::now();
	auto duration=duration_cast<milliseconds>(t2 - t1).count();
	return duration;
}