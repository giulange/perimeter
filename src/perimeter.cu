/**
 * Copyright 2015-2020 GeoProc Service.  All rights reserved.
 *
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation outside the terms of the EULA is strictly prohibited.
 *
 */

//	INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>        	/* errno */
#include <string.h>       	/* strerror */
#include <math.h>			// ceil
#include <time.h>			// CLOCKS_PER_SEC
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
// GIS
#include "/home/giuliano/git/cuda/weatherprog-cudac/includes/gis.h"

/**
 * 	PARS
 */
static const unsigned int 	threads 					= 512;//[reduce6]	No of threads working in single block
static const unsigned int 	blocks 						= 64;// [reduce6]	No of blocks working in grid (this gives also the size of output Perimeter, to be summed outside CUDA)
static const unsigned int 	mask_len					= 40;//	[tidx2_ns]	No of pixels processed by single thread
static const unsigned int	gpuDev						= 0;
bool 						print_intermediate_arrays 	= false;
const char 					*BASE_PATH 					= "/home/giuliano/git/cuda/perimeter";
/*
 * 		DEFINE I/O files
 */
//	** INPUT
//const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/fragmentation/data/ROI.tif";
//const char 		*FIL_BIN 		= "/home/giuliano/git/cuda/fragmentation/data/BIN.tif";
//const char 		*FIL_ROI        = "/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954_roi.tif";
//const char		*FIL_BIN        = "/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954.tif";
//const char 		*FIL_ROI		= "/home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped_roi.tif";
//const char 		*FIL_BIN		= "/home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped.tif";
//const char		*FIL_ROI		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2_roi.tif";
//const char		*FIL_BIN		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2.tif";
// *most used one*
const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/fragmentation/data/lodi1954_roi.tif";
const char 		*FIL_BIN 		= "/home/giuliano/git/cuda/fragmentation/data/lodi1954.tif";
//const char 		*FIL_ROI		= "/home/giuliano/git/cuda/perimeter/data/imp_mosaic_char_2006_cropped_64kpixels_roi.tif";
//const char 		*FIL_BIN		= "/home/giuliano/git/cuda/perimeter/data/imp_mosaic_char_2006_cropped_64kpixels.tif";

//	** OUTPUT
const char 		*FIL_PERI		= "/home/giuliano/git/cuda/perimeter/data/PERI-cuda.tif";

/*
 *	kernel labels
 */
const char 		*kern_1 		= "gtranspose"		;
const char 		*kern_2 		= "tidx2_ns"		;
const char 		*kern_3 		= "gtranspose"		;
const char 		*kern_4 		= "tidx2_ns"		;
const char 		*kern_5 		= "reduce6_nvidia"	;
char			buffer[255];

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

bool isPow2(unsigned int x){ return ((x&(x-1))==0); }

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
// reduce6<T, threads, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
//dim3 dimBlock(threads, 1, 1);
//dim3 dimGrid(blocks, 1, 1);
// when there is only one warp per block, we need to allocate two warps
// worth of shared memory so that we don't index shared memory out of bounds
//int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(const T *g_idata, const unsigned char *ROI, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid 		= threadIdx.x;
    unsigned int i 			= blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize 	= blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i] * ROI[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) mySum += g_idata[i+blockSize] * ROI[i];
        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) if (tid < 256) sdata[tid] = mySum = mySum + sdata[tid + 256]; __syncthreads();
    if (blockSize >= 256) if (tid < 128) sdata[tid] = mySum = mySum + sdata[tid + 128]; __syncthreads();
    if (blockSize >= 128) if (tid <  64) sdata[tid] = mySum = mySum + sdata[tid +  64]; __syncthreads();
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64) smem[tid] = mySum = mySum + smem[tid + 32];
        if (blockSize >=  32) smem[tid] = mySum = mySum + smem[tid + 16];
        if (blockSize >=  16) smem[tid] = mySum = mySum + smem[tid +  8];
        if (blockSize >=   8) smem[tid] = mySum = mySum + smem[tid +  4];
        if (blockSize >=   4) smem[tid] = mySum = mySum + smem[tid +  2];
        if (blockSize >=   2) smem[tid] = mySum = mySum + smem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template<typename T>
__global__
void gtranspose(T *O, const T *I, unsigned WIDTH, unsigned HEIGHT)
{
	unsigned int tix = threadIdx.x;
	unsigned int tiy = threadIdx.y;
	unsigned int bix = blockIdx.x;
	unsigned int biy = blockIdx.y;
	unsigned int bdx = blockDim.x;
	unsigned int bdy = blockDim.y;

	//					  |--grid------|   |-block--|   |-thread--|
	unsigned int itid	= WIDTH*bdy*biy  + WIDTH*tiy  + bix*bdx+tix;
	unsigned int otid	= HEIGHT*bdx*bix + HEIGHT*tix + biy*bdy+tiy;
	unsigned int xtid	= bix*bdx+tix;
	unsigned int ytid	= biy*bdy+tiy;

	if( xtid<WIDTH && ytid<HEIGHT ){
		O[ otid ] = I[ itid ];
		//__syncthreads();
	}
}

template<typename T>
__global__ void tidx2_ns(	const unsigned char	*IN			,
							unsigned int 		map_width	,
							unsigned int 		map_height	,
							T 					*OUT		,
							unsigned int		mask_len	){

	/* NOTES:
	 * The CUDA algorithm manage the object borders at the map border as they
	 * were countorned by non-object pixels. This mean that a 1-pixel object on
	 * the map border has perimeter 4 and not 3. This should be the same
	 * approach used by the MatLab "bwperim" built-in function.
	 * This choice was taken to solve the current trouble, where the *1 must be
	 * accounted properly:
	 *	 | ...		 |
	 *   | 0 0 0 ... |
	 *   | 1 1 1 ... |
	 *   |*1 1 1 ... |
	 *   | 1 1 1 ... |
	 *   | 0 0 0 ... |
	 *	 | ...		 |
	 * The *1 pixel would contribute to the calculation of overall perimeter with
	 * "zero" value, while the most accurate for my purpose is a value equal to "one".
	 * This can happen only considering that outside the map we assume every pixel
	 * is zero, as the following:
	 * 0 | ...		 |
	 * 0 | 0 0 0 ... |
	 * 0 | 1 1 1 ... |
	 * 0 |*1 1 1 ... |
	 * 0 | 1 1 1 ... |
	 * 0 | 0 0 0 ... |
	 * 0 | ...		 |
	 * This pattern is valid for both East–Ovest and North–South searching
	 * directions.
	 *
	 * The CUDA algorithm is based on running THIS kernel twice for:
	 * 	(1) East–West dir 	(after transpose "|" ––> "––")
	 * 	(2) North–South dir (after transpose "––" ––> "|")
	 * THIS kernel performs the algebraic sum of three rows:
	 *   > the row at top 	 of current pixel  +[ tid+(ii-1)+map_width ]
	 *   > the row at bottom of current pixel  -[ tid+(ii+1)+map_width ]
	 *
	 *  Particular cases are figured out according to blockIdx.x position.
	 *  See later comments!
	 */
	unsigned int ii			= 0;
	unsigned int tix 		= blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int tiy 		= blockIdx.y*mask_len;
	unsigned int tid 		= tix + tiy*map_width;
	unsigned int latest_row	= 0;

	if( tix < map_width ){
		/*	Here I distinguish between 4 kind of tiles(i.e. of blockIdx.y):
		 * 		> 0			The case is particular only for threads at first row.
		 * 		> [1,end-1]	Cases are all general (I always have one row at North and one at South).
		 * 		> end		The case is particular only for threads at last  row.
		 */
		// ***first tile*** top map border
		if(blockIdx.y==0){
			ii=0;
				// here I assume that the pixel on the border has a non-object pixel outside, because I use 2*IN[tid+ii*map_width]
				// I should use 1*IN[tid+ii*map_width] to assume that objects always continues outside!!
				OUT[tid+ii*map_width] += (2*IN[tid+ii*map_width] -0 					   -IN[tid+(ii+1)*map_width] )*IN[tid+ii*map_width];
			for(ii=1;ii<mask_len;ii++)
				OUT[tid+ii*map_width] += (2*IN[tid+ii*map_width] -IN[tid+(ii-1)*map_width] -IN[tid+(ii+1)*map_width] )*IN[tid+ii*map_width];
		}
		// ***all centre tiles*** inside the map
		if(blockIdx.y>0 && blockIdx.y<gridDim.y-1){
			/*	This is the most general case/formulation:
			 */
			for(ii=0;ii<mask_len;ii++)
				OUT[tid+ii*map_width] += (2*IN[tid+ii*map_width] -IN[tid+(ii-1)*map_width] -IN[tid+(ii+1)*map_width] )*IN[tid+ii*map_width];
		}
		// ***last tile*** bottom map border
		if(blockIdx.y==gridDim.y-1){
			latest_row = map_height-tiy-1;// e.g. 210 -(4*50) -1 = 9 ==>Ok!!!
			for(ii=0;ii<latest_row;ii++)
				OUT[tid+ii*map_width] += (2*IN[tid+ii*map_width] -IN[tid+(ii-1)*map_width] -IN[tid+(ii+1)*map_width] )*IN[tid+ii*map_width];
			ii=latest_row;
				// the pattern of how objects are accounted on the border is the same reported for top map border
				OUT[tid+ii*map_width] += (2*IN[tid+ii*map_width] -IN[tid+(ii-1)*map_width] -0 						 )*IN[tid+ii*map_width];
		}
	}
}

/**
 * Explanation here.
 */
int main( int argc, char **argv ) {

	/*
	 * 		ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.
	// ** metadata
	metadata 			MDbin,MDroi,MDdouble,MDbin_trans,MDdouble_trans; // ,MDtranspose
	unsigned int		map_len;
	// ** time
	clock_t				start_t,end_t;
	unsigned int 		elapsed_time	= 0;
	// ** gpu card
	cudaDeviceProp		devProp;
	//unsigned int		gpuDev=0;
	// ** count the number of kernels that must print their output:
	unsigned int 		count_print = 0;

	// query current GPU properties:
	CUDA_CHECK_RETURN( cudaSetDevice(gpuDev) );
	cudaGetDeviceProperties(&devProp, gpuDev);

	/*
	 * 		LOAD METADATA
	 */
	MDbin					= geotiffinfo( FIL_BIN, 1 );
	MDroi 					= geotiffinfo( FIL_ROI, 1 );
	// set metadata to eventually print arrays after any CUDA kernel:
	MDdouble 				= MDbin;
	MDdouble.pixel_type		= GDT_Float64;
	// set metadata for transposed array
	MDbin_trans 			= MDbin;
	MDbin_trans.width 		= MDbin.heigth;
	MDbin_trans.heigth 		= MDbin.width;
	MDdouble_trans 			= MDdouble;
	MDdouble_trans.width 	= MDdouble.heigth;
	MDdouble_trans.heigth 	= MDdouble.width;

	// Set size of all arrays which come into play:
	map_len 				= MDbin.width*MDbin.heigth;
	size_t	sizeChar		= map_len*sizeof( unsigned char );
	size_t	sizeDouble		= map_len*sizeof( double );

	/*
	 * 	INITIALIZE CPU & GPU ARRAYS
	 * 	This is the cheapest chain of calculations (in terms of gpu storage) that sets the required arrays:
	 * 			  #1				     #2					#3					         #4					       #5
	 * 		BIN ––(transpose)––> ROI ––––(2TID-NS)––> TMP ––(transpose)––> PERI || BIN ––(2TID-NS)––> PERI & ROI ––(reduce)––> ∑Perimeter
	 * 		char				 char				  double			   doub    char				  doub & char				double
	 * 																									     |––>load ROI now!!
	 *	I need only {BIN,ROI,TMP,PERI}
	 */
	// **host
	unsigned char *BIN		= (unsigned char *) CPLMalloc( sizeChar );
	unsigned char *ROI 		= (unsigned char *) CPLMalloc( sizeChar );
	double 					*host_PERI,*h_print_double;
	unsigned char			*h_print_uchar;
	// **dev
	unsigned char			*dev_BIN, *dev_ROI;
	double 					*dev_PERI,*dev_TMP;
	// initialize grids on CPU MEM:
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_PERI, 	sizeDouble) );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&h_print_double,sizeDouble) );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&h_print_uchar,	sizeChar) 	);
	// initialize grids on GPU MEM:
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_BIN, 		sizeChar) 	);
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_ROI,  	sizeChar) 	);
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_TMP, 		sizeDouble) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_PERI,  	sizeDouble) );
	CUDA_CHECK_RETURN( cudaMemset(		dev_TMP, 0,  			sizeDouble) );// check if this works on Jcuda/Java (I remember some troubles)
	/*
	 * 	BIN
	 */
	// load
	printf("Importing...\t%s\n",FIL_BIN);
	geotiffread( FIL_BIN, MDbin, &BIN[0] );
	// H2D:
	CUDA_CHECK_RETURN( cudaMemcpy(dev_BIN, BIN, 	sizeChar, cudaMemcpyHostToDevice) );
	// memset:
/*	CUDA_CHECK_RETURN( cudaMemset(dev_ROI, 0,  					sizeDouble) );
	CUDA_CHECK_RETURN( cudaMemset(dev_BIN, 0,  					sizeDouble) );
*/


	/*
	 * 		KERNELS GEOMETRY
	 * 		NOTE: use ceil() instead of the "%" operator!!!
	 */
	int sqrt_nmax_threads = floor(sqrt( devProp.maxThreadsPerBlock ));
	unsigned int 	gdx_kTidx2NS, gdy_kTidx2NS, gdx_trans, gdy_trans, gdx_kTidx2NS_t, gdy_kTidx2NS_t;
	// k(gtransform)
	gdx_trans 	= ((unsigned int)(MDbin.width  % sqrt_nmax_threads)>0) + MDbin.width  / sqrt_nmax_threads;
	gdy_trans 	= ((unsigned int)(MDbin.heigth % sqrt_nmax_threads)>0) + MDbin.heigth / sqrt_nmax_threads;
	dim3 block_trans( sqrt_nmax_threads, sqrt_nmax_threads, 1);// ––> block geometry is always the same
	dim3 grid_trans ( gdx_trans, gdy_trans );				   // ––> grid geometry for "go" 	transpose
	dim3 grid_trans2( gdy_trans, gdx_trans );				   // ––> grid geometry for "back"	transpose

	// k(2*TID - NS)
	gdx_kTidx2NS 	= ((unsigned int)(MDbin.width % (sqrt_nmax_threads*sqrt_nmax_threads))>0) + (MDbin.width  / (sqrt_nmax_threads*sqrt_nmax_threads));
	gdy_kTidx2NS 	= (unsigned int)((MDbin.heigth % mask_len)>0) + floor(MDbin.heigth / mask_len);
	dim3 block_kTidx2NS( sqrt_nmax_threads*sqrt_nmax_threads,1,1 );
	dim3 grid_kTidx2NS ( gdx_kTidx2NS,gdy_kTidx2NS,1 );
	gdx_kTidx2NS_t 	= ((unsigned int)(MDbin.heigth % (sqrt_nmax_threads*sqrt_nmax_threads))>0) + (MDbin.heigth  / (sqrt_nmax_threads*sqrt_nmax_threads));
	gdy_kTidx2NS_t 	= (unsigned int)((MDbin.width % mask_len)>0) + floor(MDbin.width / mask_len);
	dim3 block_kTidx2NS_t( sqrt_nmax_threads*sqrt_nmax_threads,1,1);
	dim3 grid_kTidx2NS_t ( gdx_kTidx2NS_t,gdy_kTidx2NS_t,1);

	/*		KERNELS INVOCATION
	 *
	 *			*************************
	 *			-1- gtranspose	= ƒ( BIN  ––> ROI  )  ::  "|"	–> 	"––"
	 *			-2- tidx2_ns 	= ƒ( ROI  ––> TMP  )  ::  "––" 	–> 	"––" 		{+East/Ovest}
	 *			-3- gtranspose 	= ƒ( TMP  ––> PERI )  ::  "––" 	–> 	"|"
	 *			-4- tidx2_ns 	= ƒ( BIN  ––> PERI )  ::  "|" 	–> 	"|"			{+North/South}
	 *			-5- reduce6 	= ƒ( PERI,ROI  	   )  ::  "|*|" –> 	"∑Perimeter"
	 *			*************************
	 * 			  #1				     #2					#3					         #4					       #5
	 * 		BIN ––(transpose)––> ROI ––––(2TID-NS)––> TMP ––(transpose)––> PERI || BIN ––(2TID-NS)––> PERI & ROI ––(reduce)––> ∑Perimeter
	 *
	 */
	printf("\n\n");

	// ***-1-***	BIN ––(transpose)––> ROI
	start_t = clock();
	gtranspose<unsigned char><<<grid_trans,block_trans>>>( dev_ROI, dev_BIN, MDbin.width, MDbin.heigth );
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_1,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(h_print_uchar,dev_ROI,	(size_t)sizeChar,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_1);
		geotiffwrite( FIL_BIN, buffer, MDbin_trans, h_print_uchar );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	// ***-2-***	ROI ––––(2TID-NS)––> TMP
	start_t = clock();
	tidx2_ns<double><<<grid_kTidx2NS_t,block_kTidx2NS_t>>>( dev_ROI, MDbin.heigth, MDbin.width, dev_TMP, mask_len );
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_2,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(h_print_double,dev_TMP,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_2);
		geotiffwrite( FIL_BIN, buffer, MDdouble_trans, h_print_double );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	// ***-3-***	TMP ––(transpose)––> PERI
	start_t = clock();
	gtranspose<double><<<grid_trans2,block_trans>>>( dev_PERI, dev_TMP, MDbin.heigth, MDbin.width );
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_3,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(h_print_double,dev_PERI,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_3);
		geotiffwrite( FIL_BIN, buffer, MDdouble, h_print_double );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	// ***-4-***	BIN ––(2TID-NS)––> PERI
	start_t = clock();
	tidx2_ns<double><<<grid_kTidx2NS,block_kTidx2NS>>>( dev_BIN, MDbin.width, MDbin.heigth, dev_PERI, mask_len );
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_4,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(h_print_double,dev_PERI,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_4);
		geotiffwrite( FIL_BIN, buffer, MDdouble, h_print_double );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	// ***-5-***	PERI & ROI ––(reduce)––> Perimeter
	/*
	 * 	ROI, import now because before the GPU mem space was used for temporary stuff!
	 */
	// load
	printf("Importing...\t%s\n",FIL_ROI);
	geotiffread( FIL_ROI, MDroi, &ROI[0] );
	// H2D:
	CUDA_CHECK_RETURN( cudaMemcpy(dev_ROI, ROI, 	sizeChar, cudaMemcpyHostToDevice) );
	//–––––– tmp
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(  blocks, 1, 1);
	double *h_Perimeter, *d_Perimeter;// it's size is equal to the number of blocks within grid!
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&d_Perimeter,  blocks*sizeof( double )) );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void **)&h_Perimeter,	blocks*sizeof( double )) );
	//CUDA_CHECK_RETURN( cudaMemset(dev_PERI, 1,  			sizeDouble) );
	start_t = clock();
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
	//–––––– tmp
	start_t = clock();
	if (isPow2(map_len)){ reduce6<double, 512, true> <<< dimGrid, dimBlock, smemSize >>>(dev_PERI, dev_ROI, d_Perimeter, map_len);
	}else{	 			  reduce6<double, 512, false><<< dimGrid, dimBlock, smemSize >>>(dev_PERI, dev_ROI, d_Perimeter, map_len);}
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_5,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	//printf("  -%d- %12s\t%6f [msec]\n",++count_print,kern_5,(int)( (double)(end_t  - start_t ) ));
/*	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_IO,dev_IO,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_5);
		geotiffwrite( FIL_BIN, buffer, MDdouble, host_IO );
	}
*/	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	// total
	printf("______________________________________\n");
	printf("  %16s\t%6d [msec]\n\n", "Total time:",elapsed_time );

	int sumPerimeter = 0;
	CUDA_CHECK_RETURN( cudaMemcpy(h_Perimeter,d_Perimeter,	(size_t)blocks*sizeof( double ),cudaMemcpyDeviceToHost) );
	for(unsigned int ii=0;ii<blocks;ii++){ sumPerimeter += h_Perimeter[ii]; }
	printf("Perimeter = %d\n\n",sumPerimeter);

	// save the map with pixel-by-pixel computed perimeter (for checking purpose, within MatLab)
	CUDA_CHECK_RETURN( cudaMemcpy(host_PERI,dev_PERI,		(size_t)sizeDouble,				cudaMemcpyDeviceToHost) );
	// save on HDD
	geotiffwrite( FIL_ROI, FIL_PERI, MDdouble, host_PERI );

	return 0;
}
