#include <stdio.h>
#include <stdlib.h>
#include "parameters.h"
#include <sys/time.h>
#include <time.h>
#include "MyMM.h"


#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>  // fma

static double gtod_ref_time_sec = 0.0;

#define A(i,j) a[ (i)*lda + (j) ]  // row first
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

double dclock()
{
        double         the_time, norm_sec;
        struct timeval tv;

        gettimeofday( &tv, NULL );

        if ( gtod_ref_time_sec == 0.0 )
                gtod_ref_time_sec = ( double ) tv.tv_sec;

        norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;

        the_time = norm_sec + tv.tv_usec * 1.0e-6;

        return the_time;
}

void random_matrix( int m, int n, float *a, int lda )
{
  // float frand48();
  int i,j;
  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
      A( i,j ) = 2.0 * (float)drand48( ) - 1.0;
}


/* Routine for computing C = A * B + C */

void REF_MMult( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j, p;
  for ( i=0; i<m; i++ ){
    for ( j=0; j<n; j++ ){
      for ( p=0; p<k; p++ ){
	      C( i,j ) = C( i,j ) +  A( i,p ) * B( p,j );
      }
    }
  }
}


void copy_matrix( int m, int n, float *a, int lda, float *b, int ldb )
{
  int i, j;
  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
      B( i,j ) = A( i,j );
}
  
float compare_matrices( int m, int n, float *a, int lda, float *b, int ldb )
{
  int i, j;
  float max_diff = 0.0, diff;
  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
    {
      diff = abs( A( i,j ) - B( i,j ) );
      max_diff = ( diff > max_diff ? diff : max_diff );
    }

  return max_diff;
}



// ====================== main =================================================

int main(){  
  // int 
  //   p, 
  //   m, n, k,
  //   lda, ldb, ldc, 
  //   rep;

  float
    dtime, dtime_best,        
    gflops, 
    diff;

  float 
    *pa, *pb, *pc, *pdst, *cref, *cold;    
  
  // printf( "MY_MMult = [\n" );


  float a[4] = {1.0, 2.0, 3.0, 4.0};
  float b[4] = {1.0, 1.0, 1.0, 1.0};
  float c[4] = {1.0, 1.0, 1.0, 1.0};
  float dst[4] ;
    pa = (float *) malloc(4 * sizeof(float));
    pb = (float *) malloc(4 * sizeof(float));
    pc = (float *) malloc(4 * sizeof(float));
  // pa = &a;
  // pb = &b;
  pdst = &dst[0];
  __m128 dstreg ;
  __m128 ereg = _mm_setzero_ps();

  __m128 freg = _mm_setzero_ps();
  __m128 greg = _mm_setzero_ps();
  __m128 hreg = _mm_setzero_ps();
  __m128 ireg = _mm_setzero_ps();
  __m128 jreg = _mm_setzero_ps();
  
  

  int NN = 10000;
  gflops = NN * 2.0*1.0e-09;

  // dstreg = _mm_fmadd_ps(areg, breg, creg);
  // _mm_store_ps(pc, creg);

  // for(int i  = 0 ; i < 4; i ++){
  //   printf("%i  %f \n", i ,*(pc+i));
  // }

  for(int loop = 0; loop < 3; loop++){

  __m128 areg = _mm_loadu_ps(&a[0]);
  __m128 breg = _mm_loadu_ps(&b[0]);
  __m128 creg = _mm_loadu_ps(&c[0]);
  dstreg = _mm_setzero_ps();


    dtime = dclock();
    for(int i = 0; i < NN/10; i++){
      dstreg = _mm_fmadd_ps(dstreg, dstreg, dstreg);
      areg = _mm_fmadd_ps(areg, areg,areg);
      breg = _mm_fmadd_ps(breg, breg,breg);
      creg = _mm_fmadd_ps(creg,creg,creg);
      ereg = _mm_fmadd_ps(ereg, ereg,ereg);

      freg = _mm_fmadd_ps(freg, freg,freg);
      greg = _mm_fmadd_ps(greg, greg,greg);
      hreg = _mm_fmadd_ps(hreg, hreg,hreg);
      ireg = _mm_fmadd_ps(ireg, ireg,ireg);
      jreg = _mm_fmadd_ps(jreg, jreg,jreg);

      // dstreg = _mm_fmadd_ps(areg, breg, dstreg);
      // creg = _mm_fmadd_ps(areg, breg, breg);
      
    }
    dtime = dclock() - dtime;
    if( loop == 0)
      dtime_best = dtime;
    else
      dtime_best = (dtime < dtime_best ? dtime : dtime_best);
  }
  
  _mm_store_ps(pdst, dstreg);
  for(int i  = 0 ; i < 4; i ++){
    printf("%i  %f \n", i ,*(pdst+i));
  }
  
  printf( "%d %le \n", NN, gflops / dtime_best );
  // free(a);
  // free(b);

  



  // for(int i = 0; i < 1000; i++){

  // }


  // for( p = PFIRST; p<=PLAST; p+=PINC){
  //   m = p;
  //   n = p;
  //   k = p;

  //   gflops = 2.0 * m * n * k * 1.0e-09;

  //   lda = k;
  //   ldb = n;
  //   ldc = n;

  //   // a = (float *) malloc(lda * (k+1) * sizeof(float));
  //   a = (float *) malloc(m * lda * sizeof(float));
  //   b = (float *) malloc((k+1) * ldb * sizeof(float));
  //   c = (float *) malloc(m* ldc  * sizeof(float));
  //   cold = (float *) malloc(m * ldc * sizeof(float));
  //   cref = (float *) malloc(m * ldc * sizeof(float));

  //   random_matrix( m, k, a, lda);       // a = random
  //   random_matrix( k, n, b, ldb);       // b = random
  //   random_matrix( m, n, cold, ldc);    // cold = random

  //   copy_matrix( m, n, cold, ldc, cref, ldc);   // cref = cold

  //   REF_MMult(m, n, k, a, lda, b, ldb, cref, ldc);

  //   for( rep = 0; rep < NREPEATS; rep++){
  //     copy_matrix( m, n, cold, ldc, c, ldc);    // c = cold
  //     dtime = dclock();

  //     MY_MMult( m, n, k, a, lda, b, ldb, c, ldc);

  //     dtime = dclock() - dtime;

  //     if( rep == 0)
  //       dtime_best = dtime;
  //     else
  //       dtime_best = (dtime < dtime_best ? dtime : dtime_best);
  //   }

  //   diff = compare_matrices( m, n, c, ldc, cref, ldc);

  //   printf( "%d %le %le \n", p, gflops / dtime_best, diff );

  //   free(a);
  //   free( b );
  //   free( c );
  //   free( cold );
  //   free( cref );

  // }
  // printf( "];\n" );

  exit( 0 );
    
}