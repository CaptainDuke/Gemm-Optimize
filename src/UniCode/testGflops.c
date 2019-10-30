
/*
  Useage: 
            gcc-9 -g -O1 -msse3 -mfma testGflops.c
            icc -g -O2 -msse3 -mfma testGflops.c
 */

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

// ====================== main ================================================
int main(){  
  float
    dtime, dtime_best,        
    gflops, 
    diff;


  long NN = 1000000;
  gflops = 8* NN * 2.0*1.0e-09;

  __m256 areg = _mm256_set1_ps(0.0);
  __m256 breg = _mm256_set1_ps(0.0);
  __m256 creg = _mm256_set1_ps(0.0);
  __m256 dreg = _mm256_set1_ps(0.0);
  __m256 ereg = _mm256_set1_ps(0.0);

  __m256 freg = _mm256_set1_ps(0.0);
  __m256 greg = _mm256_set1_ps(0.0);
  __m256 hreg = _mm256_set1_ps(0.0);
  __m256 ireg = _mm256_set1_ps(0.0);
  __m256 jreg = _mm256_set1_ps(0.0);

  for(int loop = 0; loop < 3; loop++){
    dtime = dclock();

    for(long i = 0; i < NN; i++){
        areg = _mm256_fmadd_ps(areg, areg,areg);
        breg = _mm256_fmadd_ps(breg, breg,breg);
        creg = _mm256_fmadd_ps(creg, creg,creg);
        dreg = _mm256_fmadd_ps(dreg, dreg,dreg);
        ereg = _mm256_fmadd_ps(ereg, ereg,ereg);

        freg = _mm256_fmadd_ps(freg, freg,freg);
        greg = _mm256_fmadd_ps(greg, greg,greg);
        hreg = _mm256_fmadd_ps(hreg, hreg,hreg);
        ireg = _mm256_fmadd_ps(ireg, ireg,ireg);
        jreg = _mm256_fmadd_ps(jreg, jreg,jreg);
    }
    
    dtime = dclock() - dtime;
    printf("%le %le %le %le %le    %le %le %le %le %le \n", 
            areg[0], breg[0], creg[0], dreg[0], ereg[0],
            freg[0], greg[0], hreg[0], ireg[0], jreg[0]
            );
            
    if( loop == 0)
      dtime_best = dtime;
    else
      dtime_best = (dtime < dtime_best ? dtime : dtime_best);
  }
  printf( "%ld %le \n", NN, gflops * 10 / dtime_best );
  exit( 0 );
}