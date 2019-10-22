/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */

void AddDot( int, float *, int, float *, float * );
void AddDot1x4(int, float *, int, float *, int, float *, int);

void MY_MMult( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;
  for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
    for ( j=0; j<n; j+=1 ){        /* Loop over the columns of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */
      AddDot1x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
   
    }
  }
}

void AddDot1x4(int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  AddDot( k, &A( 0,0 ), ldb, &B( 0,0 ), &C( 0,0 ) );
  AddDot( k, &A( 1,0 ), ldb, &B( 0,0 ), &C( 1,0 ) );
  AddDot( k, &A( 2,0 ), ldb, &B( 0,0 ), &C( 2,0 ) );
  AddDot( k, &A( 3,0 ), ldb, &B( 0,0 ), &C( 3,0 ) );

  // int p;
  // for(p = 0; p < k; p++){
  //   C(0, 0) += A(0, p) * B(p, 0);
  //   C(1, 0) += A(1, p) * B(p, 0);
  //   C(2, 0) += A(2, p) * B(p, 0);
  //   C(3, 0) += A(3, p) * B(p, 0);

  // }
}

/* Create macro to let X( i ) equal the ith element of x */

// #define X(i) x[ (i)*incx ]
#define Y(i) y[ (i)*incy ]

void AddDot( int k, float *x, int incy,  float *y, float *gamma )
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.

     Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */
 
  int p;

  for ( p=0; p<k; p++ ){
    // *gamma += X( p ) * y[ p ];     
    *gamma += x[p] * Y(p);     
  }
}
