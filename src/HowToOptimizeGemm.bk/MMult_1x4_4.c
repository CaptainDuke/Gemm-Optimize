/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot1x4(int, double *, int, double *, int, double *, int);
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C , count 4 per time*/
    for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */
      AddDot1x4(k, &A(i,0), lda, &B(0, j), ldb, &C(i,j), ldc);


    }
  }
}

void AddDot1x4(int k, double* a, int lda, double *b, int ldb, double * c, int ldc)
{
  int p;
  for(p = 0; p < k; p++)
  {
    C(0,0) += A(0,p) * B(p,0);
  }

  for(p = 0; p < k; p++)
  {
    C(0,1) += A(0,p) * B(p,1);
  }

  for(p = 0; p < k; p++)
  {
    C(0,2) += A(0,p) * B(p,2);
  }

  for(p = 0; p < k; p++)
  {
    C(0,3) += A(0,p) * B(p,3);
  }
}



/* Create macro to let X( i ) equal the ith element of x */

#define X(i) x[ (i)*incx ]

void AddDot( int k, double *x, int incx,  double *y, double *gamma )
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.

     Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */
 
  int p;

  for ( p=0; p<k; p++ ){
    *gamma += X( p ) * y[ p ];     // notice: column first
  }
}
