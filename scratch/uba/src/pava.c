/* pava.c: R extension, PAVA (Pool Adjacent Violators Algorithm)  */
/* By Bahjat Qaqish */
/************************************************************/

//#include <R.h>

//typedef  double DBL;

#include <stdio.h>
#include <math.h>
/*************************************************/
/*
Pool y[i:j] using weights w[i:j]
*/
 void wpool (double *y, double *w, int i, int j)
{
  int k;
  double s0=0, s1=0;

  for (k=i; k<=j; k++) {s1 += y[k]*w[k]; s0 += w[k];}
  s1 /= s0;
  for (k=i; k<=j; k++) y[k] = s1;
}

/*************************************************/

 void wpava  (double *y, double *w, int *np)
/*
Apply weighted pava to y[0:n-1] using weights w[0:n-1]
*/
{
  int npools, n = *np;

  if (n <= 1) return;
  n--;

  /* keep passing through the array until pooling is not needed */
  do {
    int i = 0;
    npools = 0;
    while (i < n) {
      int k = i;
      /* starting at y[i], find longest non-increasing sequence y[i:k] */
      while (k < n && y[k] >= y[k+1])  k++;
      if (y[i] != y[k]) {wpool(y, w, i, k); npools++;}
      i = k+1;
    }
  } while (npools > 0);
}

/*************************************************/

 void upool (double *y, int i, int j)
/*
Pool y[i:j]
*/
{
  int k;
  double s=0;

  
  //printf("Upool %d - %d -> %f - %f \n",i,j, y[i],y[j]);

  for (k=i; k<=j; k++) {s += y[k];}
  s /= (j-i+1);

  //for (k=i; k<=j; k++) printf("%f ",y[k]);
  //printf("\n s = %f\n",s);
  

  for (k=i; k<=j; k++) y[k] = s;
}

/*************************************************/

 void  upava  (double *y, int *np)
/*
Apply pava to y[0:n-1]
*/
{
  int npools, n = *np;

  if (n <= 1) return;
  n--;

  /* keep passing through the array until pooling is not needed */
  do {
    int i = 0;
    npools = 0;
    while (i < n) {
      int k = i;
      /* starting at y[i], find longest non-increasing sequence y[i:k] */
      while (k < n && (double)(y[k] - y[k+1]) >= 0.0)  k++;
      if (fabs(y[i] - y[k]) != 0.0) {upool(y, i, k); npools++;}
      i = k+1;
    }
  } while (npools > 0);
}

/*************************************************/

void  pava  (double *y, double *w, int *np)
/*
Apply pava to y[0:n-1] using weights w[0:n-1]
Calls an unweighted version if all weights are equal and != 0
Does nothing if all weights are == 0
Calls a weighted version otherwise
*/
{
  int n = *np, i=1;
  double w0;

  if (n <= 1) return;

  w0 = w[0];
  while (i < n && w[i] == w0) i++;
  if (i == n) {
    if (w0 == 0.0) return;  /* all weights are == 0 */
    else upava(y, np);      /* unweighted */
  }
  else wpava(y, w, np);     /* weighted   */
}

