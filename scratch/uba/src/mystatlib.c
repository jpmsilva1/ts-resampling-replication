
/* Various useful statistical subroutines */

const char *statlib_version_date="April 10, 2007";

#include "mystatlib.h"  /* This checks if any prototypes are wrong */
                      /* This also protects forward references in this file */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>    /* Basic math functions like sqrt() and log() */
#include <signal.h>  /* For set_newsignals() */
#include <time.h>    /* For initializing a random-number generator */

/* Many (or most) of the following functions were adapted
     from Press etal ``Numerical recipes in C'', 2nd edition */


double cdf( double x)
{

  double z1 = exp(-x*x/2);
  double z2 = exp(-x*x*0.5857864); // const is 2-sqrt(2)
  double s = 7*z1 + 16*z2 + (7 + 0.7853982*x*x)*z1*z1; // const is pi/4
  double t = sqrt(1 - s/30) / 2;
  return x > 0 ? t + 0.5 : 0.5 - t;

} 

double normcdf(double x, double mu, double sigma) {

  double new_x = (x - mu)/(sigma * sqrt(2));

  //printf("x %f mu %f s %f -> %f\n",x,mu,sigma,new_x);

  return stdnorm_cdf(new_x);
}

// Abromowitz and Stegun approximation 
double stdnorm_cdf(const double x)
{
  const double b1 =  0.319381530;
  const double b2 = -0.356563782;
  const double b3 =  1.781477937;
  const double b4 = -1.821255978;
  const double b5 =  1.330274429;
  const double p  =  0.2316419;
  const double c  =  0.39894228;

  if(x >= 0.0) {
      double t = 1.0 / ( 1.0 + p * x );
      return (1.0 - c * exp( -x * x / 2.0 ) * t *
      ( t *( t * ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
  }
  else {
      double t = 1.0 / ( 1.0 - p * x );
      return ( c * exp( -x * x / 2.0 ) * t *
      ( t *( t * ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
    }
}

// Trapezoidal Rule: (b - a) * (f(b) + f(a))/2
double trapezoid(int n, const double *x, const double *y) {

  int i;
  double sum = 0.0, incr;

  for(i = 1; i < n; i++) {
    incr = (x[i] - x[i-1]) * (y[i] + y[i-1]);
    sum += incr;
  }

  return sum * 0.5;

}

/* Read an array  xx[]  of length  nn  and stores the sample mean */
/*   and sample variance in doubles whose ADDRESSES */
/*   are in `meanptr' and `varptr' */

void getmeanvar (const double *xx, int mm,
    double *xmeanptr, double *xvarptr)
 { if (xx!=NULL && mm>0)
    { int i;  double dsum,dmean,dvar;
      for (dsum=0.0, i=0; i<mm; i++) dsum+=xx[i];
      dmean=dsum/mm;  dvar=0.0;
      if (mm>1)
       { for (dsum=0.0, i=0; i<mm; i++)
           dsum+=(xx[i]-dmean)*(xx[i]-dmean);
         dvar=dsum/(mm-1); }
      /* Return whatever variables the user wants */
      if (xmeanptr!=NULL) *xmeanptr=dmean;
      if (xvarptr!=NULL) *xvarptr=dvar;
            }}

/* Return the median value of a sorted array of length mm
   If mm is odd (Example: 1,3,5,8,14  for mm=5)
      the median is at (m+1)/2 (Example: 3) and offset [m/2] 
   If mm is even (Example: 1,3,5,8  for mm=4)
      the median is the average of values at mm/2 and (mm+2)/2
      (offsets (mm-2)/2 and mm/2)  */

double getmedian_sorted (const double *xx, int mm)
 { double medval=0.0;
   if (xx!=NULL && mm>0)
    { if (mm%2) medval=xx[mm/2];
      else  medval=(xx[(mm-2)/2] + xx[mm/2])/2; }
   return medval; }



/* Put midranks of array  wwdat[]  of length ntot */
/*   into the array  mrank[]  at corresponding offsets  */

#define MR_DTOL 1e-12     /* Allow for round-off error in tie groups */

void makeranks(double *mrank, const double *wwdat, int ntot,
  int *tgsize, int *ntgptr)
 { int i,j,r, ntg;
   /* `local static' variables are permanent, but they can't be used */
   /*   (are not visible) to any other function. */
   /* These are buffers for an array of doubles and an array of */
   /*   pointers to doubles, along with the current buffer lengths. */
   /* Initial values of all permanent variables are always 0 (or NULL) */
   static double *wtemp, **mranptr;
   static int nxlen;
   /* Make sure that the buffers  wtemp[],mranptr[]  are present */
   /*   and are long enough. */
   if (ntot>nxlen)    /* Buffers  wtemp,mranptr  will overflow */
    { /* If already allocated, return memory to system (don't be greedy) */
      if (wtemp!=NULL) { free(wtemp);  free(mranptr); }
      nxlen=ntot+100;   /* Allow for some extra room */
      /* Allocate memory for buffers */
      wtemp=calloc(nxlen+3,sizeof(double));
      mranptr=calloc(nxlen+3,sizeof(double *)); }
   /* Copy wwdat[] to buffer  wtemp[]  so that it won't be corrupted */
   /* At the same time, initialize mranptr[] array to point to values */
   /*   in mrank[].  &(var) means the address of var, or a pointer to var. */
   /* The next step will sort (wtemp[r],mranptr[r]) together. */
   /* The  mranptr[r]  value means that this subroutine will always know */
   /*   where to put the midrank of  wwdat[r]  */
   for (r=0; r<ntot; r++)
    { wtemp[r]=wwdat[r];  mranptr[r]=&(mrank[r]); }
   /* Bubble-sort the values (wtemp[r],mranptr[r]) together */
   for (i=0; i<ntot; i++) for (j=i+1; j<ntot; j++)
     if (wtemp[i]>wtemp[j])    /* Out of order, since i<j */
      { /* Switch values of  (wtemp[i],mranptr[i]), (wtemp[j],mranptr[j]) */
        double *wptr, wval=wtemp[i];  wtemp[i]=wtemp[j];  wtemp[j]=wval;
        wptr=mranptr[i];  mranptr[i]=mranptr[j];  mranptr[j]=wptr; }

   /* Assign midranks */
   ntg=0;   /* Number of tie-groups of size > 1 */
   for (i=0; i<ntot; i=j)
    { int k, ts;  double mr, firstval=wtemp[i];  /* First of tie group */
      /* The tie-group will be from i to j-1, inclusive */
      /* The first statement after the `do' is always executed */
      /*   at least once */
      /* The loop repeats as long as the `while' condition is TRUE */
      /* When the loop ends, j will point to the next value or ntot */
      /* MR_DTOL allows for round-off error in tie groups */
      j=i;  do j++; while(j<ntot && fabs(wtemp[j]-firstval)<MR_DTOL);
      mr=(i+j+1)/2.0;  /* The average rank for the tie group */
      for (k=i; k<j; k++) *(mranptr[k])=mr;  /* Fill in the midranks */
      /* Accumulate tiegroup sums */
      ts=j-i;  /* The tie-group size */
      if (ts>1)
       { if (tgsize!=NULL) tgsize[ntg]=ts;
         ntg++; }}  /* Keep track of tiegroup sizes and numbers */
   /* If the user doesn't want the value of ntg, don't insist. */
   if (ntgptr!=NULL) *ntgptr=ntg;
        }


/* Convert (tiegroup,ntiegroups) array to rank-sum tiesum */

double cubic_tiesum(const int *tgsize, int ntiegroups)
 { int i;  double tiesum=0.0;
   if (tgsize!=NULL)
    { for (i=0; i<ntiegroups; i++)
       { int ts=tgsize[i];  tiesum+=(ts-1)*ts*(ts+1); }}
   return tiesum; }


/* Provide footnotes for a table */

char *starp (double pp)
 { if (pp<0.01) return "(**)";
   else if (pp<=0.05)  return "(*)";
   else return ""; }


/* Comparison function for sorting doubles using system `quick sort' */
/* If  zarr[]  is an array of nn doubles, then    */
/*                                                */
/*    qsort(zarr,nn, sizeof(double),dubsort);     */
/*                                                */
/* will sort  zarr[]  in increasing order. */

/* Returns -1 if in order, 1 if out of order, 0 if tie */

int sortdub(const void *p, const void *q)
 { double xx;
   const double *pd=p, *qd=q;  /* Convert to pointers to doubles */
   xx = (*pd) - (*qd);         /* Difference between values pointed to */
   if (xx > 0.0) return 1;       /* Out of order */
   else if (xx<0.0) return -1;   /* In order */
   /* else a tie value */
   return 0; }

/* Alternative name */

int dubsort(const void *p, const void *q)
 { return sortdub(p,q); }


/* Comparison function for sorting doubles using system `quick sort' */
/* If  iarr[]  is an array of nn integers, then   */
/*                                                */
/*    qsort(iarr,nn, sizeof(int),dubsort);        */
/*                                                */
/* will sort  iarr[]  in increasing order. */

/* Returns -1 if in order, 1 if out of order, 0 if tie */

int sortint(const void *p, const void *q)
 { int xx;
   const int *pd=p, *qd=q;     /* Convert to pointers to ints */
   xx = (*pd) - (*qd);         /* Difference between values pointed to */
   if (xx > 0) return 1;       /* Out of order */
   else if (xx<0) return -1;   /* In order */
   /* else a tie value */
   return 0; }

/* Alternative name */

int intsort(const void *p, const void *q)
 { return sortint(p,q); }


/* Often useful for squares and cubes of complicated expressions */

double square(double x)
 { return x*x; }

double cube(double x)
 { return x*x*x; }


/*
** sum_kahan
** =======
** Computes sums using the Kahan correction algorithm.
**
** This algorithm, also known as compensated summation, 
** significantly reduces the numerical error in the total obtained by adding a sequence of finite 
** precision floating point numbers, by keeping a separate running compensation (a variable to accumulate 
** small errors).  
*/

double sum_kahan(int n, const double *f) {
  
  double sum = f[0];
  double c = 0.0, y, t;
  int i;

  for(i = 1; i < n; i++) {

    y = f[i] - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;
  
  }

  return sum;
}

/* SCCS @(#)mysort.c	1.6 12/13/99 */
/*
** quick sort routine : sort a vector of floats, and carry along an int
**
**  x:     vector to sort on
**  start: first element of x to sort
**  stop:  last element of x to sort
**  cvec:  a vector to carry along
*/

int compareUp(double a, double b) {
  double res = a - b;//inceasing

  if(res < 0.0)
    return -1;
  if(res > 0.0)
    return 1;
  return 0.0;
  
}

int compareDn(double a, double b) {
  double res = b - a;//decreasing

  if(res < 0.0)
    return -1;
  if(res > 0.0)
    return 1;
  return 0.0;
  
}

void mysort(int start, int stop, double *x, int *cvec, int sortUp)
 {
 int i, j, k;
 double temp, median;
 int tempd;
 int  (*mycompareTo)();
 
 if(sortUp)
   mycompareTo = compareUp;
 else
   mycompareTo = compareDn;

  while (start < stop) {
    /*
    ** first-- if the list is short, do an ordinary insertion sort
    */
    if ((stop-start)<11) {
	for (i=start+1; i<=stop; i++) {
	    temp = x[i];
	    tempd= cvec[i];
	    j=i-1;

	    while (j >= start && 
		   (mycompareTo(x[j],temp) > 0.0)) {
		x[j+1] = x[j];
		cvec[j+1] = cvec[j];
		j--;
		}
	    x[j+1] = temp;
	    cvec[j+1]  = tempd;
	    }
	return;
	}

    /*
    ** list is longer -- split it into two
    **  I use the median of 3 values as the split point
    */
    i=start;
    j=stop;
    k = (start + stop)/2;

    median = x[k];
    if (mycompareTo(x[i],x[k]) >= 0.0) {      /* one of j or k is smallest */
    //    if (x[i] >= x[k]) {      /* one of j or k is smallest */
      //	if (x[j] > x[k]) {   /* k is smallest */
      if (mycompareTo(x[j],x[k]) > 0.0) {   /* k is smallest */
	//if (x[i] > x[j])  median = x[j];
	if (mycompareTo(x[i],x[j]) > 0.0)  median = x[j];
	else median= x[i];
      }
    }
    else {
      //if (x[j] < x[k]) {
      if (mycompareTo(x[j],x[k]) < 0.0) {
	if (mycompareTo(x[i],x[j]) > 0.0) median = x[i];
	else median = x[j];
      }
    }

    /* 
    **  Now actually do the partitioning 
    **   Because we must have at least one element >= median, "i"
    **   will never run over the end of the array.  Similar logic
    **   applies to j.
    ** A note on the use of "<" rather than "<=".  If a list has lots
    **   of identical elements, e.g. 80/100 are "3.5", then we will
    **   often go to the swap step with x[i]=x[j]=median.  But we will
    **   get the pointers i and j to meet approximately in the middle of
    **   the list, and that is THE important condition for speed in a
    **   quicksort.
    **   
    */
    while (i<j) {
	/*
	** top pointer down till it points at something too large
	*/
      //while (x[i] < median) i++;
      while (mycompareTo(x[i],median) < 0.0) i++;

	/*
	** bottom pointer up until it points at something too small
	*/
      //while(x[j] > median) j--;

      while(mycompareTo(x[j],median) > 0.0) j--;

	if (i<j) {
	  //if (x[i] > x[j]) {  /* swap */
	  if (mycompareTo(x[i],x[j]) > 0.0) {  /* swap */
	    temp = x[i];
	    x[i] = x[j];
	    x[j] = temp;
	    tempd= cvec[i];   cvec[i] =cvec[j];  cvec[j] =tempd;
	  }
	  i++; j--;
	}
    }

    /*
    ** The while() step helps if there are lots of ties.  It will break
    **  the list into 3 parts: < median, ==median, >=median, of which only
    **  the top and bottom ones need further attention.
    ** The ">=" is needed because i may be  == to j
    */
    //while (x[i] >= median && i>start) i--;
    while (mycompareTo(x[i],median) >= 0.0 && i>start) i--;
    
    //while (x[j] <= median && j<stop ) j++;
    while (mycompareTo(x[j],median) <= 0.0 && j<stop ) j++;
    
    /*
    ** list has been split, now do a recursive call
    **   always recur on the shorter list, as this keeps the total
    **       depth of nested calls to less than log_base2(n).
    */
    if ((i-start) < (stop-j)) { /* top list is shorter */
      if ((i-start)>0) mysort(start,i, x, cvec, sortUp);
	start =j; 
	}

    else {    /* bottom list is shorter */
      if ((stop -j)>0) mysort(j,stop, x, cvec, sortUp);
	stop=i; 
	}
     }
 }


