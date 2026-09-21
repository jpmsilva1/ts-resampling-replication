/* u_metrics.c May 2010 */
/*
** Calculate an utility measure for a given partition.
**
All the metrics imply the Kahan summation algorithm. 

** Rita P. Ribeiro 
*/

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "mystatlib.h"

#include "ubaS.h" // ALLOC
#include "util.h"
#include "utilproto.h"


/* ----------------------------------------------------------- 
  MU and NMU
  -----------------------------------------------------------
*/  
double mu(int n,  double *u, double *wt,
	  phi_out *y_phiF, phi_out *ypred_phiF,
	  util_fun *utilF) {

  int i;
  double s_num, s_den;
  double *num, *den;

  if((num = (double *)ALLOC(n, sizeof(double))) == NULL)  exit(EXIT_FAILURE);
  if((den = (double *)ALLOC(n, sizeof(double))) == NULL)  exit(EXIT_FAILURE);
      
  // is not a simple average, because of weights

  for(i = 0; i< n; i++) {
    num[i] = u[i] * wt[i];
    den[i] = wt[i];
  }

  s_den = sum_kahan(n,den);

  assert(s_den > 0.0); /* to ensure no division by zero */

  s_num = sum_kahan(n,num);

  /* free(num); num = NULL; */
  /* free(den); den = NULL; */


  return s_num/s_den;
}


/*
  ----------------------------------------------------------- 
  Normalized Mean Utility
  -----------------------------------------------------------
*/  

double norm_mu(double su, double wts, double Bmax) {

  double mu = (double) ((su + wts*Bmax)/(2*wts*Bmax));
  
  
  if(isnan(mu)) {
    printf("NAN alert: mu %f su %f wts %f ",mu,su,wts);
    mu = 0.5;
  }
  return mu;
}

/*
  ----------------------------------------------------------- 
  Precision
    // Neste caso e a mesma coisa pq a previsão foi sempre a mesma!!!
    // os sinais derivados forma os mesmos.
    // Aqui na precision a unica duvida q persiste é se sao usados 
    // todos os sianis positivos ou nenhum.. depende se a previsao 
    // e um evento ou nao.

  -----------------------------------------------------------
*/  
double precision(int n, double *u, double *wt,
		 phi_out *y_phiF, phi_out *ypred_phiF,		 
		 util_fun *utilF) {

  int i, n_num = 0, n_den = 0;
  double s_num, s_den, u_temp;
  double *num, *den;
  
  if((num = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((den = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  for(i = 0; i < n; i++) {
   
      // check if it is a positive prediction
      
      if(ypred_phiF[i].y_phi >= utilF->event_thr) {
	if(utilF->use_util) 
	  u_temp = fabs(1 + u[i]);	  
	else 
	  u_temp = 1.0;

	num[n_num++] = u_temp * wt[i];      
       
	if(utilF->use_util) 
	  u_temp = fabs(1 + ypred_phiF[i].y_phi);
	else 
	  u_temp = 1.0;

	den[n_den++] = u_temp * wt[i];
	
      } 
  }
  

  if(n_num == 0)
    return DELTA;
  
  s_den = sum_kahan(n_den,den);

  assert(s_den > 0.0); /* to ensure no division by zero */

  s_num = sum_kahan(n_num,num);
  
  return s_num/s_den;  
}


/*
  ----------------------------------------------------------- 
  Recall
  Notice that this function is 99% similar to Precision.
  It is only defined for semantic purposes.
  -----------------------------------------------------------
*/  
double recall(int n, double *u,
	      double *wt,
	      phi_out *y_phiF, phi_out *ypred_phiF,
	      util_fun *utilF) {

  int i, n_num = 0, n_den = 0;
  double s_num, s_den, u_temp;
  double *num, *den;

  if((num = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((den = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  for(i = 0; i < n; i++) {
 
    // check if it is positive
    if(y_phiF[i].y_phi >= utilF->event_thr) { 
      
      if(utilF->use_util) 
	u_temp = fabs(1 + y_phiF[i].y_phi);
      else
	u_temp = 1.0;

      den[n_den++] = u_temp * wt[i];
                      
      if(utilF->use_util) 
	u_temp = fabs(1 + u[i]);
      else
	u_temp = 1.0;
	
      num[n_num++] = u_temp * wt[i];
      
    }     
  }     
  
  if(n_num == 0)
    return DELTA;
  
  s_den = sum_kahan(n_den,den);

  assert(s_den > 0.0); /* to ensure no division by zero */

  s_num = sum_kahan(n_num,num);

  return s_num/s_den;  
}


/*
double precision(int n, double *u, double *wt,
		 phi_out *y_phiF, phi_out *ypred_phiF,		 
		 util_fun *utilF) {

  int i, n_num = 0, n_den = 0;
  double s_num, s_den, u_temp;
  double *num, *den, *scores;
  int *idx;

  if((num = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((den = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((scores = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((idx = (int *)ALLOC(n, sizeof(int))) == NULL) exit(EXIT_FAILURE);


  get_RankScores(n,y_phiF,ypred_phiF,u,wt,utilF,scores,idx);

  for(i = 0; i < n; i++) {

    if(scores[i] > (1 - u[idx[i]])/2) {
      
      // check if it is tp
      
      if(y_phiF[idx[i]].y_phi >= utilF->event_thr) {
	if(utilF->use_util) 
	  u_temp = fabs(1 + u[idx[i]]);	  
	else 
	  u_temp = 1.0;

	num[n_num++] = u_temp * wt[idx[i]];      
       
	if(utilF->use_util) 
	  u_temp = fabs(1 + y_phiF[idx[i]].y_phi);
	else 
	  u_temp = 1.0;

	den[n_den++] = u_temp * wt[idx[i]];
	
      } else { // it is a fp

	if(utilF->use_util) 
	  u_temp = fabs(2 - utilF->p * (1 - y_phiF[idx[i]].y_phi));
	else
	  u_temp = 1.0;

	den[n_den++] = u_temp * wt[idx[i]];
      }            
    }
  }

  if(n_num == 0)
    return DELTA;

  s_den = sum_kahan(n_den,den);

  assert(s_den > 0.0); // to ensure no division by zero 

  s_num = sum_kahan(n_num,num);

  // cannot free these
  //free(num); num = NULL;
  //free(den); den = NULL;
  //free(scores); scores = NULL;
  //free(idx); idx = NULL;

  return s_num/s_den;  
}



double recall(int n, double *u,
	      double *wt,
	      phi_out *y_phiF, phi_out *ypred_phiF,
	      util_fun *utilF) {

  int i, n_num = 0, n_den = 0;
  double s_num, s_den, u_temp;
  double *num, *den, *scores;
  int *idx;

  if((num = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((den = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((scores = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((idx = (int *)ALLOC(n, sizeof(int))) == NULL) exit(EXIT_FAILURE);

  get_RankScores(n,y_phiF,ypred_phiF,u,wt,utilF,scores,idx);

  for(i = 0; i < n; i++) {
 
    // check if it is pos
    if(y_phiF[idx[i]].y_phi >= utilF->event_thr) { 
      
      if(utilF->use_util) 
	u_temp = fabs(1 + y_phiF[idx[i]].y_phi);
      else
	u_temp = 1.0;

      den[n_den++] = u_temp * wt[idx[i]];
                
      if(scores[i] > (1 - u[idx[i]])/2) {
	if(utilF->use_util) 
	  u_temp = fabs(1 + u[idx[i]]);
	else
	  u_temp = 1.0;
	
	num[n_num++] = u_temp * wt[idx[i]];
	
      }     
    }     
  }

  

  if(n_num == 0)
    return DELTA;

  s_den = sum_kahan(n_den,den);

  assert(s_den > 0.0); // to ensure no division by zero 

  s_num = sum_kahan(n_num,num);

  // cannot free these
  //free(num); num = NULL;
  //free(den); den = NULL;
  //free(scores); scores = NULL;
  //free(idx); idx = NULL;

  return s_num/s_den;  
}

*/

/*
  ----------------------------------------------------------- 
  F-measure
  -----------------------------------------------------------
*/  
double fm(int n, double *u,
	  double *wt,
	  phi_out *y_phiF, phi_out *ypred_phiF,
	  util_fun *utilF) {
  
  double prec, rec, m;
 
  
  // less efficient
  prec = precision(n, u, wt, y_phiF, ypred_phiF, utilF);
  rec = recall(n, u, wt, y_phiF, ypred_phiF, utilF);

  m = harmonic_mean(prec,rec,utilF->beta);

  return m;
}

/*
  ----------------------------------------------------------- 
  harmonic mean used by Fm
  -----------------------------------------------------------
*/  
double harmonic_mean(double x, double y, double beta) {
  
  double m;

  if(fabs(x) == 0 || fabs(y) == 0)
    m = 0;
  else
    m = ((pow(beta,2) + 1) * x * y)/
      (pow(beta,2) * x + y);
  

  return m;
}


/*
  ----------------------------------------------------------- 
  AUC Metrics - Area Under Curve
  As a sum of the area of successive trapezoids = (b1 + b2)/2 * h
  It can also be obtained by the area of the bottom rectangle plus 
  area of top triangle under the curve.
  In the particular case of ROC, we inspect how much it decreases 
  from 0.5. Hence we inspect row wise instead of column wise. 
  -----------------------------------------------------------
*/  

/*
  ----------------------------------------------------------- 
  aucroc 
  
  -----------------------------------------------------------
*/  
double aucroc(int n, double *u,
	      double *wt,
	      phi_out *y_phiF, phi_out *ypred_phiF,
	      util_fun *utilF) {

  int i_last, i, npts, pn_size;
  double util_pos, util_neg, m = 0.0, 
    cur_tpr, cur_fpr, tpr, fpr, rowrectangle, extrtriangle,
    slope, last_tpr, fp_max;
  double *cutoffs, *tp, *fp;//, *tn, *fn;
  
  npts = max_curve_size(n,utilF);
  
  if((cutoffs = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((tp = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((fp = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  
  pn_values(n, u, wt, y_phiF, ypred_phiF, utilF,
	    cutoffs, tp, fp, npts, 
	    &util_pos, &util_neg, &pn_size);
  

  if(util_pos == 0 || util_neg == 0)
    return m; // possible experimental setting (e.g. Monte Carlo)
  
  fp_max = utilF->max_fpr * util_neg;

  // pn_size should be at least 2
  i_last = pn_size-1; 

  while(i_last > 1 && fp[i_last] > fp_max) i_last--;

  cur_tpr = tp[0] / util_pos;
  cur_fpr = fp[0] / util_neg;
  
  m = 0.5 * cur_fpr * cur_tpr;

  for(i = 1; i <= i_last; i++) {
    tpr = tp[i] / util_pos;
    fpr = fp[i] / util_neg;
    
    rowrectangle = fpr * (tpr - cur_tpr);
    extrtriangle = 0.5 * (fpr - cur_fpr) * (tpr - cur_tpr);
  
    m += (rowrectangle - extrtriangle);
    
    cur_tpr = tpr;
    cur_fpr = fpr;
  }


  if(i_last < pn_size-1) { // interpolate forward
    
    tpr = tp[i_last+1] / util_pos;
    fpr = fp[i_last+1] / util_neg;
    

    slope = (tpr - cur_tpr) / (fpr - cur_fpr);
     
    last_tpr = cur_tpr + slope * (fp[i_last + 1] - fp_max)/util_neg;
 

    rowrectangle = utilF->max_fpr * (last_tpr - cur_tpr);
    extrtriangle = 0.5 * (utilF->max_fpr - cur_fpr) * (last_tpr - cur_tpr);

    m += (rowrectangle - extrtriangle);
  }


  m = 1.0 - m;

  /* free(cutoffs); cutoffs = NULL; */
  /* free(tp); tp = NULL; */
  /* free(fp); fp = NULL; */

  return m;
  
}

/*
  ----------------------------------------------------------- 
  aucpr

  -----------------------------------------------------------
*/  


double aucpr(int n, double *u,
	    double *wt,
	    phi_out *y_phiF, phi_out *ypred_phiF,
	    util_fun *utilF) {

  
  int i_start = 0, j, npts, pn_size;
  double util_pos, util_neg, m = 0.0, 
    tp_min, rec, prec, cur_rec, cur_prec, 
    first_rec, first_prec, slope, colrectangle, extrtriangle;
  double *cutoffs, *tp, *fp; //*tn, *fn;

  npts = max_curve_size(n,utilF);
  
  if((cutoffs = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((tp = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((fp = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  // can be empirical or binorm...
  pn_values(n, u, wt, y_phiF, ypred_phiF, utilF,
	    cutoffs, tp, fp, npts, 
	    &util_pos, &util_neg, &pn_size);

  if(util_pos == 0 || util_neg == 0)
    return m; // possible experimental setting (e.g. Monte Carlo)

  tp_min = utilF->min_tpr * util_pos;

  while(i_start < pn_size-2 && tp[i_start] < tp_min) i_start++;
      
  rec = tp[i_start] / util_pos;
  prec = tp[i_start] / (tp[i_start] + fp[i_start]);
  
  first_rec = (tp[i_start] - tp_min) / util_pos;
  colrectangle = first_rec * prec;

  m = colrectangle;
  
  if(i_start > 0) { // interpolate backwards
    
    cur_rec = (tp[i_start-1] / util_pos);
    cur_prec = tp[i_start-1] / (tp[i_start-1] + fp[i_start-1]);
 
    slope = (prec - cur_prec) / (rec - cur_rec);
     
    first_prec = cur_prec + slope * (tp_min - tp[i_start-1]) / util_pos;
 
    extrtriangle = 0.5 * first_rec * (first_prec - prec);        

    m += extrtriangle;
  }
  
  cur_rec = rec;
  cur_prec = prec;
 
  for (j = i_start + 1; j < pn_size; j++) {

    rec = tp[j] / util_pos;
    prec = tp[j] / (tp[j] + fp[j]);
   
    colrectangle = (rec - cur_rec) * prec;
    extrtriangle = 0.5 * (rec - cur_rec) * (cur_prec - prec);
    
    m += (colrectangle + extrtriangle);

    cur_rec = rec;
    cur_prec = prec;

  }

  // 
  // free(cutoffs); cutoffs = NULL;
  // free(tp); tp = NULL;
  // free(fp); fp = NULL;


  return m;
}

/*
  ----------------------------------------------------------- 
  MAP11 - Mean Average Precision
  ----
  Our PR curve already estimates the precision values at 
  these (0, 0.1, ..., 0.9, 1) fixed recall points.
  -----------------------------------------------------------
*/  
double map11(int n, double *u,
	     double *wt,
	     phi_out *y_phiF, phi_out *ypred_phiF,
	     util_fun *utilF) {

  double m = 0, rec;
  int i, j, xy_size;
  double *x_values, *y_values, *alpha_values;
  
  if((x_values = (double *)ALLOC(utilF->ipts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((y_values = (double *)ALLOC(utilF->ipts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((alpha_values = (double *)ALLOC(utilF->ipts, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  compute_curve(false, false,
		n, 
		u, wt, y_phiF, ypred_phiF, utilF,
		x_values, y_values, alpha_values, &xy_size);

  j = 0;
  if(xy_size > 0) { // EVENTUAL EXPERIMENTAL SETTING
    for(i = 1; i <= 10; i++) {
      rec = (double) (i/10.0);
      while(x_values[j] < rec && j < xy_size) j++;
      m += y_values[j-1];
    }
  }


  /* free(x_values); x_values = NULL; */
  /* free(y_values); y_values = NULL; */
  /* free(alpha_values); alpha_values = NULL; */

  return m/10;
}

/*
  ----------------------------------------------------------- 
  BFM - Best F-measure
  -----------------------------------------------------------
*/  
double bfm(int n, double *u,
	   double *wt,
	   phi_out *y_phiF, phi_out *ypred_phiF,
	   util_fun *utilF) {

  double m, bm = 0.0;
  int i, xy_size;  
  double *x_values, *y_values, *alpha_values;
  
  if((x_values = (double *)ALLOC(utilF->ipts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((y_values = (double *)ALLOC(utilF->ipts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((alpha_values = (double *)ALLOC(utilF->ipts, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  compute_curve(false, false,
		n, 
		u, wt, y_phiF, ypred_phiF, utilF,
		x_values, y_values, alpha_values, &xy_size);
  
  if(xy_size > 0) { // EVENTUAL EXPERIMENTAL SETTING
    for(i = 0; i < utilF->ipts; i++) {   
      m = harmonic_mean(y_values[i],x_values[i],utilF->beta);
      if(m > bm) bm = m;
    }
  }

  /* free(x_values); x_values = NULL; */
  /* free(y_values); y_values = NULL; */
  /* free(alpha_values); alpha_values = NULL; */


  return bm;
}

/*
  ----------------------------------------------------------- 
  Generates curve values
  -----------------------------------------------------------
*/  


void compute_curve(bool get_roc, bool convex_hull, 
		   int n,
		   double *u, double *wt,
		   phi_out *y_phiF, phi_out *ypred_phiF, util_fun *utilF,		   
		   double *x_values, double *y_values, double *alpha_values,
		   int *xy_size) {


  int pn_size, npts;
  double *cutoffs, *tp, *fp;   
  double util_pos, util_neg;


  npts = max_curve_size(n,utilF);

  if((cutoffs = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((tp = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((fp = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  pn_values(n, u, wt, y_phiF, ypred_phiF, utilF,
	    cutoffs, tp, fp, npts, 
	    &util_pos, &util_neg, &pn_size);


  if(fabs(util_pos) > 0 && fabs(util_neg) > 0) {  

    if(get_roc)
      compute_roc(pn_size, cutoffs, tp, fp, util_pos, util_neg, 
		  x_values, y_values, alpha_values, xy_size);
    else
      compute_pr(pn_size,cutoffs, tp, fp, util_pos, util_neg, 
		 utilF->maxprec,
		 x_values, y_values, alpha_values, xy_size);
    

  } else {

    *xy_size = 0;
    x_values[0] = '\0';
    y_values[0] = '\0';
    alpha_values[0] = '\0';
  }
  
  /* free(cutoffs); cutoffs = NULL; */
  /* free(tp); tp = NULL; */
  /* free(fp); fp = NULL; */
  
}

/*
  ----------------------------------------------------------- 
  Estimate maximum number of points for the curves 
  -----------------------------------------------------------
*/  
int max_curve_size(int n, util_fun *utilF) {
 
  int npts;

  if(utilF->binorm_est)
    npts = utilF->ipts;
  else
    npts = n;
  
  
  if(utilF->ustep > 0)
    npts = npts / utilF->ustep;

    
  if(npts <= n) npts = n * 2; // safety
    
  return npts;
}
/*
  ----------------------------------------------------------- 
  Retrieve pn values 
  Empirical or binormal 
  -----------------------------------------------------------
*/  
void pn_values(int n, double *u,
	       double *wt,
	       phi_out *y_phiF, phi_out *ypred_phiF,
	       util_fun *utilF,	       	       
	       double *cutoffs, double *tp, double *fp, int npts, 
	       double *util_pos, double *util_neg, int *pn_size) {  
  

  double *cutoffs_ord, *tp_ord, *fp_ord;
  double *cutoffs_emp, *tp_emp, *fp_emp;
  double *cutoffs_hull, *tp_hull, *fp_hull;

  int hull_size;
  
  if((cutoffs_emp = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((tp_emp = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((fp_emp = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  pn_empirical(n, u, wt, y_phiF, ypred_phiF, utilF,
	       util_pos, util_neg, pn_size,
	       cutoffs_emp, tp_emp, fp_emp); 

  if(!*util_pos) return;
   
  if((cutoffs_ord = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((tp_ord = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((fp_ord = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  // sort

  pn_sort(pn_size, cutoffs_emp, tp_emp, fp_emp, cutoffs_ord, tp_ord, fp_ord);
 
  if((cutoffs_hull = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((tp_hull = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((fp_hull = (double *)ALLOC(npts, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  pn_hull(*pn_size, cutoffs_ord, tp_ord, fp_ord,
	  *util_pos, *util_neg,
	  cutoffs_hull, tp_hull, fp_hull, &hull_size);

  *pn_size = hull_size;

  
  pn_interpolate(pn_size, *util_pos, *util_neg,
		 utilF->ustep,
		 cutoffs_hull, tp_hull, fp_hull,  
		 cutoffs, tp, fp);

  /* free(cutoffs_emp); cutoffs_emp = NULL; */
  /* free(tp_emp); tp_emp = NULL; */
  /* free(fp_emp); fp_emp = NULL; */
  
  /* free(cutoffs_ord); cutoffs_ord = NULL; */
  /* free(tp_ord); tp_ord = NULL; */
  /* free(fp_ord); fp_ord = NULL; */

  /* free(cutoffs_hull); cutoffs_hull = NULL; */
  /* free(tp_hull); tp_hull = NULL; */
  /* free(fp_hull); fp_hull = NULL; */

} 

/*
  ----------------------------------------------------------- 
  Sort pn points
  -----------------------------------------------------------
*/  
void pn_sort(int *pn_size, double *cutoffs, double *tp, double *fp,	    
	     double *cutoffs_ord, double *tp_ord, double *fp_ord) {

  int i, j=0, i_start;
  int *idx; 
  double *tp_temp, *fp_temp;
  double min_tp = 0.01, new_tp, new_fp;

  if((idx = (int *)ALLOC(*pn_size, sizeof(int))) == NULL) exit(EXIT_FAILURE);
  if((tp_temp = (double *)ALLOC(*pn_size, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((fp_temp = (double *)ALLOC(*pn_size, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  
  for(i = 0; i < *pn_size; i++) {
    idx[i] = i;
    tp_temp[i] = tp[i];    
 }
  
  // order by positives (and then by negatives)
  mysort(0, *pn_size-1, tp_temp, idx, true);

  for(i = 0; i < *pn_size; i++) 
    fp_temp[i] = fp[idx[i]];    
  

  i_start=0;
    
  
  while(fabs(tp_temp[i_start]) == 0) i_start++;

  // uniformize with the theoretical 
  if(tp_temp[i_start] >= min_tp && fp_temp[i_start] > 0) { // the first was a fp

    new_tp = min_tp;
    new_fp = min_tp *(fp_temp[i_start] / tp_temp[i_start]);

    if(!contains(*pn_size,tp_temp,fp_temp,
		 new_tp,new_fp)) {      
      tp_ord[j] = new_tp;
      fp_ord[j] = new_fp;
      cutoffs_ord[j] = cutoffs[idx[i_start]];
      j++;
    }
  }

  
  for(i = i_start; i < *pn_size; i++) {
    tp_ord[j] = tp[idx[i]];
    fp_ord[j] = fp[idx[i]];
    cutoffs_ord[j] = cutoffs[idx[i]];
    j++;
  }

  /* free(idx); idx = NULL; */
  /* free(tp_temp); tp_temp = NULL; */
  /* free(fp_temp); fp_temp = NULL; */

  *pn_size = j;
}

// assuming order pn space
// very naive... may change to bin_search
bool contains(int pn_size, double *tp, double *fp,
	      double tp_point, double fp_point) {
  int i;
  bool look_tp = true, look_fp = false;

  for(i = 0; i < pn_size; i++) {
    if(look_tp) {
      if(tp[i] > tp_point)
	return false;      
      if(tp[i] == tp_point) {
	look_tp = false;
	look_fp = true;
      }
    }
    if(look_fp) {
      if(tp[i] != tp_point)
	return false;
      if(fp[i] == fp_point)
	return true;	  
    }
  }
  return false;
}

/*
  ----------------------------------------------------------- 
  From pn prediction values to curve values
  -----------------------------------------------------------
*/  

void compute_roc(int npts, 	      
		 double *cutoffs, 
		 double *tp, double *fp, 
		 double util_pos, double util_neg,		 
		 double *x_values, double *y_values, double *alpha_values,
		 int *xy_size) {
  
  int i;
 

  x_values[0] = 0.0;
  y_values[0] = 0.0;
  alpha_values[0] = INFINITY;

  for(i = 1;i < npts; i++) {
    x_values[i] = fp[i]/util_neg;
    y_values[i] = tp[i]/util_pos;
    alpha_values[i] = cutoffs[i];    
    
  }
  
  
  x_values[npts] = 1.0;
  y_values[npts] = 1.0;
  alpha_values[npts] = 0.0;
  

  x_values[npts+1] = '\0';
  y_values[npts+1] = '\0';
  alpha_values[npts+1] = '\0';

  *xy_size = npts + 1;
}

/*
  ----------------------------------------------------------- 
  Empirical pn values
  -----------------------------------------------------------
*/  

void compute_pr(int npts, 	      
		double *cutoffs, 
		double *tp, double *fp, 
		double util_pos, double util_neg, bool maxprec,		 
		double *x_values, double *y_values, double *alpha_values,
		int *xy_size) {

  double tpr, tpr_i, slope, new_tp, new_fp;
  int i = 0, j = 0, k, k_max = 1000;
  

  for(k = 1; k < k_max; k++) {    

    tpr = (double) k/k_max;

    tpr_i = tp[i]/util_pos;
     
    if(tpr <= tpr_i) {

      if(i == 0) {

	x_values[j] = tpr;
	y_values[j] = tp[i]/(tp[i] + fp[i]);
	alpha_values[j] = cutoffs[i];
	
        j++;

	if(isnan(y_values[j-1])) {
	  exit(1);
	}


      } else {
        
        slope = (fp[i] - fp[i-1])/(tp[i] - tp[i-1]);
        
        new_tp = tpr * util_pos;
        new_fp = fp[i-1] + slope * (new_tp - tp[i-1]);

	x_values[j] = tpr;
	y_values[j] = new_tp/(new_tp + new_fp);
	alpha_values[j] = cutoffs[i-1];             

        j++;

	if(isnan(y_values[j-1])) {
	  exit(1);
	}

	
      }                  
    } else { // tpr > tpr_i

      while(i < npts && tpr > tpr_i) {
        i++;
        tpr_i = tp[i]/util_pos;	  
      }
      
      slope = (fp[i] - fp[i-1])/(tp[i] - tp[i-1]);
      
      new_tp = tpr * util_pos;
      new_fp = fp[i-1] + slope * (new_tp - tp[i-1]);
      
      x_values[j] = tpr;
      y_values[j] = new_tp/(new_tp + new_fp);
      alpha_values[j] = cutoffs[i-1];
      j++;

      if(isnan(y_values[j-1])) {
	exit(1);
      }

    }    
    
  }

  
    x_values[j] = 1.0;
    y_values[j] = util_pos/(util_pos+util_neg);
    alpha_values[j] = 0.0;
    j++;
  

    x_values[j] = '\0';
    y_values[j] = '\0';
    alpha_values[j] = '\0';
    
    *xy_size = j;
    
    
  if(maxprec)
    compute_maxprec(*xy_size, y_values);

}

/*
  ----------------------------------------------------------- 
  Calculate the cumulative maximum precision values
  -----------------------------------------------------------
*/  
void compute_maxprec(int y_size, double *y_values) {

  int i;
  double y_max = -INFINITY;  

  for(i = y_size-1; i >= 0; i--) {
    if(y_max < y_values[i]) y_max = y_values[i];    
    y_values[i] = y_max;
  }

}

/*
  ----------------------------------------------------------- 
  Empirical pn values
  -----------------------------------------------------------
*/  

void get_RankScores(int n, 
		    phi_out *y_phiF, phi_out *ypred_phiF,
		    double *u, double *wt,
		    util_fun *utilF,
		    double *scores, int *scores_order) {
  		    
  int i, np = n;
  double *scores_orig, *scores_u;

  if((scores_orig = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE); 
  if((scores_u = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  for(i = 0; i < n; i++) {
    scores_order[i] = i;   
    scores_u[i] = u[i]; 

    if(y_phiF[i].y_phi >= utilF->event_thr)
      scores_orig[i] = 1.0;
    else
      scores_orig[i] = 0.0;

  }

  mysort(0, n-1, scores_u, scores_order,true);

  
  if(utilF->calibrate) {
    
    for(i = 0; i < n; i++) 
      scores[i] = scores_orig[scores_order[i]];
    
    upava(scores, &np);
    
  } else {
  
    for(i = 0; i < n; i++) 
      scores[i] = scores_u[i];
    
  }
  
  // cannot free these
  //free(scores_u); scores_u = NULL;
  //free(scores_orig); scores_orig = NULL;

}


/*
  ----------------------------------------------------------- 
  Empirical pn values
  -----------------------------------------------------------
*/  
void pn_empirical(int n, double *u,
		  double *wt,
		  phi_out *y_phiF, phi_out *ypred_phiF,
		  util_fun *utilF,	 
		  double *util_pos, double *util_neg, int *pn_size,
		  double *cutoffs, double *tp, double *fp) {
  

  int i=0, j=0;
  int *idx;
  double *scores;
  double tp_util = 0, fp_util = 0;
  double prev_score = INFINITY;

  if((idx = (int *)ALLOC(n, sizeof(int))) == NULL) exit(EXIT_FAILURE);
  if((scores = (double *)ALLOC(n, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  *util_pos = 0;
  *util_neg = 0;
  
  get_RankScores(n,y_phiF,ypred_phiF,u,wt,utilF,scores,idx);


  for(i = 0; i < n; i++) {
    if(y_phiF[idx[i]].y_phi >= utilF->event_thr) {
      
      if(utilF->use_util) 
	*util_pos += (1 + y_phiF[idx[i]].y_phi);
      else
	*util_pos += 1;


    } else {
      if(utilF->use_util)
	*util_neg += (2 - utilF->p * (1 - y_phiF[idx[i]].y_phi));	
      else
	*util_neg +=1;    

    }
    
  }
  
  for(i = n-1; i >= 0; i--) {
       
    if(prev_score != scores[i]) {
      cutoffs[j] = scores[i];
      tp[j] = tp_util;
      fp[j] = fp_util;
      prev_score = scores[i];
      
      j++;
    }
    
    if(y_phiF[idx[i]].y_phi >= utilF->event_thr) 
      if(utilF->use_util) {
	tp_util += (1 + u[idx[i]]); 	
      } else {
	tp_util++;
      }
    else
      if(utilF->use_util) {
	fp_util += (1 - u[idx[i]]); 
      } else {
	fp_util++;
      }
  }
  
  cutoffs[j] = scores[i-1];
  tp[j] = tp_util;
  fp[j] = fp_util;

  j++;

  *pn_size = j;
  
  cutoffs[*pn_size] = '\0';
  tp[*pn_size] = '\0';
  fp[*pn_size] = '\0';

  /* free(idx); idx = NULL; */
  /* free(scores); scores = NULL; */
 
}

/*
  ----------------------------------------------------------- 
  Interpolate pn values to a minimum utility step gap
  -----------------------------------------------------------
*/  
void pn_interpolate(int *pn_size, double util_pos, double util_neg, 
		    double ustep, 
		    double *cutoffs_ord, double *tp_ord, double *fp_ord, 
		    double *cutoffs, double *tp, double *fp) {

  int i, j=0;
  double slope, cur_tp, cur_fp;

  for(i = 0; i < (*pn_size) - 1; i++) {
       
    if(tp_ord[i] == tp_ord[i+1]) continue;

    tp[j] = tp_ord[i];
    fp[j] = fp_ord[i];
    cutoffs[j++] = cutoffs_ord[i];

    slope = (fp_ord[i+1] - fp_ord[i]) / 
      (tp_ord[i+1] - tp_ord[i]);
    
    cur_tp = tp_ord[i];

    if(!isfinite(slope))
      exit(1);

    while(fabs(tp_ord[i+1] - cur_tp) > ustep) {
      
      cur_fp = fp_ord[i] + slope * (cur_tp - tp_ord[i] + ustep);      
      cur_tp += ustep; 

      tp[j] = cur_tp;
      fp[j] = cur_fp;
      cutoffs[j++] = cutoffs_ord[i];      
    }    
  }
  
  if(fabs(tp[j-1] - tp_ord[(*pn_size) - 1]) > 0.0) {
    tp[j] = tp_ord[(*pn_size) - 1];
    fp[j] = fp_ord[(*pn_size) - 1];
    cutoffs[j++] = cutoffs_ord[(*pn_size) - 1];
  }

  *pn_size = j;
  
}


bool is_acc(double uv) {
  if (get_sign(uv) == 1.0)
    return true;
  else
    return false;
}

double get_sign(double uv) {
  if (uv > 0.0) 
    return 1.0;
  else
    return -1.0;
}

/*
  ----------------------------------------------------------- 
  pn_hull
  -----------------------------------------------------------
*/  

/*
Adapted from
##=========================================================================
## build.hull
##-------------------------------------------------------------------------
##
## From the PN space, it builds the Convex Hull and the Achievable PR Curve
## 
## Finds the achievable PR Curve as described in:
## Jesse Davis & Mark Goadrich,
## “The Relationship Between Precision­Recall and ROC Curves”, ICML 2006. 
##
## i) first, find the ROC convex hull; 
## ii) for each point to be included in the hull,
## use the confusion matrix to construct it
## in PR space;
## iii) use PR space interpolation to draw the
## achievable PR curve between the newly created PR points;
## iv) do not forget to interpolate precision.
##
## > 
##
## Coded by: Rita Ribeiro
##=========================================================================
*/

void pn_hull(int npts, 	      
	     double *cutoffs, double *tp, double *fp, 
	     double util_pos, double util_neg, 
	     double *cutoffs_hull, double *tp_hull, double *fp_hull, 
	     int *hull_size) {

  int i, j, orientation;
  double tpr, fpr, tpr1, fpr1, tpr2, fpr2;

  cutoffs_hull[0] = cutoffs[0];
  tp_hull[0] = tp[0];
  fp_hull[0] = fp[0];

  cutoffs_hull[1] = cutoffs[npts-1];
  tp_hull[1] = util_pos;
  fp_hull[1] = util_neg;

  *hull_size = 2;

  // For each point in the pn space
  for(i = 0; i < (npts-1); i++) {


    // if the point is in the curve
    if(contains(*hull_size,tp_hull,fp_hull,
		tp[i],fp[i])) 
      continue;


    tpr = tp[i] / util_pos;
    fpr = fp[i] / util_neg;


    // update the hull
    j = 0;

    
    while(j < *hull_size-1) {

      tpr1 = tp_hull[j] / util_pos;
      fpr1 = fp_hull[j] / util_neg;
      tpr2 = tp_hull[j+1] / util_pos;
      fpr2 = fp_hull[j+1] / util_neg;
            

    //In our X scan from 0 to 1, we know that p can't be on the
    //hull as soon as we exceed X.

      if(fpr < fpr1) break; // next pn point

      // fpr >= fpr1 
      
      orientation = find_orientation(fpr1, tpr1,
				     fpr, tpr,
				     fpr2, tpr2);
      
      
      if(orientation == 0 && // on the hull -> collinear
	 between(fp_hull[j]/util_neg, tp_hull[j]/util_pos, 
		 fpr, tpr,
		 fp_hull[j+1]/util_neg, tp_hull[j+1]/util_pos)
	 ) { 
	
	/*
	memcpy(tp_hull+j+2,tp_hull+j+1,(npts-j)*sizeof(double)); //shift right
	memcpy(fp_hull+j+2,fp_hull+j+1,(npts-j)*sizeof(double)); //shift right
	memcpy(cutoffs_hull+j+2,cutoffs_hull+j+1,(npts-j)*sizeof(double)); //shift right
	*/

	memmove(tp_hull+j+2,tp_hull+j+1,(npts-j)*sizeof(double)); //shift right
	memmove(fp_hull+j+2,fp_hull+j+1,(npts-j)*sizeof(double)); //shift right
	memmove(cutoffs_hull+j+2,cutoffs_hull+j+1,(npts-j)*sizeof(double)); //shift right

	tp_hull[j+1] = tp[i];
	fp_hull[j+1] = fp[i];
	cutoffs_hull[j+1] = cutoffs[i];
	


        break; //next pn point
        
      } else if(orientation == -1) { //left
        
		
        // Connects i'th hull point to p
      
	/*
	memcpy(tp_hull+j+2,tp_hull+j+1,(npts-j)*sizeof(double)); //shift right
	memcpy(fp_hull+j+2,fp_hull+j+1,(npts-j)*sizeof(double)); //shift right
	memcpy(cutoffs_hull+j+2,cutoffs_hull+j+1,(npts-j)*sizeof(double)); //shift right
	*/

	memmove(tp_hull+j+2,tp_hull+j+1,(npts-j)*sizeof(double)); //shift right
	memmove(fp_hull+j+2,fp_hull+j+1,(npts-j)*sizeof(double)); //shift right
	memmove(cutoffs_hull+j+2,cutoffs_hull+j+1,(npts-j)*sizeof(double)); //shift right

	tp_hull[j+1] = tp[i];
	fp_hull[j+1] = fp[i];
	cutoffs_hull[j+1] = cutoffs[i];

        j++;
        (*hull_size)++;


	/*
        ##  p now becomes the i'th point.  Return if 
        ##  p is the next-to-last point.
	*/     
          
	/*
        ##  Now determine whether any hull points after p are under
        ##  the new hull.      
	*/

	
        while(j < (*hull_size - 1) &&
              find_orientation(fpr, tpr,
			       fp_hull[j+1]/util_neg,tp_hull[j+1]/util_pos,
			       fp_hull[j+2]/util_neg,tp_hull[j+2]/util_pos
			       ) 
	      == 1) {

	  /*
	  memcpy(tp_hull+j,tp_hull+j+1,(npts-j)*sizeof(double)); //shift left
	  memcpy(fp_hull+j,fp_hull+j+1,(npts-j)*sizeof(double)); //shift left
	  memcpy(cutoffs_hull+j,cutoffs_hull+j+1,(npts-j)*sizeof(double)); //shift left
          */

	  memmove(tp_hull+j,tp_hull+j+1,(npts-j)*sizeof(double)); //shift left
	  memmove(fp_hull+j,fp_hull+j+1,(npts-j)*sizeof(double)); //shift left
	  memmove(cutoffs_hull+j,cutoffs_hull+j+1,(npts-j)*sizeof(double)); //shift left

          //pn does not have (0,0) at the beginning, thus ... 
	  (*hull_size)--;	

        }
        
        break; //next pn point
      }
      j++;
  
    } // hull loop
  
  } // pn points loop

  
  tp_hull[*hull_size] = '\0';
  fp_hull[*hull_size] = '\0';
  cutoffs_hull[*hull_size] = '\0';

}

/* ----------------------------------------------------------
   find_orientation

## It uses the determinant of the orientation matrix for 
## local points p1 - p - p2 for a clockwise polygon.
## Thus, 
## if det < 0 then the seq of points is convex 
##          => orientation = right (1);
## if det > 0 then the seq of points is concave 
##          => orientation = left (-1);
## else the seq of points is collinear
##          => orientation = on (0); 
---------------------------------------------------------- */
int find_orientation(double x1, double y1, 
		     double x, double y,
		     double x2, double y2) {
		       
  double det;

  det = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1);

  if(det < 0)
    return 1;
  else if(det > 0)
    return -1;
  else
    return 0;
      
}

/*
##=========================================================================
## between
##-------------------------------------------------------------------------
## Determine whether p is on the line segment joining 
##  p1 to p2.  When this is called, we already know that p0, p1 and p2 are
##  co-linear, so just compare the co-ordinate ranges.
##=========================================================================
*/
bool between(double x1, double y1,
	     double x, double y,
	     double x2, double y2) {

  int resX = 0, resY = 0;


  if(x2 > x1) {
    if(x1 <= x && x <= x2) 
      resX = 1;
  } else {
    if(x2 <= x && x <= x1) 
      resX = 1;
  }

  if(y2 > y1) {
    if(y1 <= y && y <= y2) 
      resY = 1;
  } else {
    if(y2 <= y && y <= y1) 
      resY = 1;
  }
    
  return (bool) (resX * resY);
}
