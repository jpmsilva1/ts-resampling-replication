
/* utility.c May 2010 */
/*
** Calculate the utility of a set of prediction
**
** Rita P. Ribeiro 
*/

#include <stdio.h>
#include <math.h>

#include "ubaS.h" // ALLOC
#include "util.h"
#include "utilproto.h"


/************************************************************/
/*                                                          */
/*        INTERFACE FUNCTIONS WITH R                        */
/*                                                          */
/************************************************************/

/* ============================================================ */
// new_util
// interface function with R
/* ============================================================ */
void r2util(Sint *n, 
	    double *y,  double *ypred, 
	    double *phiF_args, 
	    double *loss_args,
	    double *utilF_args,
	    Sint *return_uv, double *u) {
  
  r2util_init(phiF_args, loss_args, utilF_args);

  r2util_eval(n, y, ypred, return_uv, u);

}

/* ============================================================ */
// set_util
// interface function with R
/* ============================================================ */
void r2util_init(double *phiF_args, 
		 double *loss_args,
		 double *utilF_args) {

  phiF = phi_init(phiF_args);  
 
  bumpI = bumps_set(phiF->H, loss_args);
  
  utilF = util_init(utilF_args);
}

/* ============================================================ */
// r2bumps_info
// interface function with R
/* ============================================================ */
void r2bumps_info(Sint *n, 
		  double *y,  
		  double *phiF_args, 
		  double *loss_args) {

  int i;
 
  phiF = phi_init(phiF_args);  
  
  bumpI = bumps_set(phiF->H, loss_args);

  printf("B->n: %d\n",bumpI->n);
  
  for(i=0; i < bumpI->n; i++) 
    printf("Bump %d Bleft %f Bmax %f Bloss %f\n", i, 
	   bumpI->bleft[i],bumpI->bmax[i],bumpI->bloss[i]);
	   
  

}


/* ============================================================ */
// eval_util
// interface function with R
/* ============================================================ */
void r2util_eval(Sint *n, 
		 double *y,  double *ypred, 
		 Sint *return_uv, double *u) {

  int i;
  double *wt, um;
  phi_out *y_phiF, *ypred_phiF;

  if((y_phiF = (phi_out *)ALLOC((int) *n, sizeof(phi_out))) == NULL) exit(EXIT_FAILURE);
  if((ypred_phiF = (phi_out *)ALLOC((int) *n, sizeof(phi_out))) == NULL) exit(EXIT_FAILURE);
  

  util_core((int) *n, y, ypred, y_phiF, ypred_phiF, u);

 	   
  if((bool) *return_uv == false) { // estimate u metric
    
    if((wt = (double *)ALLOC((int) *n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
    for(i=0;i<(int) *n;i++) wt[i] = 1.0;
      
    um = (*utilF->umetric)((int) *n, u, wt, y_phiF, ypred_phiF, utilF);
   
    //free(wt); wt = NULL;

    *u = um;
  }

  // cannot free this!
  //free(y_phiF); y_phiF = NULL;
  //free(ypred_phiF); ypred_phiF = NULL;

}

/* ============================================================ */
// Util-based PN values
// interface function with R
/* ============================================================ */
void r2util_pn(Sint *n,  
	       double *y,  double *ypred, 
	       double *phiF_args, 
	       double *loss_args,
	       double *utilF_args,		 
	       double *cutoffs, 
	       double *tp, double *fp, 
	       double *util_pos, double *util_neg, int *pn_size) {

  
  r2util_init(phiF_args, loss_args, utilF_args);

  r2util_pnvalues(n, y, ypred, 
		  cutoffs, tp, fp, 
		  util_pos, util_neg, pn_size);

}

/* ============================================================ */
// Util-based PN values of an already initialized utility setup
// 
/* ============================================================ */
void r2util_pnvalues(Sint *n,  
		     double *y,  double *ypred, 
		     double *cutoffs, 
		     double *tp, double *fp, 
		     double *util_pos, double *util_neg, int *pn_size) {
  
  int i;
  phi_out *y_phiF, *ypred_phiF;
  double *wt, *u;

  if((wt = (double *)ALLOC((int) *n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((u = (double *)ALLOC((int) *n, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  if((y_phiF = (phi_out *)ALLOC((int) *n, sizeof(phi_out))) == NULL) exit(EXIT_FAILURE);
  if((ypred_phiF = (phi_out *)ALLOC((int) *n, sizeof(phi_out))) == NULL) exit(EXIT_FAILURE);
  
  
  util_core((int) *n, y, ypred, y_phiF, ypred_phiF, u);
  

  for(i=0;i<(int) *n;i++) wt[i] = 1.0;
    
  
  pn_values((int) *n, u, wt, y_phiF, ypred_phiF, utilF,
	    cutoffs, tp, fp, (int) *n, // re-check this
	    util_pos, util_neg, pn_size);

  /* free(wt); wt = NULL; */
  /* free(u); u = NULL; */
  /* free(y_phiF); y_phiF = NULL; */
  /* free(ypred_phiF); ypred_phiF = NULL; */

}



/* ============================================================ */
// Util-based Curve
// interface function with R
/* ============================================================ */
void r2util_curve(Sint *get_roc, Sint *get_hull,
		  Sint *n,  
		  double *y,  double *ypred, 
		  double *phiF_args, 
		  double *loss_args,
		  double *utilF_args,			  
		  double *x_values, double *y_values, double *alpha_values,
		  int *xy_size) {

  
  r2util_init(phiF_args, loss_args, utilF_args);

  r2util_curvevalues(get_roc, get_hull,
		     n, y, ypred, 
		     x_values, y_values, alpha_values, xy_size);
  
}

/* ============================================================ */
// Util-based Curve of previously initialized utility setup
// interface function with R
/* ============================================================ */
void r2util_curvevalues(Sint *get_roc, Sint *get_hull,
			Sint *n,  
			double *y,  double *ypred, 
			double *x_values, double *y_values, double *alpha_values, 
			int *xy_size) {

  int i;
  phi_out *y_phiF, *ypred_phiF;
  double *wt, *u;

  if((wt = (double *)ALLOC((int) *n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
  if((u = (double *)ALLOC((int) *n, sizeof(double))) == NULL) exit(EXIT_FAILURE);

  if((y_phiF = (phi_out *)ALLOC((int) *n, sizeof(phi_out))) == NULL) exit(EXIT_FAILURE);
  if((ypred_phiF = (phi_out *)ALLOC((int) *n, sizeof(phi_out))) == NULL) exit(EXIT_FAILURE);

  util_core((int) *n, y, ypred, y_phiF, ypred_phiF, u);

 
  for(i=0;i<(int) *n;i++) wt[i] = 1.0;
  
  compute_curve((bool) *get_roc, (bool) *get_hull, 
		(int) *n, 
		u, wt, y_phiF, ypred_phiF, utilF,
		x_values, y_values, alpha_values, xy_size);

  /* free(wt); wt = NULL; */
  /* free(u); u = NULL; */
  /* free(y_phiF); y_phiF = NULL; */
  /* free(ypred_phiF); ypred_phiF = NULL; */
    
}

/* ============================================================ */
// Util-based Curve
// interface function with R
/* ============================================================ */

void r2util_rankscores(Sint *n,  
		       double *y,  double *ypred, 
		       double *phiF_args, 
		       double *loss_args,
		       double *utilF_args,			  
		       double *u,  double *scores, int *idx) {

 
  int i;
  phi_out *y_phiF, *ypred_phiF;
  
  double *wt;

  if((wt = (double *)ALLOC((int) *n, sizeof(double))) == NULL) exit(EXIT_FAILURE);
 
  r2util_init(phiF_args, loss_args, utilF_args);

  if((y_phiF = (phi_out *)ALLOC((int) *n, sizeof(phi_out))) == NULL) exit(EXIT_FAILURE);
  if((ypred_phiF = (phi_out *)ALLOC((int) *n, sizeof(phi_out))) == NULL) exit(EXIT_FAILURE);
  
  util_core((int) *n, y, ypred, y_phiF, ypred_phiF, u);
  for(i=0;i<(int) *n;i++) wt[i] = 1.0;
    
  // a patch ... because of stock market 
  utilF -> calibrate = 1;
  

  get_RankScores((int) *n,y_phiF,ypred_phiF,u,wt,utilF,scores,idx);
  
  /* free(wt); wt = NULL; */
  /* free(y_phiF); y_phiF = NULL; */
  /* free(ypred_phiF); ypred_phiF = NULL; */
}



/*
  ----------------------------------------------------------- 
  Init Util 
  -----------------------------------------------------------
*/ 
util_fun *util_init(double *utilF_args) {

  util_fun *utilF;

  if((utilF = (util_fun *)ALLOC(1, sizeof(util_fun))) == NULL) exit(EXIT_FAILURE);
   
  utilF->umetric_name = (utilmetric) utilF_args[0];  
  utilF->umetric_type = (utype) utilF_args[1];  
  utilF->use_util = (bool) utilF_args[2];

  utilF->p = utilF_args[3];
  utilF->Bmax = utilF_args[4];  
  
  utilF->event_thr = utilF_args[5];
  utilF->score_thr = utilF_args[6];

  utilF->binorm_est = (bool) utilF_args[7];
  utilF->ipts = (int) utilF_args[8];
  utilF->beta = utilF_args[9];
  utilF->min_tpr = utilF_args[10];
  utilF->max_fpr = utilF_args[11];
  
  utilF->maxprec = (bool) utilF_args[12];
  utilF->calibrate = (bool) utilF_args[13];
  utilF->ustep = utilF_args[14];
  

  switch(utilF->umetric_name) {
  case MU:  
    utilF->umetric = mu;
    break;
  case NMU:  
    utilF->umetric = mu;
    break;
  case P:  
    utilF->umetric = precision;
    break;
  case R:  
    utilF->umetric = recall;
    break;
  case Fm:  
    utilF->umetric = fm;
    break;
  case AUCROC:  
    utilF->umetric = aucroc;
    break;
  case AUCPR:  
    utilF->umetric = aucpr;
    break;
  case MAP11:  
    utilF->umetric = map11;
    break;
  case BFM:  
    utilF->umetric = bfm;
    break;
  }
  

  return utilF;
}

/* ============================================================ */
// util core function
// 
/* ============================================================ */
void util_core(int n, double *y,  double *ypred,
	       phi_out *y_phiF, phi_out *ypred_phiF,
	       double *u) {
  int i;

  for(i = 0; i < n; i++) {
    
    y_phiF[i] = (*phiF->phi_value)(y[i], phiF);
    ypred_phiF[i] = (*phiF->phi_value)(ypred[i], phiF);

    u[i] = 
	util_value(y[i], ypred[i], y_phiF[i], ypred_phiF[i], 
		   phiF, bumpI, utilF);
      
  }

}


/* ============================================================ */
// util core function
// 
/* ============================================================ */
double util_value(double y, double ypred,
		  phi_out y_phiF, phi_out ypred_phiF,
		  phi_fun *phiF, phi_bumps *bumpI,
		  util_fun *utilF) {

  double lb, lc, ycphi, l;
  double jphi, benef, cost, uv;
  
  benefcost_lin(y, ypred,
		ypred_phiF.y_phi,phiF, bumpI,
		&lb, &lc, &ycphi);


  l = fabs(y - ypred);


  if(lb == 0 || l > lb) 
    benef = 0;
  else
    benef = (1 - l/lb);
  /* ---------------------------------------------------------------- */
          
  jphi = jphi_value(y_phiF.y_phi,ycphi,utilF->p);// + 1e-7;

  if(lc == 0 || l > lc)
    cost = 1;
  else
    cost = l/lc;


  /* ---------------------------------------------------------------- */
  
  uv = y_phiF.y_phi * benef - jphi * cost;

  // July, 2014
  //uv = (y_phiF.y_phi * benef + 1e-7) - jphi * cost;

  /*
  // STILL NOT EFFICIENT
  switch(utilF->umetric_type) {
    //case base:  
    //uv = utilF->Bmax * uv;
  case norm:
    uv = (uv + 1) / 2; 
  case rank:
    uv = fabs(uv);
  }
  */

  if(utilF->umetric_type == norm)
    //     || utilF->umetric_type == rank)
    uv = (uv + 1) / 2;
  
  return uv;
}


//------------------------------------------------
// Benefits Linearization
void benefcost_lin(double y, double ypred,
		   double ypred_phi,
		   phi_fun *phiF, phi_bumps *bumpI,
		   double *lb, double *lc, double *ycphi) {

  double lossA, yc;
  int i = 1, rightmost_closed = 0, all_inside = 0, mfl = 0;
 
  if(bumpI->n > 1)    {
    i = findInterval(bumpI->bleft,bumpI->n,
		     y,
		     rightmost_closed,all_inside,i,&mfl); 

  }
  if(i > 0) i--; // Re-check this ...

  lossA = INFINITY;
  /*--------------------------------------------------*/
  /* Benefits loss tolerance */

  if(ypred <= y) {

    if(i > 0 && isfinite(bumpI->bleft[i]))
      lossA = fabs(y - bumpI->bleft[i]);  
  
  } else {

    if((i+1) < bumpI-> n && isfinite(bumpI->bleft[i+1]))
      lossA = fabs(y - bumpI->bleft[i+1]);
   
  }
  

  if(lossA < bumpI->bloss[i])
    *lb = lossA;
  else
    *lb = bumpI->bloss[i];



  lossA = INFINITY;
  /*--------------------------------------------------*/
  /* Costs loss tolerance */
 
  if(ypred <= y) {

    if(i > 0 && isfinite(bumpI->bmax[i-1]))
      lossA = fabs(y - bumpI->bmax[i-1]);

 
  } else {

    if(i+1 < bumpI-> n && isfinite(bumpI->bmax[i+1]))
      lossA = fabs(y - bumpI->bmax[i+1]);
   
    
  }

  // If the err of committing regarding an action is more serious...
  if(lossA < bumpI->bloss[i])
    *lc = lossA;
  else
    *lc = bumpI->bloss[i];

  

  if(ypred <= y) 
    yc = y - *lc;
  else
    yc = y + *lc;
     
  *ycphi = ypred_phi;
    
}

