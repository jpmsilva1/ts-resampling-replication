/** 

** The utility realted structures and functions prototypes.
**   This helps the ansi compiler do tight checking.
Rita P. Ribeiro

**/

/* --------------------------------------------------------- */
/* Phi Function */
/* --------------------------------------------------------- */

EXTERN void r2phi(Sint *n, double *y,  
		  double *phiF_args,
		  double *y_phi, double *yd_phi, double *ydd_phi);

EXTERN void r2phi_init(double *phiF_args);

EXTERN void r2phi_eval(Sint *n, double *y,  
		       double *y_phi, double *yd_phi, double *ydd_phi);		       
		       

EXTERN void r2jphi_eval(Sint *n, double *y_phi, double *ypred_phi, double *p,
			double *jphi);

EXTERN double jphi_value(double y_phi, double ypred_phi, double p);

EXTERN phi_fun *phi_init(double *phiF_args);

EXTERN hermiteSpl *phiSpl_init(double *phiF_args);

EXTERN phi_out phiSpl_value(double y, phi_fun *phiF);

/* --------------------------------------------------------- */
/* Bumps Info*/
/* --------------------------------------------------------- */

EXTERN phi_bumps *bumps_set(hermiteSpl *H, double *loss_args); 

EXTERN void r2bumps_info(Sint *n, 
			 double *y,  
			 double *phiF_args, 
			 double *loss_args);
/* --------------------------------------------------------- */
/* Util Function */
/* --------------------------------------------------------- */


EXTERN void r2util(Sint *n, 
		   double *y,  double *ypred,
		   double *phiF_args, 
		   double *loss_args, 
		   double *utilF_args,
		   Sint *return_uv, double *u);

EXTERN void r2util_init(double *phiF_args, 
			double *loss_args, 
			double *utilF_args);

EXTERN void r2util_eval(Sint *n, 
			double *y,  double *ypred,		     
			Sint *return_uv, double *u);

EXTERN void r2util_pn(Sint *n,  
		      double *y,  double *ypred, 
		      double *phiF_args, 
		      double *loss_args,
		      double *utilF_args,		 
		      double *cutoffs, double *tp, double *fp, 
		      double *util_pos, double *util_neg, int *pn_size);


EXTERN void r2util_pnvalues(Sint *n,  
			    double *y,  double *ypred, 			   	 
			    double *cutoffs, double *tp, double *fp, 
			    double *util_pos, double *util_neg, int *pn_size);




EXTERN void r2util_curve(Sint *get_roc, Sint *get_hull, 
			 Sint *n,  
			 double *y,  double *ypred, 
			 double *phiF_args, 
			 double *loss_args,
			 double *utilF_args,			  
			 double *x_values, double *y_values, double *alpha_values, 
			 int *xy_size);


EXTERN void r2util_curvevalues(Sint *get_roc, Sint *get_hull, 
			       Sint *n,  
			       double *y,  double *ypred, 
			       double *x_values, double *y_values, double *alpha_values,
			       int *xy_size);



EXTERN void r2util_rankscores(Sint *n,  
			      double *y,  double *ypred, 
			      double *phiF_args, 
			      double *loss_args,
			      double *utilF_args,			  
			      double *u,  double *scores, int *idx);
/********************************************************/


EXTERN util_fun *util_init(double *utilF_args);

EXTERN void util_core(int n, double *y,  double *ypred,
		      phi_out *y_phiF, phi_out *ypred_phiF,
		      double *u);

EXTERN double util_value(double y, double ypred,
			 phi_out y_phiF, phi_out ypred_phiF,
			 phi_fun *phiF, phi_bumps *bumpI, util_fun *utilF);


EXTERN void benefcost_lin(double y, double ypred,
			  double ypred_phi,
			  phi_fun *phiF, phi_bumps *bumpI,
			  double*lb, double*lc, double *ycphi);


EXTERN void benefcost_lin_old(double y, double ypred,
			  double ypred_phi,
			  phi_fun *phiF, phi_bumps *bumpI,
			  double*yb, double*yc, double *ycphi);
EXTERN double norm_mu(double su, double wts, double Bmax);


EXTERN double mu(int n, double *u,
		 double *wt,
		 phi_out *y_phiF, phi_out *ypred_phiF,
		 util_fun *utilF);

EXTERN double mnu(int n, double *u,
		  double *wt,
		  phi_out *y_phiF, phi_out *ypred_phiF,
		  util_fun *utilF);

EXTERN double precision(int n, double *u,
			double *wt,
			phi_out *y_phiF, phi_out *ypred_phiF,
			util_fun *utilF);

EXTERN double recall(int n, double *u,
		     double *wt,
		     phi_out *y_phiF, phi_out *ypred_phiF,
		     util_fun *utilF);

EXTERN double fm(int n, double *u,
		 double *wt,
		 phi_out *y_phiF, phi_out *ypred_phiF,
		 util_fun *utilF);

EXTERN double aucroc(int n, double *u,
		     double *wt,
		     phi_out *y_phiF, phi_out *ypred_phiF,
		     util_fun *utilF);

EXTERN double aucpr(int n, double *u,
		   double *wt,
		   phi_out *y_phiF, phi_out *ypred_phiF,
		   util_fun *utilF);

EXTERN double map11(int n, double *u,
		    double *wt,
		    phi_out *y_phiF, phi_out *ypred_phiF,
		    util_fun *utilF);

EXTERN double bfm(int n, double *u,
		  double *wt,
		  phi_out *y_phiF, phi_out *ypred_phiF,
		  util_fun *utilF);

EXTERN double harmonic_mean(double precision, double recall, double beta);


double mySigmoid(double x, double s);

void compute_curve(bool get_roc, bool get_hull, 
		   int n,
		   double *u,
		   double *wt,
		   phi_out *y_phiF, phi_out *ypred_phiF,
		   util_fun *utilF,		   
		   double *x_values, double *y_values, double *alpha_values,
		   int *xy_size);

void compute_roc(int npts, 	      
		 double *cutoffs, 
		 double *tp, double *fp, 
		 double util_pos, double util_neg,		 
		 double *x_values, double *y_values, double *alpha_values,
		 int *xy_size);

void compute_pr(int npts, 	      
		double *cutoffs, 
		double *tp, double *fp, 
		double util_pos, double util_neg, bool maxprec,		 
		double *x_values, double *y_values, double *alpha_values,
		int *xy_size);

void compute_pr1(int npts, 	      
		 double *cutoffs, 
		 double *tp, double *fp, 
		 double util_pos, double util_neg, bool maxprec,		 
		 double *x_values, double *y_values, double *alpha_values,
		 int *xy_size);

void compute_maxprec(int y_size, double *y_values);

void pn_values(int n, double *u,
	       double *wt,
	       phi_out *y_phiF, phi_out *ypred_phiF,
	       util_fun *utilF,	       	       
	       double *cutoffs, double *tp, double *fp, int npts, //?
	       double *util_pos, double *util_neg, int *pn_size);


void pn_empirical(int n, double *u,
		  double *wt,
		  phi_out *y_phiF, phi_out *ypred_phiF,
		  util_fun *utilF,	       	       
		  double *util_pos, double *util_neg, int *pn_size,
		  double *cutoffs, double *tp, double *fp);
		  

void pn_binormal(int n, double *u,
		 double *wt,
		 phi_out *y_phiF, phi_out *ypred_phiF,
		 util_fun *utilF,	       	       
		 double *cutoffs, double *tp, double *fp, 
		 double *util_pos, double *util_neg, int *pn_size);


void binormal_est(int n, double *u,
		  double *wt,
		  phi_out *y_phiF, phi_out *ypred_phiF,
		  util_fun *utilF,
		  double *alpha, 
		  double *mean_pos, double *std_pos,
		  double *mean_neg, double *std_neg,
		  double *util_pos, double *util_neg);


void costbenef_matrix(double alpha,
		      double mean_pos, double std_pos, 
		      double mean_neg, double std_neg,		 
		      double util_pos, double util_neg, 
		      double *scores,
		      int npts,		 
		      double *cutoffs,double *tp, double *fp);


void pn_sort(int *pn_size, 
	     double *cutoffs, double *tp, double *fp, 
	     double *cutoffs_ord, double *tp_ord, double *fp_ord); 

void pn_interpolate(int *pn_size, double util_pos, double util_neg, 
		    double ustep, 
		    double *cutoffs_ord, double *tp_ord, double *fp_ord, 
		    double *cutoffs, double *tp, double *fp);

void pn_hull(int npts, 	      
	     double *cutoffs, double *tp, double *fp, 
	     double util_pos, double util_neg, 	     
	     double *cutoffs_hull, double *tp_hull, double *fp_hull, 
	     int *hull_size);

int max_curve_size(int n, util_fun *utilF);

int find_orientation(double tpr, double fpr,
		     double tprA, double fprA, 
		     double tprB, double fprB);

bool between(double tprA, double fprA,
	     double tpr, double fpr,
	     double tprB, double fprB);

bool contains(int pn_size, double *tp, double *fp,
	      double tp_point, double fp_point);


double get_sign(double uv);

bool is_acc(double uv); 


void get_RankScores(int n, phi_out *y_phiF, phi_out *ypred_phiF,
		    double *u, double *wt,
		    util_fun *utilF,
		    double *scores, int *scores_order);
