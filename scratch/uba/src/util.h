/** 

** The utility realted structures and functions prototypes.
**   This helps the ansi compiler do tight checking.
Rita P. Ribeiro

**/

#include "pchip.h"
#include "pava.h"


#ifndef FLOAT
#define FLOAT float   
#endif

#ifdef MAINHT
#define EXTERN
#else
#define EXTERN extern
#endif

#define IS_ZERO(a) fabs(a) < 0.000001 // double data type problems
#define CHECK_NORM(a,b) ((b) < 0.000001 ? 1.0 : (a)/(b))


// typedef enum {false, true} bool;

typedef enum {extremes, range} phimethod;

typedef enum {base, norm, rank} utype;

typedef enum {MU, NMU, AUCROC, AUCPR, MAP11, BFM, P, R, Fm} utilmetric;

#define DELTA 0.00001 // a value to avoid the null tradeoff of P and R 

// this struct should be improved
typedef struct phi_out {
  double y_phi;
  double yd_phi;
  double ydd_phi;
} phi_out;

// a piecewise cubic Hermite interpolant polynomial (self-contained)
typedef struct {
  phimethod method;
  hermiteSpl *H; 
  phi_out  (*phi_value)();
} phi_fun;

typedef struct {
  int n;
  double *bleft;//x axis of left local min
  double *bmax;//x axis of local max  
  double *bloss;//x axis of local max  
} phi_bumps; 

typedef struct util_fun {
  utilmetric umetric_name;  
  utype umetric_type;
  bool use_util; // we might want to use as a version no costs
  double p;
  double Bmax;
  double event_thr;
  double score_thr; // by default is 0.5
  bool binorm_est;  
  int ipts; // nr of interpolation points to use in binorm est
  double beta; // for Fm
  double min_tpr;
  double max_fpr;
  bool maxprec; // for the PR curve;
  bool calibrate; // calibrate the scores
  double ustep; // for interpolation as the maximum utility gap
  double (*umetric)();
} util_fun;


static phi_fun *phiF;
static phi_bumps *bumpI;
static util_fun *utilF;


