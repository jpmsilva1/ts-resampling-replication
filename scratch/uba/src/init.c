#include <R.h>
#include <Rinternals.h>
#include "R_ext/Rdynload.h"

#include "ubaS.h"
#include "util.h"
#include "utilproto.h"


/*
Registiring native routines. 
By native routine, we mean an entry point in compiled code.
For each function you want to register, you need a definition in the cMethods 
array. 
The first element of each struct is a string by which you can call 
the function from R. 
The second is a pointer to the function (DL_FUNC is a cast to convert the 
pointer into a different type internally). 
The third is the number of arguments the function has, and the fourth (optional)
is the array of data types.
*/

// Functions to be invoked with .C
static const R_CMethodDef CEntries[] = {
  {"r2phi", (DL_FUNC) &r2phi, 6},
  {"r2phi_init", (DL_FUNC) &r2phi_init, 1},
  {"r2phi_eval", (DL_FUNC) &r2phi_eval, 5},
  {"r2jphi_eval", (DL_FUNC) &r2jphi_eval, 5},
  {"r2bumps_info", (DL_FUNC) &r2bumps_info, 4},
  {"r2util", (DL_FUNC) &r2util, 8},
  {"r2util_init", (DL_FUNC) &r2util_init, 3},
  {"r2util_eval", (DL_FUNC) &r2util_eval, 5},
  {"r2util_pn", (DL_FUNC) &r2util_pn, 12},
  {"r2util_pnvalues", (DL_FUNC) &r2util_pnvalues, 11},
  {"r2util_curve", (DL_FUNC) &r2util_curve, 12},
  {"r2util_curvevalues", (DL_FUNC) &r2util_curvevalues, 9},
  {"r2util_rankscores", (DL_FUNC) &r2util_rankscores, 9},
  {NULL, NULL, 0}
};

// initialization of the package
void R_init_uba(DllInfo *dll)
{
  R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
