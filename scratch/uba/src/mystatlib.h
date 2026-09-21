
/* Various useful statistical subroutines */


double cdf( double x);

double normcdf(double x, double mu, double sigma);

double stdnorm_cdf(const double x);

double trapezoid(int n, const double *x, const double *y);

void getmeanvar (const double *xx, int mm,
		 double *xmeanptr, double *xvarptr);
 

double getmedian_sorted (const double *xx, int mm);
 

void makeranks(double *mrank, const double *wwdat, int ntot,
	       int *tgsize, int *ntgptr);
 

/* Convert (tiegroup,ntiegroups) array to rank-sum tiesum */

double cubic_tiesum(const int *tgsize, int ntiegroups);


char *starp (double pp);

int sortdub(const void *p, const void *q);
/* Alternative name */

int dubsort(const void *p, const void *q);

int sortint(const void *p, const void *q);


int intsort(const void *p, const void *q);

double sum_kahan(int n, const double *f);

/* Often useful for squares and cubes of complicated expressions */

double square(double x);

double cube(double x);

int compareUp(double a, double b);

int compareDn(double a, double b);

void mysort(int start, int stop, double *x, int *cvec, int sortUp);


