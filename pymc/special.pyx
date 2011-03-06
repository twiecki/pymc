cdef extern from "math.h":
	double log(double x)      
    
cdef double pi = 3.14159265
cdef double inf = 1.7976931348623157e308

cdef extern from "math.h":
	double log(double x)      

cdef double gln_coeff[6]
gln_coeff[0] = 76.18009173
gln_coeff[0] =-86.50532033
gln_coeff[0] =24.01409822
gln_coeff[0] =-1.231739516
gln_coeff[0] =0.00120858003
gln_coeff[0] = -.00000536382

cdef double gammaln(double xx):
	"""Return the logarithm of the gamma function
	Corresponds to scipy.special.gammaln"""
	
	
	cdef int i
	cdef double x,ser,tmp
	
	x = xx
	tmp = x + 5.5
	tmp -= (x+0.5) * log(tmp)
	ser = 1.000000000190015
	
	for i in range(6):
		x = x+1
		ser = ser + gln_coeff[i]/x
	
	return -tmp + log(2.50662827465*ser/xx)

cdef double psi(double x):

	"""taken from 
	Bernardo, J. M. (1976). Algorithm AS 103: Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317. 
	http://www.uv.es/~bernardo/1976AppStatist.pdf """
	
	cdef double y, R, psi
	cdef double S  = 1.0e-5
	cdef double C = 8.5
	cdef double S3 = 8.333333333e-2
	cdef double S4 = 8.333333333e-3
	cdef double S5 = 3.968253968e-3
	cdef double D1 = -0.5772156649     

	y = x
	
	if y <= 0.0:
		return psi 
	    
	if y <= S :
		return D1 - 1.0/y
	
	while y < C:
		psi -= 1.0 / y
		y = y + 1
	
	R = 1.0 / y
	psi += log(y) - .5 * R 
	R= R*R
	psi -= R * (S3 - R * (S4 - R * S5))
	
	return psi

a_n = 100
cdef double factln_a[100]
#Initialize the table to negative values. 
for i in range(a_n):
	factln_a[i] = -1.

cdef double factln(int n) :
	"""gammln Returns ln(n!). """
	
	cdef pass_val = n + 1
	if n < 0:
		return -inf
	
	if n < 99:
		if (factln_a[n] < 0.) :
			factln_a[n] = gammaln(pass_val) 
		return factln_a[n]
	else :
		return gammaln(pass_val) 