cdef double gammaln(double) nogil
cdef double psi(double ) nogil
cdef double factln(int ) nogil

cdef extern from "math.h":
    double log(double x)  nogil    