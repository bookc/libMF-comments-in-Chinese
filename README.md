This is the libMF source files with comments in **Chinese**.

The version of libMF source files is **libmf-1.0**.

The libMF official website is
[http://www.csie.ntu.edu.tw/~cjlin/libmf](http://www.csie.ntu.edu.tw/~cjlin/libmf).

LIBMF is an open source tool for approximating an incomplete matrix using the product of two
matrices in a latent space. Matrix factorization is commonly used in collaborative filtering. Main
features of LIBMF include

In addition to the latent user and item features, we add user bias, item bias, and average terms for
better performance.
LIBMF can be parallelized in a multi-core machine. To make our package more efficient, we use SSE
instructions to accelerate the vector product operations.
For a data sets of 250M ratings, LIBMF takes less then eight minutes to converge to a reasonable
level.

