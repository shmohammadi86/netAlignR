#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;

#define ARMA_USE_CXX11_RNG
#define DYNSCHED

arma::sp_mat netAlign_arma(arma::sp_mat A_mat, arma::sp_mat B_mat, arma::sp_mat L_mat,
							double alpha = 1.0,
							double beta = 1.0,
							double gamma = 0.99,
							int maxiter = 100,
							bool finalize = false);

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::sp_mat netAlign(arma::sp_mat A, arma::sp_mat B, arma::sp_mat L,
							double alpha = 1.0,
							double beta = 1.0,
							double gamma = 0.99,
							int maxiter = 100,
							bool finalize = false) {

	arma::sp_mat L_matched = netAlign_arma(A, B, L, alpha, beta, gamma, maxiter, finalize);
	
	return(L_matched);
}
