#ifndef _NETALIGNKERNEL_H_
#define _NETALIGNKERNEL_H_

#include<iostream>
#include<cmath>
#include<cfloat>
#include<cstdlib>
#include"sparse.h"
#include"defs.h"
#include"coloringAndMatchingKernels.h"
#include <RcppArmadillo.h>

using namespace std;

//#define MKL 1
#define CHUNK 1000
#define DYNSCHED 1

CRS_Mat* extractCRSMatrix(graph* G);
CRS_Mat* createSquareMatrix(graph* A, graph* B, graph* L);
CRS_Mat* createSquareMatrix(CRS_Mat* A, CRS_Mat* B, graph* L);

double cdot(int size, double* x, double* y);
void caxpy(int row, double alpha,
        double* x, double* y);
 void cmatVec(int row, int col,
          double* vc,int* jc, int* ic, double* v, double* res);

double intmatch(int n, int m, 
        int nedges, int *v1, int *v2, double *weight, 
        int *mi);
double exact_match(double* ind, double* li, double* lj, double* w, graph* G);

double bipartite_match(double* ind, int* Mate,
        double* ws, omp_lock_t* nlocks, double* w, double* wperm, graph* G);
double bipartite_match(double* ind, int* Mate, int* HeaviestPointer, 
        int *Q1, int *Q2, double* w, double* wperm, graph* G);
double bipartite_match(double* ind, int* Mate, int* HeaviestPointer, 
        double* w, double* wperm, graph* G);
double bipartite_match(double* ind, int* Mate, int* HeaviestPointer, double* li, double* lj, double* w, graph* G);


double evaluate_overlap(double *x, CRS_Mat *S, double *temp);
double evaluate_weight(int n, double *x, double *w);
double evaluate_objective(double alpha, double beta, 
            double *x, CRS_Mat *S, double *w, double *temp);

int* build_perm(CRS_Mat* S);
void copy(int l, double* s, double* t);

struct netalign_parameters;
double* netAlignMR(CRS_Mat* S, Vec w, graph* L, Vec li, Vec lj, 
    double* wperm, netalign_parameters opts, double* objective);
double* netAlignMP(CRS_Mat* S, Vec w, graph* L, Vec li, Vec lj, 
    double* wperm, netalign_parameters opts, double* objective);
double* netAlignMPTasked(CRS_Mat* S, Vec w, graph* L, Vec li, Vec lj, 
    double* wperm, netalign_parameters opts, double* objective);
void netAlign(int argc, char** params);

arma::sp_mat netAlign_arma(arma::sp_mat A, arma::sp_mat B, arma::sp_mat L);

#include <getopt.h>

struct netalign_parameters {
    
    const char *problemname;
    double alpha;
    double beta;
    double gamma;
    int maxiter;
    const char *alg;
    int dampingtype;
    int batchrounding;
    bool verbose;
    bool finalize;
    bool quiet; 
    bool approx;
    bool limitthreads;
    int chunk;
    const char *outFile;
    
    netalign_parameters();   
    void usage();    
    bool parse(int argc, char *argv[]);
};

#endif
