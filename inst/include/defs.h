#ifndef _DEFS_H
#define _DEFS_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define MilanRealMax HUGE_VAL       // +INFINITY
#define MilanRealMin -MilanRealMax  // -INFINITY

typedef struct /* the edge data structure */
{ 
  int id;
  int head;
  int tail;
  //int weight;
  double weight;
} edge;

typedef struct /* the graph data structure */
{ 
  int numVertices;        /* Number of columns                                */
  int sVertices;          /* Number of rows: Bipartite graph: number of S vertices; T = N - S */
  int numEdges;           /* Each edge stored twice, but counted once         */
  int  * edgeListPtrs;    /* start vertex of edge, sorted, primary key        */
  edge * edgeList;        /* end   vertex of edge, sorted, secondary key      */
  int  * marks;		  /* array for marking/coloring of vertices	      */
} graph;

///COMPRESSED SPARSE COLUMN FORMAT: (edges stored only once)
typedef struct 
{ 
  int nRows;      /* Number of rows    */
  int nCols;      /* Number of columns */
  int nNNZ;       /* Number of nonzeros -- Each edge stored only once       */
  int *RowPtr;    /* Row pointer        */
  int *RowInd;	  /* Row index          */
  double *Weights;/* Edge weights       */
} matrix_CSC;


///COMPRESSED SPARSE ROW FORMAT: (edges stored only once)
typedef struct 
{ 
  int nRows;      /* Number of rows    */
  int nCols;      /* Number of columns */
  int nNNZ;       /* Number of nonzeros -- Each edge stored only once       */
  int *ColPtr;    /* Col pointer        */
  int *ColInd;	  /* Col index          */
  double *Weights;/* Edge weights       */
} matrix_CSR;

/* Utility functions */
/*void   prand(int howMany, double *randList); //Pseudorandom number generator (serial)
void   intializeCsrFromCsc(matrix_CSC*, matrix_CSR*);

int  removeEdges(int, int, edge *, int);
void SortEdgesUndirected(int, int, edge *, edge *, int *);
void SortNodeEdgesByIndex(int, edge *, edge *, int *);
void SortNodeEdgesByWeight(int, edge *, edge *, int *);
void writeGraphInMetisFormat(graph *, char *);
void displayGraphCharacterists(graph *);
void displayMatrixCsc(matrix_CSC *X);
void displayMatrixCsr(matrix_CSR *Y);
void displayMatrixProperties(matrix_CSC *X);


void sortEdgesMatrixCsc(matrix_CSC *X);
void sortEdgesMatrixCsr(matrix_CSR *Y);*/

static inline double timer() {
  return omp_get_wtime();
}

void parse_MatrixMarket(graph * G, char *fileName);
void parse_MatrixMarket_CSC(matrix_CSC * M, char *fileName);
void parse_Simple_CSC(matrix_CSC * M, char *fileName);
void parse_YVector(int *Y, int sizeY, char *fileName);
void create_graph(graph * G, double sSize, double tSize, double wSize, double* sArray, double* tArray, double* wArray);
void parse_STW(double* s, double*t, double* w, char *fileName);
void update_weight(graph* G, double* li, double* lj, double* w);
void get_params(int* NS, int* NT, int* NE, char* filename);


#endif
