#include "defs.h"
#include<stdlib.h>
#include "netAlignKernel.h"
/* Remove self- and duplicate edges.                                */
/* For each node, we store its non-duplicate edges as a linked list */
int removeEdges2(int NV, int NE, edge *edgeList) {
  int NGE = 0;
  int *head = (int *) malloc(NV * sizeof(int));     /* head of linked list points to an edge */   
  int *next = (int *) malloc(NE * sizeof(int));     /* ptr to next edge in linked list       */
  
/* Initialize linked lists */
#pragma omp parallel for
  for (int i = 0; i < NV; i++) 
    head[i] = -1;
#pragma omp parallel for
  for (int i = 0; i < NE; i++) 
    next[i] = -2;
  
  for (int i = 0; i < NE; i++) {
    int sv  = edgeList[i].head;
    int ev  = edgeList[i].tail;
    if (sv == ev) continue;    /* self edge */  
    int * ptr = head + sv;     /* start at head of list for this key */
    while (1) {
      int edgeId = *ptr;
      if (edgeId == -1) {         /* at the end of the list */
        edgeId = *ptr;             /* lock ptr               */
        if (edgeId == -1) {       /* if still end of list   */
          int newId = NGE;
          NGE++;     /* increment number of good edges */
          edgeList[i].id = newId;                 /* set id of edge                 */    
          next[i] = -1;                           /* insert edge in linked list     */
          *ptr = i;
          break;
        }
        *ptr = edgeId;
      } else 
        if (edgeList[edgeId].tail == ev) break;     /* duplicate edge */
        else  ptr = next + edgeId;
    }
  }
  
  /* Move good edges to front of edgeList                    */
  /* While edge i is a bad edge, swap with last edge in list */
  for (int i = 0; i < NGE; i++) {
    while (next[i] == -2) {
      int k = NE - 1;
      NE--;
      edgeList[i] = edgeList[k];
      next[i] = next[k];
    }
  }
  
  free(head);
  free(next);
  return NGE;
}

/* Since graph is undirected, sort each edge head --> tail AND tail --> head */
void SortEdgesUndirected2(int NV, int NE, edge *list1, edge *list2, int *ptrs) {

#pragma omp parallel for
  for (int i = 0; i < NV + 2; i++) 
    ptrs[i] = 0;
  ptrs += 2;
  
  /* Histogram key values */
  for (int i = 0; i < NE; i++) {
    int head = list1[i].head;
    int tail = list1[i].tail;
    ptrs[head] ++;
    ptrs[tail] ++;
  }
  
  /* Compute start index of each bucket */
  for (int i = 1; i < NV; i++) 
    ptrs[i] += ptrs[i-1];
  ptrs --;
  
  /* Move edges into its bucket's segment */
  for (int i = 0; i < NE; i++) {
    int head  = list1[i].head;
    int index = ptrs[head] ++;
    list2[index].id     = list1[i].id;
    list2[index].head   = list1[i].head;
    list2[index].tail   = list1[i].tail;
    list2[index].weight = list1[i].weight;
    
    int tail   = list1[i].tail;
    index      = ptrs[tail] ++;
    list2[index].id     = list1[i].id;
    list2[index].head   = list1[i].tail;
    list2[index].tail   = list1[i].head;
    list2[index].weight = list1[i].weight;
  } 
}//End of SortEdgesUndirected2()

/* Sort each node's neighbors by tail from smallest to largest. */
void SortNodeEdgesByIndex2(int NV, edge *list1, edge *list2, int *ptrs) {
  for (int i = 0; i < NV; i++) {
    edge *edges1 = list1 + ptrs[i];
    edge *edges2 = list2 + ptrs[i];
    int size     = ptrs[i+1] - ptrs[i];    
    /* Merge Sort */
    for (int skip = 2; skip < 2 * size; skip *= 2) {
      for (int sect = 0; sect < size; sect += skip)  {
        int j = sect;
        int l = sect;
        int half_skip = skip / 2;
        int k = sect + half_skip;
        
        int j_limit = (j + half_skip < size) ? j + half_skip : size;
        int k_limit = (k + half_skip < size) ? k + half_skip : size;
        
        while ((j < j_limit) && (k < k_limit)) {
          if   (edges1[j].tail < edges1[k].tail) {edges2[l] = edges1[j]; j++; l++;}
          else                                   {edges2[l] = edges1[k]; k++; l++;}
        }
        while (j < j_limit) {edges2[l] = edges1[j]; j++; l++;}
        while (k < k_limit) {edges2[l] = edges1[k]; k++; l++;}
      }
      edge *tmp = edges1;
      edges1 = edges2;
      edges2 = tmp;
    }
    
    // result is in list2, so move to list1
    if (edges1 == list2 + ptrs[i])
      for (int j = ptrs[i]; j < ptrs[i+1]; j++) list1[j] = list2[j];
  } 
}//End of SortNodeEdgesByIndex2()


/*-------------------------------------------------------*
 * This function reads a MATRIX MARKET file and build the graph
 *-------------------------------------------------------*/
void parse_MatrixMarket(graph * G, char *fileName) {
  //printf("Parsing a Matrix Market File...\n");
/*#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int nprocs   = omp_get_num_procs();
    int tid      = omp_get_thread_num();
    //if (tid == 0)
      //printf("parse_MatrixMarket: Number of threads: %d\n Number of procs: %d\n", nthreads, nprocs);
  }*/
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  //char line[1024];
  //fgets(line, 1024, file);
  char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];    
  if (fscanf(file, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
    printf("parse_MatrixMarket(): bad file format - 01");
    exit(1);
  }
  //printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
  if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
    printf("Error: The first line should start with %%MatrixMarket word \n");
    exit(1);
  }
  if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
    printf("Error: The Object should be matrix or Matrix or MATRIX \n");
    exit(1);
  }
  if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
    printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
    exit(1);
  }
  int isComplex = 0;    
  if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
    isComplex = 1;
    printf("Warning: Will only read the real part. \n");
  }
  int isPattern = 0;
  if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
    isPattern = 1;
    printf("Note: Matrix type is Pattern. Will set all weights to 1.\n");
    //exit(1);
  }
  int isSymmetric = 0, isGeneral = 0;
  if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
    isGeneral = 1;
  else {
    if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
      isSymmetric = 1;
      printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
    }
  }     
  if ( (isGeneral==0) && (isSymmetric==0) )       {
    printf("Warning: Matrix type should be General or Symmetric. \n");
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  char line[1024];
  do {
    assert(fgets(line, 1024, file) != NULL);
  } while ( line[0] == '%' );
  
  /* Read the matrix parameters */
  int NS=0, NT=0, NV = 0, NE=0;
  if (sscanf(line, "%d %d %d",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket(): bad file format - 02");
    exit(1);
  }
  NV = NS + NT;
  //printf("|S|= %d, |T|= %d, |E|= %d \n", NS, NT, NE);

  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* S vertices: 0 to NS-1                                               */
  /* T vertices: NS to NS+NT-1                                           */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each vertex
  int  *edgeListPtr = (int *)  malloc((NV+1) * sizeof(int));
//#pragma omp parallel for
  for (int i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  edge *edgeListTmp; //Read the data in a temporary list
  int newNNZ = 0;    //New edges because of symmetric matrices
  int Si, Ti;
  double weight = 1;
  if( isSymmetric == 1 ) {
    //printf("Matrix is of type: Symmetric Real or Complex\n");
    //printf("Weights will be converted to positive numbers.\n");
    edgeListTmp = (edge *) malloc(2 * NE * sizeof(edge));
    for (int i = 0; i < NE; i++) {
      if (isPattern == 1)
        assert(fscanf(file, "%d %d", &Si, &Ti) == 2);
      else
        assert(fscanf(file, "%d %d %lf", &Si, &Ti, &weight) == 3);
      Si--; Ti--;            // One-based indexing
      //weight = fabs(weight); //Make it positive  : Leave it as is
     if ( Si == Ti ) {
        edgeListTmp[i].head = Si;       //The S index
        edgeListTmp[i].tail = NS+Ti;    //The T index 
        edgeListTmp[i].weight = weight; //The value
        edgeListPtr[Si+1]++;
        edgeListPtr[NS+Ti+1]++;
      }
      else { //an off diagonal element: Also store the upper part
        //LOWER PART:
        edgeListTmp[i].head = Si;       //The S index 
        edgeListTmp[i].tail = NS+Ti;    //The T index 
        edgeListTmp[i].weight = weight; //The value
        edgeListPtr[Si+1]++;
        edgeListPtr[NS+Ti+1]++;
        //UPPER PART:
        edgeListTmp[NE+newNNZ].head = Ti;       //The T index
        edgeListTmp[NE+newNNZ].tail = NS+Si;    //The S index
        edgeListTmp[NE+newNNZ].weight = weight; //The value
        newNNZ++; //Increment the number of edges
        edgeListPtr[Ti+1]++;
        edgeListPtr[NS+Si+1]++;
      }
    }
  } //End of Symmetric
    /////// General Real or Complex ///////
  else {
    //printf("Matrix is of type: Unsymmetric Real or Complex\n");
   //printf("Weights will be converted to positive numbers.\n");
    edgeListTmp = (edge *) malloc( NE * sizeof(edge));
    for (int i = 0; i < NE; i++) {
      if (isPattern == 1)
        assert(fscanf(file, "%d %d", &Si, &Ti) == 2);
      else
        assert(fscanf(file, "%d %d %lf", &Si, &Ti, &weight) == 3);
      //printf("(%d, %d) %lf\n",Si, Ti, weight);
      Si--; Ti--;            // One-based indexing
      //weight = fabs(weight); //Make it positive    : Leave it as is
      edgeListTmp[i].head = Si;       //The S index
      edgeListTmp[i].tail = NS+Ti;    //The T index
      edgeListTmp[i].weight = weight; //The value
      edgeListPtr[Si+1]++;
      edgeListPtr[NS+Ti+1]++;
    }
  } //End of Real or Complex
  
  fclose(file); //Close the file
  //printf("Done reading from file.\n");
  if( isSymmetric ) {
    //printf("Modified the number of edges from %d ", NE);
    NE += newNNZ; //#NNZ might change
    printf("to %d \n", NE);
  }
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = timer();
  for (int i=0; i<=NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = timer();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %d, edgeListPtr[NV]= %d\n", NE*2, edgeListPtr[NV]);
  
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  time1 = timer();
  //int  *marks    = (int *)  malloc( NV  * sizeof(int));
//#pragma omp parallel for
  //for (int i = 0; i < NV; i++)
    //marks[i] = 0;
  
  
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  time2 = timer();
  printf("Time for allocating memory for marks and edgeList = %lf\n", time2 - time1);
  
  time1 = timer();
  //Keep track of how many edges have been added for a vertex:
  printf("Oooooooooooooo\n");
  int  *added    = (int *)malloc( NV  * sizeof(int));
  if(added==NULL)
        printf("Oooooooooooooo\n");
//#pragma omp parallel for
  for (int i = 0; i < NV; i++)
    added[i] = 0;
  
  //printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
//#pragma omp parallel for
  for(int i=0; i<NE; i++) {
    int head      = edgeListTmp[i].head;
    int tail      = edgeListTmp[i].tail;
    double weight = edgeListTmp[i].weight;
    
    int Where = edgeListPtr[head] + added[head];   
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    added[head]++;
    //Now add the counter-edge:
    Where = edgeListPtr[tail] + added[tail];
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;
    added[tail]++;
  }
  time2 = timer();
  printf("Time for building edgeList = %lf\n", time2 - time1);
    
  time1 = timer();
  edge *TmpList = (edge *) malloc (2*NE * sizeof(edge));
  SortNodeEdgesByIndex2(NV, edgeList, TmpList, edgeListPtr);
  time2 = timer();
  //totTime += time2 - time1;
  printf("Time to sort edge lists by index = %lf\n", time2 - time1);  

  G->sVertices    = NS;
  G->numVertices  = NV;
  G->numEdges     = NE;
  //G->marks        = marks;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  free(TmpList);
  //free(added);
  //free(edgeListTmp);
  
}

/*-------------------------------------------------------*
 * This function reads a Matrix Market file and build a 
 * matrix in CSR format
 *-------------------------------------------------------*/
void parse_MatrixMarket_CSC(matrix_CSC * M, char *fileName) {
  printf("Parsing a Matrix Market File...\n");
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int nprocs   = omp_get_num_procs();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("Coloring Rouinte: Number of threads: %d\n Number of procs: %d\n", nthreads, nprocs);
  }
  double time1, time2;      
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  assert(fgets(line, 1024, file) != NULL);
  char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];    
  if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
    printf("parse_MatrixMarket_CSC(): bad file format - 01\n");
    exit(1);
  }
  printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
  if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
    printf("Error: The first line should start with %%MatrixMarket word \n");
    exit(1);
  }
  if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
    printf("Error: The Object should be matrix or Matrix or MATRIX \n");
    exit(1);
  }
  if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
    printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
    exit(1);
  }
  int isComplex = 0;    
  if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
    isComplex = 1;
    printf("Warning: Will only read the real part. \n");
  }
  if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
    printf("Error: Cannot handle if data type is Pattern \n");
    exit(1);
  }
  int isSymmetric = 0, isGeneral = 0;
  if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
    isGeneral = 1;
  else {
    if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
      isSymmetric = 1;
      printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
    }
  }     
  if ( (isGeneral==0) && (isSymmetric==0) )       {
    printf("Warning: Matrix type should be General or Symmetric. \n");
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  do {
    assert(fgets(line, 1024, file) != NULL);
  } while ( line[0] == '%' );
  
  /* Read the matrix parameters */
  int NS=0, NT=0, NV = 0, NE=0;
  if (sscanf(line, "%d %d %d",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket_CSC(): bad file format - 02\n");
    exit(1);
  }
  NV = NS + NT;
  printf("|S|= %d, |T|= %d, |E|= %d \n", NS, NT, NE);
  
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each column
  int  *rowPtr = (int *)  malloc((NT+1) * sizeof(int));
#pragma omp parallel for
  for (int i=0; i <= NT; i++)
    rowPtr[i] = 0;
  
  edge *edgeListTmp; //Read the data in a temporary list
  int newNNZ = 0;    //New edges because of symmetric matrices
  int Si, Ti;
  double weight;
  if( isSymmetric == 1 ) {
    printf("Matrix is of type: Symmetric Real or Complex\n");
    edgeListTmp = (edge *) malloc(2 * NE * sizeof(edge));
    for (int i = 0; i < NE; i++) {
      assert(fscanf(file, "%d %d %lf", &Si, &Ti, &weight) == 3);
      Si--; Ti--;            // One-based indexing
      //weight = fabs(weight); //Make it positive  : Leave it as is
      if ( Si == Ti ) {
        edgeListTmp[i].head = Si;     //The S index = Row
        edgeListTmp[i].tail = Ti;     //The T index = Col 
        edgeListTmp[i].weight = weight; //The value
        rowPtr[Ti+1]++; //Increment for Column
      }
      else { //an off diagonal element: Also store the upper part
        //LOWER PART:
        edgeListTmp[i].head = Si;       //The S index 
        edgeListTmp[i].tail = Ti;    //The T index 
        edgeListTmp[i].weight = weight; //The value
        rowPtr[Ti+1]++;
        //UPPER PART:
        edgeListTmp[NE+newNNZ].head = Ti;    //The S index
        edgeListTmp[NE+newNNZ].tail = Si;    //The T index
        edgeListTmp[NE+newNNZ].weight = weight; //The value
        newNNZ++; //Increment the number of edges
        rowPtr[Si+1]++;
      }
    }
  } //End of Symmetric
  /////// General Real or Complex ///////
  else {
    printf("Matrix is of type: Unsymmetric Real or Complex\n");
    edgeListTmp = (edge *) malloc( NE * sizeof(edge));
    for (int i = 0; i < NE; i++) {
      assert(fscanf(file, "%d %d %lf", &Si, &Ti, &weight) == 3);
      Si--; Ti--;            // One-based indexing
      //weight = fabs(weight); //Make it positive    : Leave it as is
      edgeListTmp[i].head = Si;    //The S index = Row
      edgeListTmp[i].tail = Ti;    //The T index = Col
      edgeListTmp[i].weight = weight; //The value
      rowPtr[Ti+1]++;
    }
  } //End of Real or Complex

  fclose(file); //Close the file
  printf("Done reading from file.\n");
  if( isSymmetric ) {
    printf("Modifying number of edges from %d ", NE);
    NE += newNNZ; //#NNZ might change
    printf("to %d \n", NE);
  }
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = timer();
  for (int i=0; i<NT; i++) {
    rowPtr[i+1] += rowPtr[i]; //Prefix Sum
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = timer();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: |E| = %d, rowPtr[NV]= %d\n", NE, rowPtr[NT]);
  
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  time1 = timer();
  //Every edge stored ONLY ONCE!
  int *rowIndex   = (int*)    malloc (NE * sizeof(int)); 
  double *weights = (double*) malloc (NE * sizeof(double));
  time2 = timer();
  printf("Time for allocating memory for marks and edgeList = %lf\n", time2 - time1);
  
  time1 = timer();
  //Keep track of how many edges have been added for a vertex:
  int  *added    = (int *)  malloc( NT  * sizeof(int));
#pragma omp parallel for
  for (int i = 0; i < NT; i++) 
    added[i] = 0;
  
  printf("Building edgeList...\n");
#pragma omp parallel for
  for(int i=0; i<NE; i++) {
    int head      = edgeListTmp[i].head;   //row id
    int tail      = edgeListTmp[i].tail;   //col id
    double weight = edgeListTmp[i].weight; //weight
    int Where     = rowPtr[tail] + added[tail];
    rowIndex[Where] = head;  //Add the row id
    weights[Where]  = weight;
    added[tail]++;
  }
  time2 = timer();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  M->nRows    = NS;
  M->nCols    = NT;
  M->nNNZ     = NE;
  M->RowPtr   = rowPtr;
  M->RowInd   = rowIndex;
  M->Weights  = weights;    
  
  free(edgeListTmp);
  free(added);
}

/*-------------------------------------------------------*
 * This function reads a Simple Matrix Market file and builds a 
 * matrix in CSR format
 *-------------------------------------------------------*/
void parse_Simple_CSC(matrix_CSC * M, char *fileName) {
  printf("Parsing a Simple Matrix Market File...\n");
#pragma omp parallel 
  {
    int nthreads = omp_get_num_threads();
    int nprocs   = omp_get_num_procs();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("File I/O: Number of threads: %d\n Number of procs: %d\n", nthreads, nprocs);
  }
  double time1, time2;      
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  assert(fgets(line, 1024, file) != NULL);
  /* Read the matrix parameters */
  int NS=0, NT=0, NV = 0, NE=0;
  if (sscanf(line, "%d %d %d",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket_CSC(): bad file format - 02\n");
    exit(1);
  }
  NV = NS + NT;
  printf("|S|= %d, |T|= %d, |E|= %d \n", NS, NT, NE);
  
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each column
  int  *rowPtr = (int *)  malloc((NT+1) * sizeof(int));
#pragma omp parallel for
  for (int i=0; i <= NT; i++)
    rowPtr[i] = 0;
  
  edge *edgeListTmp; //Read the data in a temporary list
  int Si, Ti;
  double weight;
  /////// General Real or Complex ///////
  printf("Matrix is of type: Unsymmetric Real or Complex\n");
  edgeListTmp = (edge *) malloc( NE * sizeof(edge));
  for (int i = 0; i < NE; i++) {
    assert(fscanf(file, "%d %d %lf", &Si, &Ti, &weight) == 3);
    Si--; Ti--;            // One-based indexing
    //weight = fabs(weight); //Make it positive    : Leave it as is
    edgeListTmp[i].head = Si;    //The S index = Row
    edgeListTmp[i].tail = Ti;    //The T index = Col
    edgeListTmp[i].weight = weight; //The value
    rowPtr[Ti+1]++;
  }
  fclose(file); //Close the file
  printf("Done reading from file.\n");
   
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = timer();
  for (int i=0; i<NT; i++) {
    rowPtr[i+1] += rowPtr[i]; //Prefix Sum
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = timer();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: |E| = %d, rowPtr[NV]= %d\n", NE, rowPtr[NT]);
  
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  time1 = timer();
  //Every edge stored ONLY ONCE!
  int *rowIndex   = (int*)    malloc (NE * sizeof(int)); 
  double *weights = (double*) malloc (NE * sizeof(double));
  time2 = timer();
  printf("Time for allocating memory for marks and edgeList = %lf\n", time2 - time1);
  
  time1 = timer();
  //Keep track of how many edges have been added for a vertex:
  int  *added    = (int *)  malloc( NT  * sizeof(int));
#pragma omp parallel for
  for (int i = 0; i < NT; i++) 
    added[i] = 0;
  
  printf("Building edgeList...\n");
#pragma omp parallel for
  for(int i=0; i<NE; i++) {
    int head      = edgeListTmp[i].head;   //row id
    int tail      = edgeListTmp[i].tail;   //col id
    double weight = edgeListTmp[i].weight; //weight
    int Where     = rowPtr[tail] + added[tail];
    rowIndex[Where] = head;  //Add the row id
    weights[Where]  = weight;
    added[tail]++;
  }
  time2 = timer();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  M->nRows    = NS;
  M->nCols    = NT;
  M->nNNZ     = NE;
  M->RowPtr   = rowPtr;
  M->RowInd   = rowIndex;
  M->Weights  = weights;    
  
  free(edgeListTmp);
  free(added);
}

/*-------------------------------------------------------*
 * This function reads a file and builds the vector y
 * Comments are in Matrix-Market style
 *-------------------------------------------------------*/
void parse_YVector(int *Y, int sizeY, char *fileName) {
  printf("Parsing for vector Y..\n");
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int nprocs   = omp_get_num_procs();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("Coloring Rouinte: Number of threads: %d\n Number of procs: %d\n", nthreads, nprocs);
  }  
  double time1, time2;
  
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  char line[1024];
  do {
    assert(fgets(line, 1024, file) != NULL);
  } while ( line[0] == '%' );
  
  int numY=0;
  if (sscanf(line, "%d",&numY ) != 1) {
    printf("Read Y: bad file format");
    exit(1);
  }
  printf("|Y|= %d\n", numY);
  
  /*---------------------------------------------------------------------*/
  /* Read Y list  (row_id, y_value)                                      */
  /*---------------------------------------------------------------------*/
  printf("Reading the Y values...\n");
  time1 = timer();
  int row_id, y_value;    
  for (int i = 0; i < numY; i++) {      
    assert(fscanf(file, "%d %d", &row_id, &y_value) == 2);
    //printf("%d - %d\n", row_id, y_value);
    Y[row_id-1] = y_value; // One-based indexing
  }
  time2 = timer();    
  printf("Done reading from file. It took %lf seconds.\n", time2-time1);
  
  fclose(file); //Close the file

}

/*-------------------------------------------------------*
 * This function reads 3-arrays of (i,j,w) and creates 
 * The Graph data structure...
 *-------------------------------------------------------*/
void create_graph(graph * G, double sSize, double tSize, double wSize, double* sArray, double* tArray, double* wArray) {
  //double time1, time2,totTime=0.0;
  //time1 = timer();
  /* Read the matrix parameters */
  int NS=0, NT=0, NV = 0, NE=0;
  NS=sSize;
  NT=tSize;
  NE=wSize;
   
  NV = NS + NT;
  
  /*for(int i=0; i<10; i++)
        printf("%d ",sArray[i]);
  printf("\n");
  for(int i=0; i<10; i++)
        printf("%d ",tArray[i]);
  printf("\n");
  for(int i=0; i<10; i++)
        printf("%f ",wArray[i]);
  printf("\n");*/
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* S vertices: 0 to NS-1                                               */
  /* T vertices: NS to NS+NT-1                                           */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each vertex
    int  *edgeListPtr = (int *)  malloc((NV+1) * sizeof(int));
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for (int i=0; i <= NV; i++)
        edgeListPtr[i] = 0; //For first touch purposes
  
  edge *edgeListTmp; //Read the data in a temporary list
  
  int s,t;
  double w;
  edgeListTmp = (edge *) malloc( NE * sizeof(edge));
  for (int i = 0; i < NE; i++) {
         s=sArray[i];
         t=tArray[i];
         w=wArray[i];
         s--;
         t--;
   // printf("Indices: %d %d %d %f %d %d \n",i,sArray[i],tArray[i],wArray[i],s,t);
    edgeListTmp[i].head = s;       //The S index
    edgeListTmp[i].tail = NS+t;    //The T index
    edgeListTmp[i].weight = w; //The value
    edgeListPtr[s+1]++;
    edgeListPtr[NS+t+1]++;
   }
  //printf("Sizes: %d, %d, %d\n",NS,NT,NE);
  //////Build the EdgeListPtr Array: Cumulative addition 
  for (int i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  #ifdef DYNSCHED
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<NV; i++)
  {
    int s=edgeListPtr[i];
    int t=edgeListPtr[i+1];
    for(int k=s;k<t;k++)
            edgeList[k].weight=0.0; // For First touch purpose
  }
  //Keep track of how many edges have been added for a vertex:
  int  *added    = (int *)  malloc( NV  * sizeof(int));
    
  #ifdef DYNSCHED
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for (int i = 0; i < NV; i++)
    added[i] = 0;
  
   //Build the edgeList from edgeListTmp:
//#pragma omp parallel for
  for(int i=0; i<NE; i++) {
    int head = edgeListTmp[i].head;
    int tail = edgeListTmp[i].tail;
    double weight = edgeListTmp[i].weight;
    int Where = edgeListPtr[head] + added[head];   
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    added[head]++;
    //Now add the counter-edge:
    Where = edgeListPtr[tail] + added[tail];
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;
    added[tail]++;
  }
  
  free(added);
  free(edgeListTmp);
  
  edge *TmpList = (edge *) malloc (2*NE * sizeof(edge));
  SortNodeEdgesByIndex2(NV, edgeList, TmpList, edgeListPtr);
    
  G->sVertices    = NS;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->marks        = NULL;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  free(TmpList);
  //time2 = timer();
  //totTime += time2 - time1;
  //printf("Time to sort edge lists by index\n");  
}

void parse_STW(double* s, double*t, double* w, char *fileName) {
  //printf("Parsing a Matrix Market File...\n");
/*#pragma omp parallel 
  {
    int nthreads = omp_get_num_threads();
    int nprocs   = omp_get_num_procs();
    int tid      = omp_get_thread_num();
    //if (tid == 0)
      //printf("parse_MatrixMarket: Number of threads: %d\n Number of procs: %d\n", nthreads, nprocs);
  }*/
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  assert(fgets(line, 1024, file) != NULL);
  char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];    
  if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
    printf("parse_MatrixMarket(): bad file format - 01");
    exit(1);
  }
  //printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
  if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
    printf("Error: The first line should start with %%MatrixMarket word \n");
    exit(1);
  }
  if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
    printf("Error: The Object should be matrix or Matrix or MATRIX \n");
    exit(1);
  }
  if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
    printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
    exit(1);
  }
  int isComplex = 0;    
  if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
    isComplex = 1;
    printf("Warning: Will only read the real part. \n");
  }
  int isPattern = 0;
  if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
    isPattern = 1;
    printf("Note: Matrix type is Pattern. Will set all weights to 1.\n");
    //exit(1);
  }
  int isSymmetric = 0, isGeneral = 0;
  if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
    isGeneral = 1;
  else {
    if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
      isSymmetric = 1;
      printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
    }
  }     
  if ( (isGeneral==0) && (isSymmetric==0) )       {
    printf("Warning: Matrix type should be General or Symmetric. \n");
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  do {
    assert(fgets(line, 1024, file) != NULL);
  } while ( line[0] == '%' );
  
  /* Read the matrix parameters */
  int NS=0, NT=0, NV = 0, NE=0;
  if (sscanf(line, "%d %d %d",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket(): bad file format - 02");
    exit(1);
  }
  NV = NS + NT;
  //printf("|S|= %d, |T|= %d, |E|= %d \n", NS, NT, NE);

  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* S vertices: 0 to NS-1                                               */
  /* T vertices: NS to NS+NT-1                                           */
  /*---------------------------------------------------------------------*/
  int Si, Ti;
  double weight = 1;
 for (int i = 0; i < NE; i++) {
      assert(fscanf(file, "%d %d %lf", &Si, &Ti, &weight) == 3);
      s[i]=Si;
      t[i]=Ti;
      w[i]=weight;
     }
  
  fclose(file); //Close the file
  //printf("Done reading from file.\n");
  

}

void get_params(int* NS, int* NT, int* NE, char* fileName)
{

  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  assert(fgets(line, 1024, file) != NULL);
  char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];    
  if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
    printf("parse_MatrixMarket(): bad file format - 01");
    exit(1);
  }
  //printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
  if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
    printf("Error: The first line should start with %%MatrixMarket word \n");
    exit(1);
  }
  if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
    printf("Error: The Object should be matrix or Matrix or MATRIX \n");
    exit(1);
  }
  if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
    printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
    exit(1);
  }
  int isComplex = 0;    
  if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
    isComplex = 1;
    printf("Warning: Will only read the real part. \n");
  }
  int isPattern = 0;
  if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
    isPattern = 1;
    printf("Note: Matrix type is Pattern. Will set all weights to 1.\n");
    //exit(1);
  }
  int isSymmetric = 0, isGeneral = 0;
  if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
    isGeneral = 1;
  else {
    if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
      isSymmetric = 1;
      printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
    }
  }     
  if ( (isGeneral==0) && (isSymmetric==0) )       {
    printf("Warning: Matrix type should be General or Symmetric. \n");
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  do {
    assert(fgets(line, 1024, file) != NULL);
  } while ( line[0] == '%' );
  
  /* Read the matrix parameters */
  
  sscanf(line, "%d %d %d",NS, NT, NE);
  
  fclose(file); //Close the file
  
}
void update_weight(graph* G, double* li, double* lj, double* w)
{
    int NE=G->numEdges;
    int NS=G->sVertices;
    int *verPtr=G->edgeListPtrs;
    edge *verInd=G->edgeList;
   
    #pragma omp parallel for
    for(int k=0;k<NE;k++)
    {
        int i=(int)li[k]-1;  /// Assuming li, lj is one based
        int j=(int)lj[k]-1+NS;
        
        int s=verPtr[i];
        int t=verPtr[i+1];
        
        for(int x=s;x<t;x++)
            if(verInd[x].tail==j)
                verInd[x].weight=w[k];
        
        s=verPtr[j];
        t=verPtr[j+1];
        for(int x=s;x<t;x++)
            if(verInd[x].tail==i)
                verInd[x].weight=w[k];
             
    }
    
}
