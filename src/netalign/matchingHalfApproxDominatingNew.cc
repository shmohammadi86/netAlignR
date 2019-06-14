/*---------------------------------------------------------------------------*/
/*                                                                           */
/*                          Mahantesh Halappanavar                           */
/*                        High Performance Computing                         */
/*                Pacific Northwest National Lab, Richland, WA               */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*                                                                           */
/* Copyright (C) 2010 Mahantesh Halappanavar                                 */
/*                                                                           */
/*                                                                           */
/* This program is free software; you can redistribute it and/or             */
/* modify it under the terms of the GNU General Public License               */
/* as published by the Free Software Foundation; either version 2            */
/* of the License, or (at your option) any later version.                    */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU General Public License for more details.                              */
/*                                                                           */
/* You should have received a copy of the GNU General Public License         */
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 59 Temple Place-Suite 330,Boston,MA 02111-1307,USA.     */
/*                                                                           */
/*---------------------------------------------------------------------------*/

/*****************************************************************************/

#include "coloringAndMatchingKernels.h"
#include "defs.h"
#include "netAlignKernel.h" 

//////////////////////////////// Initial Extreme Matching ///////////////////////
//#define  PRINT_STATISTICS_
void algoEdgeApproxInitialExtremeMatchingBipartiteParallelPath3( graph *G, int *Mate, double* RMax, int* Visited )
{
  /*int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }*/
  //printf("Within algoEdgeApproxInitialExtremeMatchingBipartiteParallel() -- %d threads\n", nthreads);
  //double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  int NVer     = G->numVertices;
  int NS       = G->sVertices;
  int NT       = NVer - NS;
  int *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  //printf("NS= %ld  NT=%ld  NE=%ld\n", NS, NT, NEdge);
    
  //Step-1: Store local maximum for each S vertex  
  //double *RMax      = (double *) malloc (NS * sizeof(double));
  if( RMax == NULL ) {
    printf("Not enough memory to allocate for RMax \n");
    exit(1);
  }
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<NS; i++ ) {
    int adj1 = verPtr[i];
    int adj2 = verPtr[i+1];
    double RowMaximum = MilanRealMin;
    for(int k = adj1; k < adj2; k++ )
      if( RowMaximum < verInd[k].weight )
	RowMaximum = verInd[k].weight;
    RMax[i] = RowMaximum;
  }
  //int *Visited = (int *)   malloc (NT * sizeof(int));
  if( Visited == NULL ) {
    printf("Not enough memory to allocate for Processed \n");
    exit(1);
  }
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<NT; i++ ) {
    Visited[i] = 0;
  }

  //Compute matchings from edges that are locally maximum (tight)
  
  //Step-2: Find augmenting paths of length one
  //time1  = timer();
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<NS; i++ ) {
    int adj1 = verPtr[i];
    int adj2 = verPtr[i+1];
    //Scan the neighborhood for an eligible edge
    for(int k = adj1; k < adj2; k++ ) {
      int w = verInd[k].tail;
      //If the neighbor is tight and unmatched
      if( (verInd[k].weight == RMax[i])&&(Mate[w] == -1) ) {
	//Ignore if processed by another vertex
	if ( __sync_fetch_and_add(&Visited[w-NS], 1) == 0 ) {
#ifdef PRINT_STATISTICS_
	  __sync_fetch_and_add(&cardinality, 1);
#endif
	  Mate[i] = w;         //Set the Mate array
	  Mate[w] = i;
	  break;
	} //End of if(visited)
      } //End of if(Tight and unmatched)
    } //End of for inner loop
  } //End of for outer loop
  //time1  = timer() - time1;
#ifdef PRINT_STATISTICS_
  printf("\n Results after Stage 1: \n");
  printf("Cardinality: %d\n", cardinality);
#endif

  //Reuse Visited array
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<NT; i++ ) {
    Visited[i] = 0;
  }
    
  //STEP 3: Find Short augmenting paths from unmatched nodes
  //time2  = timer();
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int u=0; u<NS; u++ ) {
    if( Mate[u] >= 0 ) //Ignore matched vertices
      continue;
    
    //If u is unmatched, find an augmenting path of length 3
    int pathFound = 0;
    int adj1 = verPtr[u];
    int adj2 = verPtr[u+1];
    //Process all the neighbors
    for(int k = adj1; k < adj2; k++) {
      if ( pathFound == 1 )
	break;
      int w = verInd[k].tail;
      //Consider neighbors: tight AND matched
      if( (verInd[k].weight == RMax[u])&&(Mate[w] >= 0) ) {
	int u1 = Mate[w]; //Get the other end of matched edge
	//Check for an unmatched row node
	int adj11 = verPtr[u1]; 
	int adj12 = verPtr[u1+1];
	for(int k1 = adj11; k1 < adj12; k1++) {
	  int w1 = verInd[k1].tail;
	  //Look if the row node is matched: Tight AND unmatched edge
	  if( (verInd[k1].weight == RMax[u1])&&(Mate[w1] == -1) ) {
	    //!!!! WARNING: The logic is not validated yet
	    //Ignore if processed by another vertex
	    if ( __sync_fetch_and_add(&Visited[w1-NS], 1) == 0 ) {
	      //AUGMENT:
	      Mate[u] = w;
	      Mate[w] = u;
	      
	      Mate[u1] = w1;
	      Mate[w1] = u1;
	      
#ifdef PRINT_STATISTICS_
	      __sync_fetch_and_add(&cardinality, 1);	
#endif	    
	      pathFound = 1;
	      break;
	    }//End of if(Visited)
	  }//End of if()
	}//End of inner for loop
      }//End of if
    } //End of outer for loop 
  } // End of outer for loop
  //time2  = timer() - time2;
#ifdef PRINT_STATISTICS_
  printf("\n Results after Stage 2: \n");
  printf("Cardinality: %d\n", cardinality);
#endif
  
  /*printf("***********************************************\n");
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("Time for Phase-2           : %lf sec\n", time2);
  printf("Total Time                 : %lf sec\n", time1+time2);
  printf("***********************************************\n");*/

  //Cleanip
  //free(RMax);
  //free(Visited);

} // End of algoEdgeApproxInitialExtremeMatchingBipartiteParallel()


void algoEdgeApproxInitialExtremeMatchingBipartiteParallelPath1( graph *G, int *Mate, double* RMax, int* Visited)
{
  /*int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }*/
  //printf("Within algoEdgeApproxInitialExtremeMatchingBipartiteParallel2() -- %d threads\n", nthreads);
  //double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  int NVer     = G->numVertices;
  int NS       = G->sVertices;
  int NT       = NVer - NS;
  int *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  //printf("NS= %ld  NT=%ld  NE=%ld\n", NS, NT, NEdge);
  
  
  //Step-1: Store local maximum for each S vertex  
  //double *RMax      = (double *) malloc (NS * sizeof(double));
  if( RMax == NULL ) {
    printf("Not enough memory to allocate for RMax \n");
    exit(1);
  }
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<NS; i++ ) {
    int adj1 = verPtr[i];
    int adj2 = verPtr[i+1];
    double RowMaximum = MilanRealMin;
    for(int k = adj1; k < adj2; k++ )
      if( RowMaximum < verInd[k].weight )
	RowMaximum = verInd[k].weight;
    RMax[i] = RowMaximum;
  }
  //int *Visited = (int *)   malloc (NT * sizeof(int));
  if( Visited == NULL ) {
    printf("Not enough memory to allocate for Processed \n");
    exit(1);
  }
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<NT; i++ ) {
    Visited[i] = 0;
  }

  //Compute matchings from edges that are locally maximum (tight)
  
  //Step-2: Find augmenting paths of length one
  //time1  = timer();
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<NS; i++ ) {
    int adj1 = verPtr[i];
    int adj2 = verPtr[i+1];
    //Scan the neighborhood for an eligible edge
    for(int k = adj1; k < adj2; k++ ) {
      int w = verInd[k].tail;
      //If the neighbor is tight and unmatched
      if( (verInd[k].weight == RMax[i])&&(Mate[w] == -1) ) {
	//Ignore if processed by another vertex
	if ( __sync_fetch_and_add(&Visited[w-NS], 1) == 0 ) {
#ifdef PRINT_STATISTICS_
	  __sync_fetch_and_add(&cardinality, 1);
#endif
	  Mate[i] = w;         //Set the Mate array
	  Mate[w] = i;
	  break;
	} //End of if(visited)
      } //End of if(Tight and unmatched)
    } //End of for inner loop
  } //End of for outer loop
  //time1  = timer() - time1;
#ifdef PRINT_STATISTICS_
  printf("\n Results after Stage 1: \n");
  printf("Cardinality: %d\n", cardinality);
#endif

  /*printf("***********************************************\n");
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("***********************************************\n");*/

  //Cleanip
  //free(RMax);
  //free(Visited);
} // End of InitialExtremeMatching()


void algoEdgeApproxInitialExtremeMatchingBipartiteSerial( graph *G, int *Mate, double* RMax)
{
  //printf("Within algoEdgeApproxInitialExtremeMatchingBipartiteSerial()\n");
  //Get the iterators for the graph:
  int NS       = G->sVertices;
  int *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
    
  //Vector DUAL: the first NC elements hold the Column side duals
  //the the next NR elements hold the Row side duals
  int i=0,k=0,k1=0,u=0,u1=0, w=0,w1=0; //Temporary Variables
  int adj1=0, adj2=0, adj11=0, adj12=0;
  double RowMaximum=0.0f;
  int pathFound = 0;

  //Step-1: Store local maximum for each S vertex
  //double *RMax = (double *) malloc (NS * sizeof(double));
  if( RMax == NULL ) {
    printf("Not enough memory to allocate for RMax \n");
    exit(1);
  }  
  for( i=0; i<NS; i++ ) {
    adj1 = verPtr[i];
    adj2 = verPtr[i+1];
    RowMaximum = MilanRealMin;
    for( k = adj1; k < adj2; k++ )
      if( RowMaximum < verInd[k].weight )
	RowMaximum = verInd[k].weight;
    RMax[i] = RowMaximum;
  }

  //Compute matchings from edges that are locally maximum (tight)
  
  //Step-2: Find augmenting paths of length one
  for( i=0; i<NS; i++ ) {
    adj1 = verPtr[i];
    adj2 = verPtr[i+1];
    //Scan the neighborhood for an eligible edge
    for( k = adj1; k < adj2; k++ ) {
      w = verInd[k].tail;
      if( verInd[k].weight == RMax[i] ) { //Tight edge?
	if( Mate[w] == -1)  { //Not matched before
#ifdef PRINT_STATISTICS_
	  cardinality++;
#endif
	  Mate[i] = w;         //Set the Mate array
	  Mate[w] = i;
	  break;
	}
      } //End of if
    } //End of for inner loop
  } //End of for outer loop
  
#ifdef PRINT_STATISTICS_
  printf("\n Results after Stage 1: \n");
  printf("Cardinality: %d\n", cardinality);
#endif
  
  //STEP 3: Find Short augmenting paths from unmatched nodes
  for( u=0; u<NS; u++ ) {
    if( Mate[u] >= 0 ) //Ignore matched S vertices
      continue;
    
    pathFound = 0;
    adj1 = verPtr[u];
    adj2 = verPtr[u+1];
    //Process all the neighbors
    for(k = adj1; k < adj2; k++) {
      if ( pathFound == 1 )
	break;
      w = verInd[k].tail;
      //Consider neighbors: tight AND matched
      if( (verInd[k].weight == RMax[u])&&(Mate[w] >= 0) ) { 
	u1 = Mate[w]; //Get the other end of matched edge	
	//Check for an unmatched row node
	adj11 = verPtr[u1]; 
	adj12 = verPtr[u1+1];
	for(k1 = adj11; k1 < adj12; k1++) {
	  w1 = verInd[k1].tail;
	  //Look if the row node is matched: Tight AND unmatched edge
	  if( (verInd[k1].weight == RMax[u1])&&(Mate[w1] == -1) ) {
	    //AUGMENT:
	    Mate[u] = w;
	    Mate[w] = u;
	    
	    Mate[u1] = w1;
	    Mate[w1] = u1;
	    
#ifdef PRINT_STATISTICS_
	    cardinality++;	
#endif	    
	    pathFound = 1;
	    break;
	  }//End of if()
	}//End of inner for loop
      }//End of if
    } //End of outer for loop 
  } // End of outer for loop
  
#ifdef PRINT_STATISTICS_
  printf("\n Results after Stage 2: \n");
  printf("Cardinality: %d\n", cardinality);
#endif
} // End of InitialExtremeMatching()

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  DOMINATING EDGE ALGORITHM  ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
/**
 * @return -1 on error, 0 on success
 */
int algoEdgeApproxDominatingEdgesLinearSearchNew( graph *G, double* weight, 
    int *Mate, int* HeaviestPointer, int *Q1, int *Q2, double* RMax, int* Visited)
{
/*#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    #pragma omp master
    printf("Threads: %d\n",nthreads);
    int tid = omp_get_thread_num();
  }*/
  
  //printf("Here I m\n");
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  int NVer     = G->numVertices;
  int *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)

  if( HeaviestPointer == NULL ) {
    return -1;
  }
  //Initialize the Vectors:
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for (int i=0; i<NVer; i++)
  {  
      Mate[i]=-1;
      HeaviestPointer[i]= -1;
  }
  //The Queue Data Structure for the Dominating Set:
  //The Queues are important for synchornizing the concurrency:
  //Have two queues - read from one, write into another
  // at the end, swap the two.
  //int *Q    = (int *) malloc (NVer * sizeof(int));
  //int *Qtmp = (int *) malloc (NVer * sizeof(int));
  int *Q, *Qtmp;
  int ownQ = 0;
  if (Q1 == NULL) {
    Q    = (int *) malloc (NVer * sizeof(int));
    Qtmp = (int *) malloc (NVer * sizeof(int));
  } else {
    Q = Q1;
    Qtmp = Q2;
  }
  int *Qswap;
  if( (Q == NULL) || (Qtmp == NULL) ) {
    fprintf(stderr,"Not enough memory to allocate for the two queues \n");
    return -1;
  }
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for (int i=0; i<NVer; i++) {
    Q[i]    = -1; 
    Qtmp[i] = -1;
  }
  int QTail   =0; //Tail of the queue (implicitly will represent the size)
  int QtmpTail=0; //Tail of the queue (implicitly will represent the size)
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// PART 1 ////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  //Compute the Initial Matching Set:

  //algoEdgeApproxInitialExtremeMatchingBipartiteSerial(G,Mate,RMax);
  //algoEdgeApproxInitialExtremeMatchingBipartiteParallelPath3(G,Mate,RMax,Visited);
  algoEdgeApproxInitialExtremeMatchingBipartiteParallelPath1(G,Mate,RMax,Visited);
  time1 = timer();
  
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for (int v=0; v < NVer; v++ ) {
    //Start: COMPUTE_CANDIDATE_MATE(v)
    int adj1 = verPtr[v];
    int adj2 = verPtr[v+1];
    int w = -1;
    //double heaviestEdgeWt = MilanRealMin; //Assign the smallest Value possible first LDBL_
    double heaviestEdgeWt = 0.0f; //Assign zero to avoid negative edges
    for(int k = adj1; k < adj2; k++ ) {
      if ( Mate[verInd[k].tail] == -1 ) { //Process only if unmatched
        if( weight[k] <= 0 ) //Ignore all zero and negative weights^M
           continue;
        if( (weight[k] > heaviestEdgeWt) || 
            ((weight[k] == heaviestEdgeWt)&&(w<verInd[k].tail)) ) {
          heaviestEdgeWt = weight[k];
          w = verInd[k].tail;
        }
      }//End of if (Mate == -1)
    } //End of for loop
    HeaviestPointer[v] = w; // c(v) <- Hsv(v)
  } //End of for loop for setting the pointers
  //Check if two vertices point to each other:

  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for (int v=0; v < NVer; v++ ) {
    //If found a dominating edge:
    if ( HeaviestPointer[v] >= 0 )
      if ( HeaviestPointer[HeaviestPointer[v]] == v ) {
        Mate[v] = HeaviestPointer[v];
        //Q.push_back(u,w);
        int whereInQ = __sync_fetch_and_add(&QTail, 1);
        Q[whereInQ] = v;
      }//End of if(Pointer(Pointer(v))==v)
  }//End of for(int v=0; v < NVer; v++ )
  time1  = timer() - time1;
  //The size of Q1 is now QTail+1; the elements are contained in Q1[0] through Q1[Q1Tail]
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// PART 2 ////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  int nLoops=0; //Count number of iterations in the while loop
  while ( /*!Q.empty()*/ QTail > 0 )  {      
    //printf("Loop %d, QSize= %d\n", nLoops, QTail);
    //KEY IDEA: Process all the members of the queue concurrently:
    time2 = timer();

    #ifdef DYNSCHED  
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for (int Qi=0; Qi<QTail; Qi++) {
      //Q.pop_front();
      int v = Q[Qi];            
      int adj1 = verPtr[v]; 
      int adj2 = verPtr[v+1];
      for(int k = adj1; k < adj2; k++) {
        int x = verInd[k].tail;
        if ( Mate[x] != -1 )   // x in Sv \ {c(v)}
          continue;
        if ( HeaviestPointer[x] == v ) {
          //Start: PROCESS_EXPOSED_VERTEX(x)
          //Start: COMPUTE_CANDIDATE_MATE(x)
          int adj11 = verPtr[x];
          int adj12 = verPtr[x+1];
          int w = -1;
          //double heaviestEdgeWt = MilanRealMin; //Assign the smallest Value possible first LDBL_MIN
          double heaviestEdgeWt = 0.0f; //Assign zero to avoid negative edges
          for(int k1 = adj11; k1 < adj12; k1++ ) {
            if( Mate[verInd[k1].tail] != -1 ) // Sx <- Sx \ {v}
              continue;
            if( weight[k1] <= 0 ) //Ignore all zero and negative weights
                        continue;
            if( (weight[k1] > heaviestEdgeWt) || 
                ((weight[k1] == heaviestEdgeWt)&&(w < verInd[k1].tail)) ) {
              heaviestEdgeWt = weight[k1];
              w = verInd[k1].tail;
            }
          }//End of for loop on k1
          HeaviestPointer[x] = w; // c(x) <- Hsv(x)
          //End: COMPUTE_CANDIDATE_MATE(v)
          //If found a dominating edge:
          if ( HeaviestPointer[x] >= 0 ) 
            if ( HeaviestPointer[HeaviestPointer[x]] == x ) {
              Mate[x] = HeaviestPointer[x];
              Mate[HeaviestPointer[x]] = x;
              //Q.push_back(u);
              int whereInQ = __sync_fetch_and_add(&QtmpTail, 2);
              Qtmp[whereInQ] = x;                    //add u
              Qtmp[whereInQ+1] = HeaviestPointer[x]; //add w
            } //End of if found a dominating edge
        } //End of if ( HeaviestPointer[x] == v )
      } //End of for loop on k: the neighborhood of v
    } //End of for loop on i: the number of vertices in the Queue
    
    ///Also end of the parallel region
    //Swap the two queues:
    Qswap = Q;
    Q = Qtmp; //Q now points to the second vector
    Qtmp = Qswap;
    QTail = QtmpTail; //Number of elements
    QtmpTail = 0; //Symbolic emptying of the second queue
    nLoops++;
    time2  = timer() - time2;
    totalTime += time2;
  } //end of while ( !Q.empty() )
  if (ownQ) {
    free(Q); 
    free(Qtmp);
  }
  return (0);
} //End of algoEdgeApproxDominatingEdgesLinearSearch


double ApproxEdgeWeightedMatching( double *indicator, 
        int* Mate, int* HeaviestPointer, 
        double sSize, double tSize, double wSize,  
        double* sArray, double *tArray, double* wArray)
{
  graph* G = (graph *) malloc (sizeof(graph)) ;
  create_graph(G,sSize,tSize,wSize,sArray,tArray,wArray);
  //int *Mate = (int *) malloc (G->numVertices * sizeof(int));
  
  #ifdef DYNSCHED  
    #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
    #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<(G->numVertices); i++)
    Mate[i] = -1;
    
  algoEdgeApproxDominatingEdgesLinearSearchNew(G, wArray,Mate,HeaviestPointer,NULL,NULL,NULL,NULL);
  /* Step 4: Compute weight and cardinality of the matching */
  int NVer        = G->numVertices;
  int *verPtr     = G->edgeListPtrs;  //Vertex Pointer
  edge *verInd    = G->edgeList;      //Vertex Index
  
  double weight = 0;
  int cardinality = 0;
  for(int i=0; i<NVer; i++) {
    if ( Mate[i] >= 0 ) {
      int adj1 = verPtr[i];
      int adj2 = verPtr[i+1];
      for(int j=adj1; j < adj2; j++)
        if( verInd[j].tail == Mate[i] ) {
          weight += verInd[j].weight;
          cardinality++;
          break;
        } //End of inner if
    } //End of outer if
  } //End of for
  
  #ifdef DYNSCHED  
      #pragma omp parallel for schedule(dynamic, CHUNK)
  #else
      #pragma omp parallel for schedule(static, CHUNK)
  #endif
  for(int i=0; i<(int)wSize;i++)
  {
    int s=sArray[i]-1;
    int t=tArray[i]+sSize-1;
    if(Mate[s]==t)
    {    
       // #pragma omp atomic
        //count++;
        indicator[i]=1.00;
    }
  }
  
  free(G->edgeListPtrs);
  free(G->edgeList);
  free(G->marks);
  free(G);
  return weight/2;
}
      
double ApproxEdgeWeightedMatching( double *indicator, 
            int* Mate, int* HeaviestPointer, 
            int* Q1, int* Q2, graph* G, double* w, double* RMax, int* Visited)
{
    
  algoEdgeApproxDominatingEdgesLinearSearchNew(G, w, Mate, HeaviestPointer, Q1, Q2, RMax, Visited);
   
  int NS          = G->sVertices;
  int *verPtr     = G->edgeListPtrs;  //Vertex Pointer
  edge *verInd    = G->edgeList;      //Vertex Index
  
  double weight = 0;
  int cardinality = 0;
   
  #ifdef DYNSCHED
    #pragma omp parallel for  schedule(dynamic, CHUNK)\
    reduction(+:weight) reduction(+:cardinality)
  #else
    #pragma omp parallel for  schedule(static, CHUNK)\
    reduction(+:weight) reduction(+:cardinality)
  #endif
  for(int i=0; i<NS; i++) {
    int adj1 = verPtr[i];
    int adj2 = verPtr[i+1];
    for(int j=adj1; j < adj2; j++) {
      if( verInd[j].tail == Mate[i] ) {
        indicator[j] = 1.;
        weight += w[j];
        cardinality++;
      } else {
        indicator[j] = 0.;
      }
    }
  }
  
  return weight;
}


double ApproxEdgeWeightedMatching( double *indicator, 
            int* Mate, int* HeaviestPointer, 
            graph* G)
{
    return ApproxEdgeWeightedMatching(indicator,
                Mate, HeaviestPointer, NULL, NULL, G,NULL,NULL,NULL);
}           

