#include "netAlignKernel.h"
#include <getopt.h>
#include <string>
#include <omp.h>

using namespace std;

CRS_Mat* extractCRSMatrix(graph* G)
{
    int NE=G->numEdges;
    int NS= G->sVertices;
    int NT=G->numVertices-NS;
    int* verPtr= G->edgeListPtrs; 
    edge* verInd= G->edgeList;
    
    int* vInd=new int[NE];
    double* val=new double[NE];

    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<NS;i++)
    {
        //Ptr[i]=verPtr[i];
        int s=verPtr[i];
        int t=verPtr[i+1];
        for(int j=s;j<t;j++)
        {
            vInd[j]=verInd[j].tail-NS;
            val[j]=verInd[j].weight;      
        }
    }
    CRS_Mat* S=new CRS_Mat(val,vInd,verPtr,NS,NT,NE,1);
	
    delete[] val;
    delete[] vInd;

    return S;
}

CRS_Mat* createSquareMatrix(CRS_Mat* A, CRS_Mat* B, graph* L)
{
    int *AverPtr= A->rowPtr(); 
    
    int *AverInd= A->colInd();
    
    int *BverPtr= B->rowPtr(); 
    int *BverInd= B->colInd();
    
    int LNE=L->numEdges;
    int LNS= L->sVertices;
    int *LverPtr= L->edgeListPtrs; 
    edge *LverInd= L->edgeList;

    double* prefSum=new double[LNS+1];
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<LNS;i++)
    {
        int count=0;
        int ls=LverPtr[i];
        int lt=LverPtr[i+1];
        for(int k=ls; k<lt;k++) {
            int j=LverInd[k].tail-LNS; // adjusting the j indices
            /// now (i,j) is an edge of L now we find i' in A
            int als=AverPtr[i];
            int alt=AverPtr[i+1];
            for(int i1=als;i1<alt;i1++)
            {
                if(i!=AverInd[i1])
                {   
                    int ib=AverInd[i1];  /// found i'
                    /// now we look for the j' where (i',j') in L
                    int lls=LverPtr[ib];
                    int llt=LverPtr[ib+1];

                    for(int i2=lls;i2<llt;i2++)
                    {
                        
                        if(LverInd[i2].tail-LNS!=j)
                        {
                            int jb=LverInd[i2].tail-LNS; /// this a candidate j'
                            /// now we check whether this j' has an edge with j
                            int jbs=BverPtr[jb];
                            int jbt=BverPtr[jb+1];
                              
                            for(int i3=jbs;i3<jbt;i3++)
                            {    
                                if(BverInd[i3]==j) /// We found a sqaure
                                {
                                    count++;
                                }
                            }    
                        }
                    }
                }
            }   
        }

        prefSum[i+1]=count;
    }
    
    prefSum[0]=0;
    for(int i=2;i<LNS+1;i++)
        prefSum[i]=prefSum[i]+prefSum[i-1];

    int size=prefSum[LNS];
    
	double* si=new double[size];
    double* sj=new double[size];
    
    ///////////////////////////////////////////
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<LNS;i++)
    {
        int index=prefSum[i];
        int ls=LverPtr[i];
        int lt=LverPtr[i+1];
        for(int k=ls; k<lt;k++)
        {
               int j=LverInd[k].tail-LNS; // adjusting the j indices
                 /// now (i,j) is an edge of L now we find i' in A
                 int als=AverPtr[i];
                 int alt=AverPtr[i+1];
                 for(int i1=als;i1<alt;i1++)
                 {
                    if(i!=AverInd[i1])
                    {   
                        int ib=AverInd[i1];  /// found i'
                        /// now we look for the j' where (i',j') in L
                        int lls=LverPtr[ib];
                        int llt=LverPtr[ib+1];

                        for(int i2=lls;i2<llt;i2++)
                        {
                         if(LverInd[i2].tail-LNS!=j)
                         {
                              int jb=LverInd[i2].tail-LNS; /// this a candidate j'
                              /// now we check whether this j' has an edge with j
                              int jbs=BverPtr[jb];
                              int jbt=BverPtr[jb+1];
                              
                              for(int i3=jbs;i3<jbt;i3++)
                              {    
                                  if(BverInd[i3]==j) /// We found a sqaure
                                  {
                                     si[index]=k+1;      //// +1 for making it one based
                                     sj[index]=i2+1;     //// +1 for making it one based
                                     index++;
                                  }
                              }    
                            }
                        }
                        }
                 }
            }
       }
       
    double* sw=new double[size];
    
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0; i < size; i++)
        sw[i]=1;

    graph* G = (graph *) malloc (sizeof(graph));
    create_graph(G,LNE,LNE,size,si,sj,sw);     //// create graph accepts one based 
    
	delete[] si;
    delete[] sj;
    delete[] sw;
	delete[] prefSum;

    CRS_Mat *M;
    M=extractCRSMatrix(G);
	
	free(G->edgeListPtrs);
    free(G->edgeList);
	free(G);
    
    return M;

}

////////////////////////////////// C codes for the mkl function ////////

double cdot(int size, double* x, double* y)
{
    int nt;
    #pragma omp parallel
    {
        #pragma omp master
		nt=omp_get_num_threads();
    }

    double* locSums=new double[nt];
    double res=0;

    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<nt;i++)
        locSums[i]=0.0;

    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<size;i++)
	{
	    int tid=omp_get_thread_num();
        if(x[i]!=0. && y[i]!=0.)
            locSums[tid]+=x[i]*y[i];

    }

    for(int i=0;i<nt;i++)
        res+=locSums[i];

	delete[] locSums;

    return res;
}

void cmatVec(int rows, int cols, double* vc, int* jc, int* ic, double* v, double* res)
{
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<rows;i++)
    {
        double* sum=&res[i];
        int s=ic[i]-1;
        int t=ic[i+1]-1;
        for(int j=s;j<t;j++)
        {
            int ind=jc[j]-1;
            *sum=*sum+vc[j]*v[ind];
        }
    }
}

void caxpy(int row, double alpha, double* x, double* y)
{
    if(alpha!=0)
    {        
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for(int i=0;i<row;i++)
        {    
            int t=x[i];
            if(t!=0)
                y[i]+=alpha*t;
        }
    }    
}
/////////////////////////////////////////////////////////////////////// 

double bipartite_match(double* ind, int* Mate, int* HeaviestPointer, double* li, double* lj, double* w, graph* G)
{
    //cout<<"Start Approx"<<endl;
    double ns=G->sVertices;
    double nt=G->numVertices-ns;
    double ne=G->numEdges;
    //update_weight(G,li,lj,w);
    //ApproxEdgeWeightedMatching(ind,G);
    return ApproxEdgeWeightedMatching(ind,Mate,HeaviestPointer,ns,nt,ne,li,lj,w);
    //cout<<"End Approx"<<endl;
    
}

double bipartite_match(double* ind, 
        int* Mate, int* HeaviestPointer, int *Q1, int *Q2, double* w, 
        double* wperm, graph* G)
{
    int NE=G->numEdges;
    edge *verInd=G->edgeList;
   
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<2*NE;i++) {
        if(i<NE) {
            verInd[i].weight=w[i];          
        } else {
            verInd[i].weight=w[(int)wperm[i-NE]];
        }
     }

    return ApproxEdgeWeightedMatching(ind, Mate, HeaviestPointer, Q1, Q2, G, w,NULL,NULL);
}

double bipartite_match(double* ind, 
        int* Mate, int* HeaviestPointer, double* w, 
        double* wperm, graph* G)
{
    return bipartite_match(ind, Mate, HeaviestPointer, 
        NULL, NULL, w, wperm, G);
}


/** 
 * Determine a single permutation that has the effect of transposing
 * the value array for a structurally symmetric sparse matrix.
 * 
 * @param S the matrix S
 * @return perm an array to statically transpose the elements of 
 *   the value array corresponding to S
 */ 
int* build_perm(CRS_Mat* S) {
    //////////////// creating the permutation arrays//////////////
    int size=S->nrow();
    int snz=S->nnz();
    
    // we can just load the matrix up in one go, because
    // it is structurally symmetric, we don't have to 
    // do any analysis
    int *ptr = S->rowPtr();
    int *ind = S->colInd(); // S is one based, arg.
    
    int *newptr = new int[size+1];
    int* perm=new int[snz];
    
    // newptr is a set of pointers for the transposed
    // set of data, it just happens to be the same as
    // ptr because of the structural symmetry
    
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for (int row=0; row<size; ++row) {
        newptr[row] = ptr[row]-1;
    }
    
    for (int row=0; row<size; ++row) {
        int start=ptr[row]-1;
        int end=ptr[row+1]-1;
        for (int nzi = start; nzi < end; ++nzi) {
            int col=ind[nzi]-1;
            perm[newptr[col]++] = nzi;
        }
    }
        
    delete[] newptr;

    return perm;
}


netalign_parameters::netalign_parameters() 
: problemname(NULL), alpha(1.0), beta(2.0),
gamma(0.99), maxiter(100), alg("mp"), dampingtype(2),
batchrounding(4), verbose(true), finalize(true), quiet(false),
approx(true),limitthreads(false),chunk(1000),outFile(NULL)
{}
    
void netalign_parameters::usage() {
    // TODO finish this
    cout << "Coming soon!" <<endl;
}

bool netalign_parameters::parse(int argc, char *argv[]) {
    static struct option long_options[] = 
    {
        /* These options don't take extra arguments */
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {"nofinalize", no_argument, NULL, 'f'},
        {"quiet", no_argument, NULL, 'q'},
        {"exact", no_argument, NULL, 'e'},
        {"limitthreads", no_argument, NULL, 'l'},
        /* These do */
        {"alpha", required_argument, NULL, 'a'},
        {"beta", required_argument, NULL, 'b'},
        {"gamma", required_argument, NULL, 'g'},
        {"maxiter", required_argument, NULL, 'n'},
        {"alg", required_argument, NULL, 'm'},
        {"damping", required_argument, NULL, 'd'},
        {"batch", required_argument, NULL, 'r'},
        {"chunk", required_argument, NULL, 'c'},
	{"output", required_argument, NULL, 'o'},
        {NULL, no_argument, NULL, 0}
    };
    static const char *opt_string = "qvhfela:b:g:n:m:d:r:c:o:";
    
    
    int opt, longindex;
    opt = getopt_long(argc, argv, opt_string, long_options, &longindex);
    while (opt != -1) {
        switch (opt) {
            case 'v': verbose = true; break;
            case 'h': usage(); return false; break;
            case 'q': quiet = true; break;
            case 'f': finalize = false; break;
            case 'e': approx = false; break;
            case 'l': limitthreads = true; break;
	                
            case 'a': 
                alpha = atof(optarg);
                if (alpha < 0.) {
                    cerr << "alpha must be non-negative, but alpha = " 
                         << alpha << " < 0." << endl;
                    return false;
                }
                break;
                
            case 'b': 
                beta = atof(optarg);
                if (beta < 0.) {
                    cerr << "beta must be non-negative, but beta = " 
                         << beta << " < 0." << endl;
                    return false;
                }
                break;
                
            case 'g': 
                gamma = atof(optarg);
                if (gamma < 0. || gamma > 1.) {
                    cerr << "gamma must be between 0 and 1 but gamma = "
                         << gamma << "." << endl;
                    return false;
                }
                break;
                
            case 'c':
                chunk = atoi(optarg);
                if (chunk <= 0) {
                    cerr << "chunk must be positive, but chunk = "
                         << chunk << " <= 0" << endl;
                    return false;
                }
                break;
                
            case 'n':
                maxiter = atoi(optarg);
                if (maxiter <= 0) {
                    cerr << "maxiter must be positive, but maxiter = "
                         << maxiter << " <= 0" << endl;
                    return false;
                }
                break;
                
            case 'm':
                alg = optarg;
                break;
                
            case 'd':
                dampingtype = atoi(optarg);
                if (dampingtype <= 0 || dampingtype > 3) {
                    cerr << "damping must 1, 2, or 3, but damping = "
                         << dampingtype << "." << endl;
                    return false;
                }
                break;
                
            case 'r':
                batchrounding = atoi(optarg);
                if (batchrounding < 0) {
                    cerr << "batch must be non-negative, but batch = "
                         << batchrounding << " <= 0" << endl;
                    return false;
                }
                break;
             
	     case 'o':
                outFile = optarg;
                cout<<outFile<<"yayy"<<endl;
		if (outFile == NULL) {
                    cerr << "Output file should be specified, but file ="
                         << "NULL" << endl;
                    return false;
                }
                break; 

            default:
                cerr << "unknown argument" << endl;
                return false;   
                break;                 
        }
        opt = getopt_long(argc, argv, opt_string, long_options, &longindex);
    }
    if (argc - optind != 1) {
        cerr << "problem name not specified.  Exiting." << endl;
        return false;
    } else {
        problemname = argv[optind];
    }
    
    if (verbose && !quiet) {
        // print options
        cout << "finalize: " << finalize << endl;
        cout << "alpha: " << alpha << endl;
        cout << "beta: " << beta << endl;
        cout << "gamma: " << gamma << endl;
        cout << "maxiter: " << maxiter << endl;
        cout << "algorithm: " << alg << endl;
        cout << "damping: " << dampingtype << endl;
        cout << "batch: " << batchrounding << endl;
        cout << "approx: " << approx << endl;
        cout << "limitthreads: " << limitthreads << endl;
        cout << "problem: " << problemname << endl;
    }
    
    return true;
}

/** Compute the overlap for a given solution. 
 * 
 * @return the overlap
 * @param x the indicator vector for a solution
 * @param S the matrix of squares
 * @param temp a vector of length S->nrow,
 *   if this is NULL, then it is automatically allocated
 */
double evaluate_overlap(double *x, CRS_Mat *S, double *temp) 
{
    int n = S->nrow();
    int m= S->ncol();
    bool owntemp = false;
    if (temp == NULL) {
        owntemp = true;
        temp = new double[n];
    }
    
    //mkl_dcsrgemv(&trans, &n, S->values(), S->rowPtr(), S->colInd(), x, temp); //S*x
    cmatVec(n, m, S->values(), S->colInd(),S->rowPtr(), x, temp); //S*x

    double overlap=cdot(n, x, temp)/2.;
    
    if (owntemp) {
        delete[] temp;
    }
    
    return overlap;
}

double evaluate_weight(int n, double *x, double *w) {
    return cdot(n, x, w);
}

double evaluate_objective(double alpha, double beta, 
            double *x, CRS_Mat *S, double *w, double *temp) {
    int n = S->nrow();
    double overlap = evaluate_overlap(x, S, temp);
    double weight = evaluate_weight(n, x, w);
    return alpha*weight + beta*overlap;
}

void netAlign(int argc, char** params)
{
    netalign_parameters opts;
    if (!opts.parse(argc, params)) {
        return;
    }
    
    string basename = opts.problemname; // get the root name for the input matrices
    string Afilename = basename + "-A.mtx";
    string Bfilename = basename + "-B.mtx";
    string Lfilename = basename + "-L.mtx";

    double time1=0.,time2=0.,time3=0.,time4=0.,timet=0.;
    double objective=-1000000;
      
    int nthreads;
    #pragma omp parallel
	{
    	#pragma omp master
		nthreads=omp_get_num_threads();
	}
     
    time1=timer(); 
    CRS_Mat* A=new CRS_Mat(Afilename.c_str());
    CRS_Mat* B=new CRS_Mat(Bfilename.c_str());
    Cord_Mat* LM=new Cord_Mat(Lfilename.c_str());
    
    if (opts.verbose) {
        cout << "Finished reading data. " << timer() - time1 << " sec." <<" Threads: "<<nthreads <<endl;
    }
    
    graph* L = new graph;
    graph* LP= new graph;
             
    double* s=new double[LM->nnz()];
    double* t=new double[LM->nnz()];
    double* w=new double[LM->nnz()];
    double* wperm=new double[LM->nnz()];
    
    int* ic=LM->rowPtr();
    int*jc=LM->colInd();
    double* vc=LM->values();
    int nrow=LM->nrow();
    int ncol=LM->ncol();
    int nnz=LM->nnz();

    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<nnz;i++) {
        wperm[i]=i+1;
        s[i]=ic[i];
        t[i]=jc[i];
        w[i]=vc[i];
    }
    
    delete LM;

    timet=timer();
    create_graph(L,nrow,ncol,nnz,s,t,w);
    if (opts.verbose) {
        cout << "Time to create L: " << timer() - timet << endl;
        cout << "nonzeros in L: " << L->numEdges << endl;
    }

    int nz=L->numEdges;
    int ns=L->sVertices;
    edge* verInd=L->edgeList;
    
    /******************* creating weight permutation***********/
    create_graph(LP,nrow,ncol,nnz,s,t,wperm);
    edge* verIndp=LP->edgeList;
    
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=nz;i<2*nz;i++) {
        wperm[i-nz]=verIndp[i].weight-1;
    }
    
    free(LP->edgeListPtrs);
  	free(LP->edgeList);
    delete LP;
    /**********************Creating vectors***********************/
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<nz;i++)
    {
        //cout<<"Iter: "<<i<<endl;
        s[i]=verInd[i].head+1;
        t[i]=verInd[i].tail-ns+1;
        w[i]=verInd[i].weight;
        //cout<<"Iter: "<<i<<endl;
    }
    
    Vec weight(w,nz);
    Vec li(s,nz);
    Vec lj(t,nz);
    

    /**********************************************************/
    timet=timer();
    CRS_Mat* S=createSquareMatrix(A,B,L);
    if (opts.verbose) {
        cout << "Time to create S: " << timer() - timet << endl;
        cout << "nonzeros in S: " << S->nnz() << endl;
    }
    //CRS_Mat* S=new CRS_Mat("S.mtx");
    S->convertToOneBased();
    //cout<<A->ndiag()<<" "<<B->ndiag()<<" "<<L->numEdges<<" "<<S->nnz()<<endl;    
    
        
    //cout<<"Before Calling: "<<A->nnz()<<" "<<B->nnz()<<" "<<L->numEdges<<endl;
    double* ind=NULL;
    double* mbest=NULL;
    if (strcmp(opts.alg, "mp") == 0) {   
        time2=timer();
        mbest=netAlignMP(S, weight, L, li, lj, wperm, opts, &objective);
		time3=timer();
        ind = new double[nz];
		if (opts.finalize == true) {
            exact_match(ind, li.values(), lj.values(), mbest, L);
        }
        time4 = timer();
    } else if (strcmp(opts.alg, "mptask") == 0) {   
        time2=timer();
        mbest=netAlignMPTasked(S, weight, L, li, lj, wperm, opts, &objective);
        time3=timer();
        ind = new double[nz];
        if (opts.finalize == true) {
            exact_match(ind, li.values(), lj.values(), mbest, L);
        }
        time4 = timer();
    } else if (strcmp(opts.alg, "mr") == 0) {
        time2=timer();
        mbest=netAlignMR(S, weight, L, li, lj, wperm, opts, &objective);
        time3=timer();
        ind = new double[nz];
        if (opts.finalize == true) {
            exact_match(ind, li.values(), lj.values(), mbest, L);
        }
        time4 = time3;
    }
    
    assert(ind != NULL);
    
    //cout<<"Before Calling: "<<A->nnz()<<" "<<B->nnz()<<" "<<L->numEdges<<endl; 
    
    if (opts.finalize == true) {
        double overlap = evaluate_overlap(ind, S, NULL);
        double matchweight = evaluate_weight(nz, ind, weight.values());
        objective = opts.alpha*matchweight + opts.beta*overlap;
        cout<<"Objective: "<<objective<<endl;
        cout<<"Weight: " << matchweight << endl;
        cout<<"Overlaps: "<<overlap<<endl;
    }
        
    cout<<"Set Up Time: "<<time2-time1<<endl;
    cout<<"Solve Time: "<<time3-time2<<endl;
    
    if (opts.finalize == true) {
        cout << "Finalize Time: " << time4-time3 << endl;
    }



    if(opts.outFile != NULL)
    {
    	ofstream myfile;
		myfile.open(opts.outFile);
		     
		if(opts.finalize == true)
		{
			for(int i=0;i<nz;i++)
				myfile<<s[i]<<" "<<t[i]<<" "<<ind[i]<<endl;
		}
		else
		{
			for(int i=0;i<nz;i++)
				myfile<<s[i]<<" "<<t[i]<<" "<<mbest[i]<<endl;
		}
		
    }

/****************************** Clean Up **********************/    
   	
    delete[] wperm;
	delete[] ind;
	delete[] mbest;
    delete[] weight.values();
    delete[] li.values();
    delete[] lj.values();

	delete A;
	delete B;
    delete S;

	delete[] s;
	delete[] t;
	delete[] w;		

	free(L->edgeListPtrs);
  	free(L->edgeList);
	delete L;
}


arma::sp_mat netAlign_arma(arma::sp_mat A_mat, arma::sp_mat B_mat, arma::sp_mat L_mat,
							double alpha = 1.0,
							double beta = 1.0,
							double gamma = 0.99,
							int maxiter = 100,
							bool finalize = false) {
								
    netalign_parameters opts;
	opts.alpha = alpha;
	opts.beta = beta;
	opts.gamma = gamma;
	opts.maxiter = maxiter;
	opts.finalize = finalize;
	opts.alg = "mr";

	omp_set_num_threads(8);
	
    double time1=0.,time2=0.,time3=0.,time4=0.,timet=0.;
    double objective=-1000000;
      
	
    time1=timer(); 

    CRS_Mat* A=new CRS_Mat(A_mat);
    CRS_Mat* B=new CRS_Mat(B_mat);
    Cord_Mat* LM=new Cord_Mat(L_mat);
    
    
    graph* L = new graph;
    graph* LP= new graph;
             
    double* s=new double[LM->nnz()];
    double* t=new double[LM->nnz()];
    double* w=new double[LM->nnz()];
    double* wperm=new double[LM->nnz()];

    
    int* ic=LM->rowPtr();
    int*jc=LM->colInd();
    double* vc=LM->values();
    int nrow=LM->nrow();
    int ncol=LM->ncol();
    int nnz=LM->nnz();
	
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<nnz;i++) {
        wperm[i]=i+1;
        s[i]=ic[i];
        t[i]=jc[i];
        w[i]=vc[i];
        
    }
    
   

    delete LM;

    timet=timer();
    create_graph(L,nrow,ncol,nnz,s,t,w);
    if (opts.verbose) {
        cout << "Time to create L: " << timer() - timet << endl;
        cout << "nonzeros in L: " << L->numEdges << endl;
    }

    int nz=L->numEdges;
    int ns=L->sVertices;
    edge* verInd=L->edgeList;

    
    
    /******************* creating weight permutation***********/
    create_graph(LP,nrow,ncol,nnz,s,t,wperm);
    edge* verIndp=LP->edgeList;
    
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=nz;i<2*nz;i++) {
        wperm[i-nz]=verIndp[i].weight-1;
    }
    
    free(LP->edgeListPtrs);
  	free(LP->edgeList);
    delete LP;


    /**********************Creating vectors***********************/
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<nz;i++)
    {
        //cout<<"Iter: "<<i<<endl;
        s[i]=verInd[i].head+1;
        t[i]=verInd[i].tail-ns+1;
        w[i]=verInd[i].weight;
        //cout<<"Iter: "<<i<<endl;
    }
    
    Vec weight(w,nz);
    Vec li(s,nz);
    Vec lj(t,nz);  


    /**********************************************************/
    timet=timer();
    CRS_Mat* S=createSquareMatrix(A,B,L);
    if (opts.verbose) {
        cout << "Time to create S: " << timer() - timet << endl;
        cout << "nonzeros in S: " << S->nnz() << endl;
    }

    //CRS_Mat* S=new CRS_Mat("S.mtx");
    S->convertToOneBased();
    //cout<<A->ndiag()<<" "<<B->ndiag()<<" "<<L->numEdges<<" "<<S->nnz()<<endl;    
    
        
    //cout<<"Before Calling: "<<A->nnz()<<" "<<B->nnz()<<" "<<L->numEdges<<endl;
    double* ind=NULL;
    double* mbest=NULL;
    if (strcmp(opts.alg, "mp") == 0) {   
        time2=timer();
        mbest=netAlignMP(S, weight, L, li, lj, wperm, opts, &objective);
		time3=timer();
        ind = new double[nz];
		if (opts.finalize == true) {
            exact_match(ind, li.values(), lj.values(), mbest, L);
        }
        time4 = timer();
    } else if (strcmp(opts.alg, "mptask") == 0) {   
        time2=timer();
        mbest=netAlignMPTasked(S, weight, L, li, lj, wperm, opts, &objective);
        time3=timer();
        ind = new double[nz];
        if (opts.finalize == true) {
            exact_match(ind, li.values(), lj.values(), mbest, L);
        }
        time4 = timer();
    } else if (strcmp(opts.alg, "mr") == 0) {
        time2=timer();
        mbest=netAlignMR(S, weight, L, li, lj, wperm, opts, &objective);
        time3=timer();
        ind = new double[nz];
        if (opts.finalize == true) {
            exact_match(ind, li.values(), lj.values(), mbest, L);
        }
        time4 = time3;
    }
    
    assert(ind != NULL);
    
    //cout<<"Before Calling: "<<A->nnz()<<" "<<B->nnz()<<" "<<L->numEdges<<endl; 
    
    if (opts.finalize == true) {
        double overlap = evaluate_overlap(ind, S, NULL);
        double matchweight = evaluate_weight(nz, ind, weight.values());
        objective = opts.alpha*matchweight + opts.beta*overlap;
        cout<<"Objective: "<<objective<<endl;
        cout<<"Weight: " << matchweight << endl;
        cout<<"Overlaps: "<<overlap<<endl;
    }
        
    cout<<"Set Up Time: "<<time2-time1<<endl;
    cout<<"Solve Time: "<<time3-time2<<endl;
    
    if (opts.finalize == true) {
        cout << "Finalize Time: " << time4-time3 << endl;
    }

/*
	for(int i=0;i<nz;i++) {
		printf("%d- %d %d %f\n", i, s[i], t[i], ind[i]);
	}*/
	
	int row, col;
	double val;
	arma::sp_mat matched(A_mat.n_rows, B_mat.n_rows);
	for(int i=0;i<nz;i++) {
		row = (int)s[i] - 1;
		col = (int)t[i] - 1;
		val = ind[i] * mbest[i];
		if(ind[i] > 0) {
			matched(row, col) = val; //ind[i];
		}
	}
/****************************** Clean Up **********************/    
   	
    delete[] wperm;
	delete[] ind;
	delete[] mbest;
    delete[] weight.values();
    delete[] li.values();
    delete[] lj.values();

	delete A;
	delete B;
    delete S;

    delete[] s;
    delete[] t;
    delete[] w;
    
	free(L->edgeListPtrs);
  	free(L->edgeList);
	delete L;
	
	return matched;
}



