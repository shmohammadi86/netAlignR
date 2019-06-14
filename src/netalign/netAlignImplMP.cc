#include "netAlignKernel.h"
using namespace std;

#include <string>
#include <vector>

//#define SUITOR 1

void print_messages(int size, double *yt, double *zt) {
    for (int i=0; i<size; ++i) {
        cout << "yt[" << i << "] = " << yt[i] << ";\n";
    }
    for (int i=0; i<size; ++i) {
        cout << "zt[" << i << "] = " << zt[i] << ";\n";
    }
}

/**
 * batched rounding
 */

struct batch_rounding {
    // the best_solution is enough memory to hold the best
    // solution
    
    // batch storage
    double ** buffer;
    double* bestM;
    double bestObj;
    int nvectors;
    int size;
    int numV;
    int numS;
    int curr;
    int nprobs; // the number of problems to solve at once
    bool detached_best;
    
    // rounded objective values
    std::vector<double> objTVec;
    double *objT;
    
    // input from the main function
    double* li;
    double* lj;
    double* w;
    double* wperm;
    double alpha, beta;
    int matchtype;
    CRS_Mat* S;
    graph* L;

    // storage for matching
    int** Mate;
    #ifdef SUITOR
        double** ws;
        omp_lock_t** locks;
    #else
        int** HeaviestPointer;
        int** Q1;
        int** Q2;
        int** Visited;
        double** RMax;
    #endif

    // storage for rounding
    double** x;
    double** res;
    
    /**
     * @param nvectors_ the number of vectors to save before rounding
     * all other parameters are copied from the netAlignMP procedure.
     */
    batch_rounding(int nvectors_, 
        double* li_, double* lj_, graph* L_, 
        CRS_Mat* S_, double* w_, double* wperm_, 
        double alpha_, double beta_,  int matchtype_)
    : nvectors(nvectors_),
      detached_best(false), objTVec(nvectors),
      li(li_), lj(lj_), 
      w(w_),        
      wperm(wperm_), 
      alpha(alpha_),  beta(beta_),   matchtype(matchtype_), S(S_), L(L_)
    {
        curr = 0;
        bestObj = 0;
        //size = S->nrow();
        size = L->numEdges;
        
        // determine the maximum number of problems
        // that we'll solve simultaneously
        int maxthreads = omp_get_max_threads();
        nprobs = maxthreads;
        if (nvectors < maxthreads) {
            nprobs = nvectors;
        }

        //cout<<"Batch Memory"<<endl;
        bestM = new double[size];
        buffer = new double*[nvectors];
        objT = &objTVec[0];

        Mate=new int*[nprobs];
        #ifdef SUITOR
            ws=new double*[nprobs];
            locks=new omp_lock_t*[nprobs]; /// For parallel
        #else
            HeaviestPointer=new int*[nprobs];
            Q1=new int*[nprobs];
            Q2=new int*[nprobs];
            Visited=new int*[nprobs];
            RMax=new double*[nprobs];
        #endif

        x=new double*[nprobs];
        res=new double*[nprobs];
        
        numV = L->numVertices;
        numS = L->sVertices;
        int *verPtr = L->edgeListPtrs;
	
	    for (int i=0;i<nvectors;i++) {
            buffer[i]=new double[2*size];

            //First touch
            #ifdef DYNSCHED
                #pragma omp parallel for schedule(dynamic, CHUNK)
            #else
                #pragma omp parallel for schedule(static, CHUNK)
            #endif
            for(int j=0;j<numV;j++)
            {
                int s=verPtr[i];
                int t=verPtr[i+1];
                for(int k=s;k<t;k++)
                    buffer[i][k]=0.;
            }
        }
        
	 
	   for (int i=0; i<nprobs; ++i) {
            Mate[i]=new int[numV];
            x[i]=new double[size];
            res[i]=new double[size];
            
            #ifdef SUITOR
                ws[i]=new double[numV];
                locks[i]=new omp_lock_t[numV]; // For Parallel
                
                // First Touch
                #ifdef DYNSCHED
                    #pragma omp parallel for schedule(dynamic, CHUNK)
                #else
                    #pragma omp parallel for schedule(static, CHUNK)
                #endif
                for(int j=0;j<numV;j++)
                {
                    Mate[i][j]=-1;
                    ws[i][j]=0.0;
                    omp_init_lock(&locks[i][j]);
                }
            #else
                HeaviestPointer[i]=new int[numV];
                Q1[i]=new int[numV];
                Q2[i]=new int[numV];
                Visited[i]=new int[numV-numS];
                RMax[i]=new double[numS];
                //First Touch
                #ifdef DYNSCHED
                    #pragma omp parallel for schedule(dynamic, CHUNK)
                #else
                    #pragma omp parallel for schedule(static, CHUNK)
                #endif
                for(int j=0;j<numV;j++)
                {
                    HeaviestPointer[i][j]=-1;
                    Mate[i][j]=-1;
                    Q1[i][j]=-1;
                    Q2[i][j]=-1;
                    if(j<numS)
                        RMax[i][j]=-1;
                    if(j<(numV-numS))
                        Visited[i][j]=-1;
                }
            #endif
            
            // First Touch    
            #ifdef DYNSCHED
                #pragma omp parallel for schedule(dynamic, CHUNK)
            #else
                #pragma omp parallel for schedule(static, CHUNK)
            #endif
            for(int j=0;j<size;j++)
            {
                x[i][j]=0;
                res[i][j]=-1;
            }
        }
        //cout<<"Batch Memory done"<<endl;
        
    }
    
    ~batch_rounding() {
        for (int i=0;i<nvectors;i++) {
            delete[] buffer[i];
        }
        for (int i=0; i<nprobs; ++i) {
            delete[] Mate[i];
            
            #ifdef SUITOR
                delete[] ws[i];
                delete[] locks[i];
            #else
                delete[] HeaviestPointer[i];
                delete[] Q1[i];
                delete[] Q2[i];
                delete[] RMax[i];
                delete[] Visited[i];
            #endif
            
            delete[] x[i];
            delete[] res[i];
        }
        if (!detached_best) {
            delete[] bestM;
        }
        
        #ifdef SUITOR
            delete[] ws;
            delete[] locks;
        #else
            delete[] HeaviestPointer;
            delete[] Q1;
            delete[] Q2;
            delete[] RMax;
            delete[] Visited;
        #endif
        
        delete[] Mate;
        delete[] x;
        delete[] res;
        delete[] buffer;
    }
    // return back a pointer to our own memory, but do
    // any rounding that needs to be done.
    double *best_solution() {
        round();
        return bestM;
    } 
    /** Detach the responsibility for freeing the best solution
     * 
     * Call this routine if you want to use the solution bestM
     * after this class is destroyed.
     */
    void detach_best() {
        detached_best = true;
    }
    // the best objective value we've seen so far.
    double best_objective() {
        round();
        return bestObj;
    }
    // this just adds a new heuristic, if it exceeds the buffer
    // count, it should round and only save the best.
    void add_heuristic(double *m) 
    {
        int *verPtr = L->edgeListPtrs;
        
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for(int i=0;i<numV;i++)
        {
            int s=verPtr[i];
            int t=verPtr[i+1];
            for(int j=s;j<t;j++)
            {
                if(j<size) {
                    buffer[curr][j]=m[j];
                } else {
                    buffer[curr][j]=m[(int)wperm[j-size]];}
            }
        }

        /*for(int i=0;i<2*size;i++) {
            if(i<size) {
                buffer[curr][i]=m[i];    
            } else {
                buffer[curr][i]=m[(int)wperm[i-size]];}
        }*/
        
        curr++;

        // buffer is full do the batch rounding
        if(curr==nvectors) {
            round();
        }
    }
    
    void round() {
        // early out if we can
        if (curr == 0) {
            return;
        }
        
        // TODO allocate extra threads to matching
        // based on availability
        int nthreads=omp_get_max_threads(); // Maximum Available Threads
        int ntasks = curr;
        if (nprobs < ntasks) {
            ntasks = nprobs;
        }
        nthreads=nthreads/(ntasks); // each nested thread can have nthreads maximum
        if (nthreads < 1) { nthreads = 1; }
        
        int *jc = S->colInd();
        int *ic = S->rowPtr();
    
        //cout<<"tasks: "<<ntasks<<" "<<nthreads<<" "<<nprobs<<endl;
        #pragma omp parallel num_threads(ntasks)
        {
            #pragma omp single
            {
                for(int i=0;i<curr;i++)
                {
                    #pragma omp task
                    {
                        int tid=omp_get_thread_num();                        
                        assert(tid < nprobs);

                        // set the number of threads within a matching task
                        omp_set_num_threads(nthreads);         
                        if (matchtype == -1) {
                            #ifdef SUITOR 
                            ApproxEdgeWeightedMatching(x[tid], 
                                Mate[tid], ws[tid],locks[tid], 
                                L, buffer[tid]);
                            #else
                            //cout<<"------------"<<endl;
                            ApproxEdgeWeightedMatching(x[tid],
                                Mate[tid], HeaviestPointer[tid],
                                Q1[tid], Q2[tid], L, buffer[tid], RMax[tid], Visited[tid]);
                            #endif
                        } else {
                            exact_match(x[tid], li, lj, buffer[tid], L);
                        }
                        
                        int overlaps = 0;
                        double weight = 0.;
                        
                        #ifdef DYNSCHED
                            #pragma omp parallel for schedule(dynamic,CHUNK) \
                            reduction(+:overlaps) reduction(+:weight)
                        #else
                            #pragma omp parallel for schedule(static, CHUNK) \
                            reduction(+:overlaps) reduction(+:weight)
                        #endif
                        for (int row=0; row<size; ++row) {
                            weight += x[tid][row]*w[row];
                            for (int nzi=ic[row]-1; nzi<ic[row+1]-1; ++nzi) {
                                overlaps += x[tid][row]*x[tid][jc[nzi]-1];
                            }
                        }
                        overlaps *= 0.5;
                        objT[i]=alpha*weight + beta*overlaps;                        
                    }
                }
            }
        }
        
        int ind=-1;
        // Rounding is done. Now get the best message
        for(int i=0;i<curr;i++) {
            if(objT[i]>bestObj) {
                bestObj=objT[i];
                ind=i;
            }
        }
        
        if(ind!=-1) {  // So we got our new best solution
            copy(size,buffer[ind],bestM);
        }

        curr = 0;  // reset the index to the start of the buffer
    }
};


/** Given a sparse matrix, compute the othermax function:
 * othermax(A)_{i,j} = 
 *   case 1: max(A(i,:)) unless j = argmax(A(i,:))
 *   case 2: max(A(i, all but j))
 */
struct other_max_computer {
    int rows, size;
    int nthread;    
    std::vector<int> rowptrVec, permVec, splitsVec;
    int *rowptr, *perm, *splits;

    /**
     * NOTE THAT THIS TAKES one-based indices
     * 
     * @param l1 a one-based index for the first index of the matrix
     */
    other_max_computer(double *l1, int rows_, int size_, int nthread_) 
    : rows(rows_), size(size_), nthread(nthread_), rowptrVec(rows+1,0), permVec(size),
         splitsVec(nthread+1)
    {
        // convert the input into a CSR array and store the
        // permutation of the order 
        rowptr = &rowptrVec[0];
        perm = &permVec[0];
        splits = &splitsVec[0];
        
        // build up the pointer array, note that rowptrVec was
        // initialized to zero
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for (int i=0; i<size; ++i) {
            int r = (int)(l1[i]-1.);
            #pragma omp atomic 
            rowptr[r+1]++;
        }
        
        // TODO parallelize this cumsum as in
        // http://www.pdc.kth.se/education/tutorials/summer-school/programming-exercises-on-openmp/solutions/part3a-cumulative-sum
        int cumsum = 0;
        for (int i=0; i<rows+1; ++i) {
            cumsum += rowptr[i];
            rowptr[i] = cumsum;
        }
        assert(rowptr[rows] == size);
        
        // need atomic capture
        //#pragma omp parallel for shared(rowptr,l1)
        for (int i=0; i<size; ++i) {
            int r = (int)(l1[i]-1.);
            int loc;
            // get the location
            // #pragma omp atomic capture 
            {
                loc = rowptr[r];
                rowptr[r]++;
            }
            perm[loc] = i;
        }

        // fix the pointers now and find splits
        int newval = 0;
        double mean = (1.*(double)rows + 1.*(double)size)/(double)(nthread);
        double cur = 0.;
        splits[0] = 0;
        size_t cursplit = 1;
        for (int i=0; i<rows; ++i) {
            // fix the pointers
            int tempval = rowptr[i];
            rowptr[i] = newval;
            newval = tempval; 
            
            // update the nonzeros and find if we should split
            int curnz = newval - rowptr[i];
            cur += 1. + 1.*(double)curnz;
            if (cur > mean) {
                // we have our split
                splits[cursplit] = i+1;
                cursplit++;
                assert(cursplit <= (size_t)nthread);
                cur = 0.;
            }
        }
        assert(cursplit == (size_t)nthread);
        splits[cursplit] = rows;
        assert(newval == size);
    }
    
    /** 
     * Apply the othermax operator directly to a vector
     * othermax = othermax - othermaxfunc(vals);
     */
    void apply_balanced(double *vals, double *othermax) {
        #pragma omp parallel num_threads(nthread)
        {
        int tid = omp_get_thread_num();
        int start = splits[tid], end = splits[tid+1];
        for (int i=start; i<end; ++i) {
            int maxind = perm[rowptr[i]];
            for (int nzi=rowptr[i]+1; nzi < rowptr[i+1]; ++nzi) {
                int pi = perm[nzi];
                if (vals[pi] > vals[maxind]) {
                    maxind = pi;
                }
            }
            double bestval = vals[maxind];
            if (bestval < 0.) {
                bestval = 0.;
            }
            double secondbest = 0.;
            // assign and find the second best
            for (int nzi=rowptr[i]; nzi < rowptr[i+1]; ++nzi) {
                int pi = perm[nzi];
                if (pi != maxind) {
                    // we aren't the max in the row, which 
                    // means this is a candidate for secondmax.
                    if (vals[pi] > secondbest && vals[pi] < bestval) { 
                        secondbest = vals[pi]; 
                    }
                    // but we should get the best val
                    othermax[pi] -= bestval;
                } 
            }
            othermax[maxind] -= secondbest;
        }
        }
    }
    
    
    /** 
     * Apply the othermax operator directly to a vector
     * othermax = othermax - othermaxfunc(vals);
     */
    void apply(double *vals, double *othermax) {
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for (int i=0; i<rows; ++i) {
            int maxind = perm[rowptr[i]];
            for (int nzi=rowptr[i]+1; nzi < rowptr[i+1]; ++nzi) {
                int pi = perm[nzi];
                if (vals[pi] > vals[maxind]) {
                    maxind = pi;
                }
            }
            double bestval = vals[maxind];
            if (bestval < 0.) {
                bestval = 0.;
            }
            double secondbest = 0.;
            // assign and find the second best
            for (int nzi=rowptr[i]; nzi < rowptr[i+1]; ++nzi) {
                int pi = perm[nzi];
                if (pi != maxind) {
                    // we aren't the max in the row, which 
                    // means this is a candidate for secondmax.
                    if (vals[pi] > secondbest && vals[pi] < bestval) { 
                        secondbest = vals[pi]; 
                    }
                    // but we should get the best val
                    othermax[pi] -= bestval;
                } 
            }
            othermax[maxind] -= secondbest;
        }
    }
    
    /** 
     * Apply the othermax operator directly to a vector
     * othermax = othermax - othermaxfunc(vals);
     */
    void compute(double *vals, double *othermax) {
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for (int i=0; i<rows; ++i) {
            int maxind = perm[rowptr[i]];
            for (int nzi=rowptr[i]+1; nzi < rowptr[i+1]; ++nzi) {
                int pi = perm[nzi];
                if (vals[pi] > vals[maxind]) {
                    maxind = pi;
                }
            }
            double bestval = vals[maxind];
            if (bestval < 0.) {
                bestval = 0.;
            }
            double secondbest = 0.;
            // assign and find the second best
            for (int nzi=rowptr[i]; nzi < rowptr[i+1]; ++nzi) {
                int pi = perm[nzi];
                if (pi != maxind) {
                    // we aren't the max in the row, which 
                    // means this is a candidate for secondmax.
                    if (vals[pi] > secondbest && vals[pi] < bestval) { 
                        secondbest = vals[pi]; 
                    }
                    // but we should get the best val
                    othermax[pi] = bestval;
                } 
            }
            othermax[maxind] = secondbest;
        }
    }
};

void compute_split_points(CRS_Mat* A, std::vector<int>& splits, int nsplits) {
    int size = A->nrow();
    int nnz = A->nnz();
    double mean = ((double)size + 2.*(double)nnz)/(double)(nsplits);

    int *ic=A->rowPtr();
    
    assert(nsplits >= 1);
    
    size_t cursplit = 1;
    splits.resize(nsplits+1);
    splits[0] = 0;
    double cur = 0.;
    for (int i=0; i<size; ++i) {
        cur += 1. + 2.*(double)(ic[i+1] - ic[i]);
        if (cur > mean) {
            // we have our split!
            splits[cursplit] = i+1;
            cursplit++;
            cur = 0.;
        }
    }
    assert(cursplit == (size_t)nsplits);
    splits[cursplit] = size;
}

/**
 * @return the best set of matches generated at any step of the code
 * according to the choice of matching
 */
double* netAlignMP(CRS_Mat* S, Vec w, graph* L, Vec li, Vec lj, 
                    double* wperm, netalign_parameters opts, double* objective)
{
    double alpha = opts.alpha;
    double beta = opts.beta;
    double gamma = opts.gamma;
    int iter = opts.maxiter;
    int damping_type = opts.dampingtype;
    
    
    int nthreads;
    #pragma omp parallel
    {
        nthreads=omp_get_num_threads();
    }
    
    assert(S->nrow()==w.length());

    int size=S->nrow();
    int snz=S->nnz();
    int ns=L->sVertices;
    int nt=L->numVertices-ns;
    
    double damping_mult = 1.;

    double* dt=new double[size];
    double* yt=new double[size];
    double* zt=new double[size];
    double* yt1=new double[size];
    double* zt1=new double[size];
    double* dt1=new double[size];
    double* Fvc=new double[snz];
    double* vvc=new double[size];
    double* p=new double[size];
    double* aw=new double[size];
    double* wi=w.values();
    
    int saveIter=opts.batchrounding;
    batch_rounding histM(saveIter, li.values(), lj.values(), L, 
        S, w.values(), wperm, alpha, beta, opts.approx ? -1 : 1);

    other_max_computer omax1(li.values(), ns, size, nthreads);
    other_max_computer omax2(lj.values(), nt, size, nthreads);
    double* vc = new double[size];
    
    int *ic=S->rowPtr(); 
    
    #pragma omp parallel
    {
        #ifdef DYNSCHED
            #pragma omp for schedule(dynamic, CHUNK) nowait
        #else
            #pragma omp for schedule(static, CHUNK) nowait
        #endif 
        for(int i=0;i<size;i++)
            {
                int s=ic[i]-1;
                int t=ic[i+1]-1;
                for(int j=s;j<t;j++)
                    Fvc[j]=0.0;
            }
        
        //for(int i=0;i<snz;i++)
            //Fvc[i]=0.0;
        
        #ifdef DYNSCHED
            #pragma omp for schedule(dynamic, CHUNK) nowait
        #else
            #pragma omp for schedule(static, CHUNK) nowait
        #endif
        for(int i=0;i<size;i++) {
            aw[i]=alpha*wi[i];
            dt[i]=0.0;
            yt[i]=0.0;
            zt[i]=0.0;
            dt1[i]=0.0;
            yt1[i]=0.0;
            zt1[i]=0.0;
            vvc[i]=0.0;
            p[i]=0.0;
        }
    } /// End of parallel region

    CRS_Mat* St1=new CRS_Mat(Fvc,S->colInd(),S->rowPtr(),size,size,snz,false);
    CRS_Mat* St=new CRS_Mat(Fvc,S->colInd(),S->rowPtr(),size,size,snz,false);
    double *stvc=St->values(), *st1vc = St1->values();
   
    
    int *perm = build_perm(S);
    // compute split points in the matrix S
    std::vector<int> spSvec(nthreads+1); // split points in the matrix S
    compute_split_points(S, spSvec, nthreads);
    
    int lastiter = 0;
    double timestart;
    double time0, time1;
    double timepoints[7] = {0};
   
    for(int t=1;t<=iter;t++)
    {
        timestart=timer();
        lastiter = t;
        
        // Swap data from previous iteration
        time0=timer();

        if(t>1) {
            double *tempptr;
            tempptr = dt1; dt1=dt; dt=tempptr;
            tempptr = yt1; yt1=yt; yt=tempptr;
            tempptr = zt1; zt1=zt; zt=tempptr;
            tempptr = st1vc; st1vc=stvc; stvc = tempptr;
        }
        
        time1 = timer(); timepoints[0] += time1-time0; time0=time1;
        
        if (opts.limitthreads) {
            omp_set_num_threads(std::min(20,nthreads));
        }

        // Line 3
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif 
        for(int i=0;i<size;i++)
        {
            int s=ic[i]-1;
            int t=ic[i+1]-1;
            for(int j=s;j<t;j++)
            {
                double val=beta+st1vc[perm[j]];
                            
                if(val<0.) {
                    val=0.;
                } else {
                    if(val>beta) {
                        val=beta;
                    }
                } 
                Fvc[j] = val;
            }
        }
        
        /*for(int i=0;i<snz;i++) {
            double val=beta+st1vc[perm[i]];
                            
            if(val<0.) {
                val=0.;
            } else {
                if(val>beta) {
                    val=beta;
                }
            } 
            Fvc[i] = val;
        }*/
        
        time1 = timer(); timepoints[1] += time1-time0; time0=time1;
        
        //
        // Line 4
        //
        /*#pragma omp parallel shared(ic,Fvc,zt,yt,dt,spS) \
            num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            int start = spS[tid], end = spS[tid+1];
            for (int i=start; i<end; ++i) {
                int h=ic[i]-1;
                int t=ic[i+1]-1;
                double sum = 0.; 
                for(int j=h;j<t;j++) {    
                    sum+=Fvc[j];
                }  ///
                dt[i]=sum; // y[t] = sums, z[t] = sums
                zt[i]=yt[i]=sum+alpha*wi[i];
            }
        }*/
        
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for (int i=0; i<size; i++) {
            int s=ic[i]-1;
            int t=ic[i+1]-1;
            double sum = 0.; 
            for(int j=s;j<t;j++) {    
                sum+=Fvc[j];
            }  ///
            dt[i]=sum; // y[t] = sums, z[t] = sums
            zt[i]=yt[i]=sum+alpha*wi[i];
        }
        
        time1 = timer(); timepoints[2] += time1-time0; time0=time1;
        
        //
        // Line 5
        //
        
        if (opts.limitthreads) {
            omp_set_num_threads(std::min(40,nthreads));
        }
       
        //daxpy(&size, &alpha, w.values(), &incx, yt, &incy);

        if (t>1) {
            omax2.apply(zt1,yt);
        }

        // 
        // Line 6
        // 
        
        //daxpy(&size, &alpha, w.values(), &incx, zt, &incy);

        if(t > 1) {
            omax1.apply(yt1,zt);            
        }
        
        time1 = timer(); timepoints[3] += time1-time0; time0=time1;
        
        if (opts.limitthreads) {
            omp_set_num_threads(std::min(20,nthreads));
        }
        //
        // Line 7
        //
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif                     
        for(int i=0;i<size;i++) {
            vvc[i]=yt[i]+zt[i]-alpha*w(i)-dt[i];
        }
        
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for(int i=0;i<size;i++) {
            int s=ic[i]-1;
            int t=ic[i+1]-1;
            for(int j=s;j<t;j++)
                stvc[j]=vvc[i]-Fvc[j];    
        }
        
        time1 = timer(); timepoints[4] += time1-time0; time0=time1; 
            
        //
        // Damping
        //
        
        assert(damping_type >=1 && damping_type<=3);
        damping_mult *= gamma;

        if(damping_type==1)
        {
            #ifdef DYNSCHED
                #pragma omp parallel for schedule(dynamic, CHUNK) //nowait
            #else
                #pragma omp parallel for schedule(static, CHUNK) //nowait
            #endif
            for(int i=0;i<size;i++) {
                int s=ic[i]-1;
                int t=ic[i+1]-1;
                for(int j=s;j<t;j++)
                    stvc[j]=damping_mult*stvc[j]+(1.0-damping_mult)*st1vc[j];
            }            
            
            //for(int i=0;i<snz;i++)
                //stvc[i]=damping_mult*stvc[i]+(1.0-damping_mult)*st1vc[i];
            
            #ifdef DYNSCHED
                #pragma omp parallel for schedule(dynamic, CHUNK) //nowait
            #else
                #pragma omp parallel for schedule(static, CHUNK) //nowait
            #endif
            for(int i=0;i<size;i++) {
                yt[i]=damping_mult*yt[i]+(1.0-damping_mult)*yt1[i];
                zt[i]=damping_mult*zt[i]+(1.0-damping_mult)*zt1[i];
            }
        }
        else
        {
            #ifdef DYNSCHED
                #pragma omp parallel for schedule(dynamic, CHUNK) //nowait
            #else
                #pragma omp parallel for schedule(static, CHUNK) //nowait
            #endif
            for(int i=0;i<size;i++)
                p[i]=yt1[i]+zt1[i]-aw[i]+dt1[i];
            
            if(damping_type==2) {
                #ifdef DYNSCHED
                    #pragma omp parallel for schedule(dynamic, CHUNK) //nowait
                #else
                    #pragma omp parallel for schedule(static, CHUNK) //nowait
                #endif
                for(int i=0;i<size;i++) {
                    int s=ic[i]-1;
                    int t=ic[i+1]-1;
                    for(int j=s;j<t;j++)
                        stvc[j]=stvc[j]+(1.0-damping_mult)*(st1vc[j]+st1vc[perm[j]]-beta);
                }
                //for(int i=0;i<snz;i++)
                    //stvc[i]=stvc[i]+(1.0-damping_mult)*(st1vc[i]+st1vc[perm[i]]-beta);
                
                #ifdef DYNSCHED
                    #pragma omp parallel for schedule(dynamic, CHUNK) //nowait
                #else
                    #pragma omp parallel for schedule(static, CHUNK) //nowait
                #endif
                for(int i=0;i<size;i++) {
                    yt[i]=yt[i]+(1.0-damping_mult)*p[i];
                    zt[i]=zt[i]+(1.0-damping_mult)*p[i];
                }
            }
            else
            {
                assert(damping_type == 3);
                #ifdef DYNSCHED
                    #pragma omp parallel for schedule(dynamic, CHUNK) //nowait
                #else
                    #pragma omp parallel for schedule(static, CHUNK) //nowait
                #endif
                for(int i=0;i<size;i++) {
                    int s=ic[i]-1;
                    int t=ic[i+1]-1;
                    for(int j=s;j<t;j++)
                        stvc[j]=damping_mult*stvc[j]+(1.0-damping_mult)*(st1vc[j]+st1vc[perm[j]]-beta);
                }
                //for(int i=0;i<snz;i++)
                    //stvc[i]=damping_mult*stvc[i]+(1.0-damping_mult)*(st1vc[i]+st1vc[perm[i]]-beta);
                
                #ifdef DYNSCHED
                    #pragma omp parallel for schedule(dynamic, CHUNK) //nowait
                #else
                    #pragma omp parallel for schedule(static, CHUNK) //nowait
                #endif
                for(int i=0;i<size;i++) {
                    yt[i]=damping_mult*yt[i]+(1.0-damping_mult)*p[i];
                    zt[i]=damping_mult*zt[i]+(1.0-damping_mult)*p[i];
                }
            }
        }
        
        time1 = timer(); timepoints[5] += time1-time0; time0=time1;
        
        if (opts.limitthreads) {
            omp_set_num_threads(nthreads);
        }
        
        histM.add_heuristic(yt);
        histM.add_heuristic(zt);
        
        time1 = timer(); timepoints[6] += time1-time0; time0=time1;
        
        if (!opts.quiet) {
            cout << "iteration " << t << " " 
                 << timer() - timestart << " secs. "
                 << endl;
        }
    } /// End of iteration for loop
    
    time0 = timer();
    *objective=histM.best_objective();
    cout << "obj: " << *objective << endl;
    double* bestm = histM.best_solution();
    histM.detach_best();
    timepoints[6] += timer()-time0; 
    
    if (opts.verbose) {
        cout << "Timing Report: " << endl;
        cout << "     Setup : " << timepoints[0]/lastiter << "s/iter" << endl;
        cout << "    Line 3 : " << timepoints[1]/lastiter << "s/iter" << endl;
        cout << "    Line 4 : " << timepoints[2]/lastiter << "s/iter" << endl;
        cout << "  Line 5/6 : " << timepoints[3]/lastiter << "s/iter" << endl;
        cout << "    Line 7 : " << timepoints[4]/lastiter << "s/iter" << endl;
        cout << "   Damping : " << timepoints[5]/lastiter << "s/iter" << endl;
        cout << "  Rounding : " << timepoints[6]/lastiter << "s/iter" << endl;
    }
/****************************** clean Up ************************************/

    delete[] dt;
    delete[] yt;
    delete[] zt;
    delete[] yt1;
    delete[] zt1;
    delete[] dt1;
    delete[] Fvc;
    delete[] vvc;
    delete[] p;
    delete[] aw;
    delete[] vc;
    delete St;
	delete St1;
	delete[] perm;
		
/****************************************************************************/
    return bestm;    
}

/**
 * @return the best set of matches generated at any step of the code
 * according to the choice of matching
 */
double* netAlignMPTasked(CRS_Mat* S, Vec w, graph* L, Vec li, Vec lj, 
                    double* wperm, netalign_parameters opts, double* objective)
{
    double alpha = opts.alpha;
    double beta = opts.beta;
    double gamma = opts.gamma;
    int iter = opts.maxiter;
    int damping_type = opts.dampingtype;
    
    int nthreads;
    #pragma omp parallel
    {
        nthreads=omp_get_num_threads();
    }
    
    assert(S->nrow()==w.length());

    int size=S->nrow();
    int snz=S->nnz();
    int ns=L->sVertices;
    int nt=L->numVertices-ns;
    
    double damping_mult = 1.;

    double* dt=new double[size];
    double* yt=new double[size];
    double* zt=new double[size];
    double* yt1=new double[size];
    double* zt1=new double[size];
    double* dt1=new double[size];
    double* Fvc=new double[snz];
    double* Stdamp=new double[snz];
    double* p=new double[size];
    double* aw=new double[size];
    double* wi=w.values();
    double* omy=new double[size];
    double* omz=new double[size];
    
    double *stvc=new double[snz];
    double *st1vc = new double[snz];
   
    int saveIter=opts.batchrounding;
    batch_rounding histM(saveIter, li.values(), lj.values(), L, 
        S, w.values(), wperm, alpha, beta, opts.approx ? -1 : 1);

    other_max_computer omax1(li.values(), ns, size, nthreads);
    other_max_computer omax2(lj.values(), nt, size, nthreads);
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for(int i=0;i<snz;i++) {
            stvc[i]=0.;
            st1vc[i]=0.;
        }
    
        #pragma omp for schedule(static) nowait
        for(int i=0;i<size;i++) {
            aw[i]=alpha*wi[i];
            dt1[i]=0.0;
            yt1[i]=0.0;
            zt1[i]=0.0;
        }
    }
    
    int *ic=S->rowPtr();
    int *perm = build_perm(S);
    // compute split points in the matrix S
    std::vector<int> spSvec(nthreads+1); // split points in the matrix S
    compute_split_points(S, spSvec, nthreads);
    
    int lastiter = 0;
    double timepoints[7] = {0};
    
    assert(damping_type == 2);
    
    for(int t=1;t<=iter;t++)
    {
        lastiter = t;
        
        double time0, time1;

        // Swap data from previous iteration
        time0=timer();

        if(t>1) {
            double *tempptr;
            tempptr = dt1; dt1=dt; dt=tempptr;
            tempptr = yt1; yt1=yt; yt=tempptr;
            tempptr = zt1; zt1=zt; zt=tempptr;
            tempptr = st1vc; st1vc=stvc; stvc = tempptr;
        }
        
        time1 = timer(); timepoints[0] += time1-time0; time0=time1;
        
        // 
        
        #pragma omp parallel num_threads(2)
        {

            // Line 3
            // merged with line 4
            
            //
            // Line 4
            //
            #pragma omp task
            {
                double t0 = timer();
                omp_set_num_threads(nthreads/2);
                
                #ifdef CHUNK
                    #pragma omp parallel for schedule(dynamic,CHUNK)
                #else
                    #pragma omp parallel for schedule(static)
                #endif
                for (int i=0; i<size; ++i) {
                    int h=ic[i]-1;
                    int t=ic[i+1]-1;
                    double sum = 0.; 
                    for(int j=h;j<t;j++) { 
                        // compute the value in F
                        double val = beta+st1vc[perm[j]];
                        if(val<0.) {
                            val=0.;
                        } else {
                            if(val>beta) {
                                val=beta;
                            }
                        } 
                        Fvc[j] = val;
                        sum =+ val;
                    }  
                    dt[i]=sum; // y[t] = sums, z[t] = sums
                }
                timepoints[1] += timer() - t0;
            }
            
            //
            // Line 5, 6
            //
           
            #pragma omp task
            {
                double t0 = timer();
                omp_set_num_threads(nthreads/2);
                if (t>1) {
                    omax2.compute(zt1,omy);
                }
                if(t > 1) {
                    omax1.compute(yt1,omz);
                }
                timepoints[2] += timer() - t0;
            }
            
            // 
            // Precompute dampings
            // 
            #pragma omp task
            {
                double t0 = timer();
                omp_set_num_threads(nthreads/2);
                #pragma omp parallel 
                {
                    #pragma omp for schedule(static) nowait
                    for(int i=0;i<snz;i++) {
                        Stdamp[i]=st1vc[i]+st1vc[perm[i]]-beta;
                    }
                    #pragma omp for schedule(static) nowait
                    for(int i=0;i<size;i++) {
                        p[i]=yt1[i]+zt1[i]-aw[i]+dt1[i];
                    }
                }
                timepoints[5] += timer() - t0;
            }
        }
        
        time1 = timer(); timepoints[3] += time1-time0; time0=time1;
        
        assert(damping_type == 2);
        damping_mult *= gamma;

        //
        // Line 7
        //
        #pragma omp parallel
        {                      
            #ifdef CHUNK
                #pragma omp for schedule(dynamic,CHUNK) nowait
            #else
                #pragma omp for schedule(static)
            #endif
            for(int i=0;i<size;i++) {
                int s=ic[i]-1;
                int t=ic[i+1]-1;
                yt[i] = aw[i] + dt[i] - omy[i] + (1-damping_mult)*p[i];
                zt[i] = aw[i] + dt[i] - omz[i] + (1-damping_mult)*p[i];
                //double val = yt[i]+zt[i]-alpha*w(i)-dt[i];
                double val = aw[i] + dt[i] - omz[i] - omy[i];
                for(int j=s;j<t;j++) {
                    stvc[j]=val-Fvc[j];    
                    // apply damping
                    stvc[j]=stvc[j]+(1.0-damping_mult)*(Stdamp[j]);
                }
            }
        }
        
        time1 = timer(); timepoints[4] += time1-time0; time0=time1; 
                    
        histM.add_heuristic(yt);
        histM.add_heuristic(zt);
        
        time1 = timer(); timepoints[6] += time1-time0; time0=time1;
        
        if (!opts.quiet) {
            cout << "iteration " << t << endl;
        }
    }
    
    double time0 = timer();
    *objective=histM.best_objective();
    double* bestm = histM.best_solution();
    histM.detach_best();
    timepoints[6] += timer()-time0; 
    
    
    if (opts.verbose) {
        cout << "Timing Report: " << endl;
        cout << "     Setup : " << timepoints[0]/lastiter << "s/iter" << endl;
        cout << "    Line 3 : " << timepoints[1]/lastiter << "s/iter" << endl;
        cout << "    Line 4 : " << timepoints[2]/lastiter << "s/iter" << endl;
        cout << "  Line 5/6 : " << timepoints[3]/lastiter << "s/iter" << endl;
        cout << "    Line 7 : " << timepoints[4]/lastiter << "s/iter" << endl;
        cout << "   Damping : " << timepoints[5]/lastiter << "s/iter" << endl;
        cout << "  Rounding : " << timepoints[6]/lastiter << "s/iter" << endl;
    }
    return bestm;    
}

