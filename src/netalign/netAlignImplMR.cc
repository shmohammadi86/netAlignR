
#include "netAlignKernel.h"
using namespace std;

static int chunk;

/** A functor for maxrowmatch which saves structural state
 * 
 * Because this routine saves structural state, it can be run on new 
 * data more efficiently than a general max-weight-matching on 
 * each row.  
 * 
 * (From Matlab description)
 * Given a sparse matrix Q where each row corresponds to a possible
 * edge in a bipartite matching (i.e. Q(i,:) corresponds with li(i), lj(i))
 * then we compute the row-sum of Q subject to the constraint that
 * the values summed are a matching in li, lj.  In other words,
 * instead of q=Q*e, the standard row-sum, we compute
 *   q(i) = Q(i,:)*m_i where m_i is the maximum matching 
 *     of all the non-zero values Q(i,:) with edges from li,lj
 * 
 */
struct maxrowmatch_functor {
    // static info on the matrix Q
    int M, N;
    std::vector<int> QpVec, QrVec;
    int *Qp, *Qr;
    int max_row_nonzeros;
    std::vector<int> splitsVec;
    int *splits;
    
    // static info on the graph L
    int m, n, nedges;
    std::vector<int> liVec, ljVec;
    int *li, *lj;
    
    /** Separate out data needed to solve an individual matching problem.
     */
    struct maxrowmatch_thread {
        // maps from the big graph to the small graph
        std::vector<int> lwork1Vec, lind1Vec, lwork2Vec, lind2Vec;
        std::vector<int> se1Vec, se2Vec, sqindVec, smiVec;
        std::vector<double> swVec;
        
        int *Qp, *Qr;
        int *li, *lj;
        
        int *lwork1, *lind1, *lwork2, *lind2, *se1, *se2, *sqind, *smi;
        double *sw;
        
        int iworksize;
        std::vector<int> iworkmatchVec;
        int *iworkmatch;
        
        int worksize;
        std::vector<double> workmatchVec;
        double *workmatch;
        
        // lightweight allocation
        maxrowmatch_thread() {}
        
        // ripped from the exact match code
        
        /**
         * Run a small matching problem
         * n the number of nodes
         * m the number of nodes
         * nedges the number of edges
         * v1 is the source for each of the nedges 
         * v2 is the target for each of the nedges
         * weight is the weight of each of the nedges
         * mi is a vector saying which of v1 and v2 are used, length >= nedges
         */
        double intmatch(int n, int m, 
                int nedges, int *v1, int *v2, double *weight, 
                int *mi)
        {
            double ret, al;
            double *l1, *l2, *w;
            int *match1, *match2;
            int i, j, k, p, q, r, t1, t2;
            int *s, *t, *deg, *offset, *list, *index;
            
            // divide up workspace
            // workspace size = 3*n+m+nedges
            // iworkspace size = 9n + 4m + 2*nedges
            // note that n, m, and nedges are upper-bounded 
            // by max_row_nonzeros
            int workoffset=0; int iworkoffset=0;
            l1 = &workmatch[workoffset]; workoffset += n;
            l2 = &workmatch[workoffset]; workoffset += (n+m);
            
            s = &iworkmatch[iworkoffset]; iworkoffset += (n+m);
            t = &iworkmatch[iworkoffset]; iworkoffset += (n+m);
            offset = &iworkmatch[iworkoffset]; iworkoffset += (n);
            deg = &iworkmatch[iworkoffset]; iworkoffset += (n);
            list = &iworkmatch[iworkoffset]; iworkoffset += (nedges + n);
            index = &iworkmatch[iworkoffset]; iworkoffset += (nedges + n);
            
            w = &workmatch[workoffset]; workoffset += (nedges + n);
            match1 = &iworkmatch[iworkoffset]; iworkoffset += (n);
            match2 = &iworkmatch[iworkoffset]; iworkoffset += (n+m);
            
            // track modifications to t
            int *tmod, ntmod=0;
            tmod = &iworkmatch[iworkoffset]; iworkoffset += (n+m);
            
            assert(workoffset <= worksize);
            assert(iworkoffset <= iworksize);
            
            

            for (i = 0; i < n; i++) {
                offset[i] = 0;
                deg[i] = 1;
            }
            for (i = 0; i < nedges; i++) deg[v1[i]]++;
            for (i = 1; i < n; i++) offset[i] = offset[i-1] + deg[i-1];
            for (i = 0; i < n; i++) deg[i] = 0;
            for (i = 0; i < nedges; i++) {
                list[offset[v1[i]] + deg[v1[i]]] = v2[i];
                w[offset[v1[i]] + deg[v1[i]]] = weight[i];
                index[offset[v1[i]] + deg[v1[i]]] = i;
                deg[v1[i]]++;
            }
            for (i = 0; i < n; i++) {
                list[offset[i] + deg[i]] = m + i;
                w[offset[i] + deg[i]] = 0;
                index[offset[i] + deg[i]] = -1;
                deg[i]++;
            }
            for (i = 0; i < n; i++) {
                l1[i] = 0;
                for (j = 0; j < deg[i]; j++) {
                    if (w[offset[i]+j] > l1[i]) l1[i] = w[offset[i] + j];
                }
            }
            // initialize the primal match
            for (i = 0; i < n; i++) {
                match1[i] = -1;
            }
            // initialize the dual variables l2
            for (i = 0; i < n + m; i++) {
                l2[i] = 0;
                match2[i] = -1;
            }
            // initialize t once
            for (j=0; j < n+m; j++) {
                t[j] = -1;
            }
            
            for (i = 0; i < n; i++) {
                for (j=0; j<ntmod; j++) {
                    t[tmod[j]] = -1;
                }
                ntmod = 0;
                
                // clear the queue and add i to the head
                s[p = q = 0] = i;
                for(; p <= q; p++) {
                    if (match1[i] >= 0) break;
                    k = s[p];
                    for (r = 0; r < deg[k]; r++) {
                        j = list[offset[k] + r];
                        if (w[offset[k] + r] < l1[k] + l2[j] - 1e-8) continue;
                        if (t[j] < 0) {
                            s[++q] = match2[j];
                            t[j] = k;
                            tmod[ntmod]=j; // save our modification to t
                            ntmod++;
                            if (match2[j] < 0) {
                                for(; j>=0 ;) {
                                    k = match2[j] = t[j];
                                    // reusing p here is okay because we'll
                                    // stop below
                                    p = match1[k];
                                    match1[k] = j;
                                    j = p;
                                }
                                break; // we found an alternating path and updated
                            }
                        }
                    }
                }
                if (match1[i] < 0) {
                    al = 1e20;
                    for (j = 0; j < p; j++) {
                        t1 = s[j];
                        for (k = 0; k < deg[t1]; k++) {
                            t2 = list[offset[t1] + k];
                            if (t[t2] < 0 && l1[t1] + l2[t2] - w[offset[t1] + k] < al) {
                                al = l1[t1] + l2[t2] - w[offset[t1] + k];
                            }
                        }
                    }
                    for (j = 0; j < p; j++) l1[s[j]] -= al;
                    //for (j = 0; j < n + m; j++) if (t[j] >= 0) l2[j] += al;
                    for (j=0; j<ntmod; j++) { l2[tmod[j]] += al; }
                    i--;
                    continue;
                }
            }
            
            ret = 0;
            for (i = 0; i < n; i++) {
                for (j = 0; j < deg[i]; j++) {
                    if (list[offset[i] + j] == match1[i]) {
                        ret += w[offset[i] + j];
                    }
                }
            }        
            
            // build the matching indicator 
            for (i=0; i<nedges; i++) {
                mi[i] = 0;
            }
            for (i=0; i<n; i++) {
                if (match1[i] < m) {
                    for (j = 0; j < deg[i]; j++) {
                        if (list[offset[i] + j] == match1[i]) {
                            mi[index[offset[i]+j]] = 1;
                        }
                    }
                }
            }
            
            return ret;
        }

        
        void allocate(int *Qp_, int *Qr_, int *li_, int *lj_,
            int m, int n, int max_row_nonzeros)
        {
            Qp = Qp_;
            Qr = Qr_;
            li = li_;
            lj = lj_;
            
            lwork1Vec.resize(m);
            lind1Vec.resize(m);
            lwork1 = &lwork1Vec[0];
            lind1 = &lind1Vec[0];
            
            lwork2Vec.resize(n);
            lind2Vec.resize(n);
            lwork2 = &lwork2Vec[0];
            lind2 = &lind2Vec[0];
            
            se1Vec.resize(max_row_nonzeros);
            se2Vec.resize(max_row_nonzeros);
            swVec.resize(max_row_nonzeros);
            
            se1 = &se1Vec[0];
            se2 = &se2Vec[0];
            sw = &swVec[0];
            
            #pragma omp parallel for
            for (int i=0; i<m; ++i) {
                lind1[i] = -1;
            }
            
            #pragma omp parallel for
            for (int i=0; i<n; ++i) {
                lind2[i] = -1;
            }
            
            // allocate matching workspace
            // double workspace size = 3*n+m+nedges
            // int workspace size = 9n + 4m + 2*nedges
            // where n, m, and nedges are upper-bounded by max_row_nonzeros
            workmatchVec.resize(5*max_row_nonzeros);
            workmatch = &workmatchVec[0];
            worksize = (int)workmatchVec.size();
            iworkmatchVec.resize(15*max_row_nonzeros);
            iworkmatch = &iworkmatchVec[0];
            iworksize = (int)iworkmatchVec.size();
        }
        
        double match_in_row(int j, double *Qv, int *edgeflag)
        {
            double q = 0.;
            int smalledges = 0;
            int nsmall1 = 0; // number of vertices on side 1 of column matching
            int nsmall2 = 0; // number of vertices on side 2 of column matching
            for (int nzi=Qp[j]; nzi<Qp[j+1]; nzi++) {
                int i = (int)Qr[nzi];
                int v1 = li[i];
                int v2 = lj[i];
                int sv1=-1, sv2=-1;
                if (lind1[v1] < 0) {
                    // add it to the map
                    sv1 = nsmall1;
                    lind1[v1] = sv1;
                    lwork1[sv1] = v1;
                    nsmall1++;
                } else {
                    sv1 = lind1[v1];
                }
                if (lind2[v2] < 0) {
                    // add it to the map
                    sv2 = nsmall2;
                    lind2[v2] = sv2;
                    lwork2[sv2] = v2;
                    nsmall2++;
                } else {
                    sv2 = lind2[v2];
                }
                
                edgeflag[nzi] = 0;
                se1[smalledges] = sv1;
                se2[smalledges] = sv2;
                sw[smalledges] = Qv[nzi];
                smalledges++;
            }
            
            if (smalledges == 0) {
                return (0.);
            }
            
            q = intmatch(nsmall1, nsmall2,
                        smalledges, se1, se2, sw, 
                        &edgeflag[Qp[j]]);
            
            // reset the maps from big to small
            for (int k=0; k<nsmall1; k++) {
                lind1[lwork1[k]] = -1;
            }
            for (int k=0; k<nsmall2; k++) {
                lind2[lwork2[k]] = -1;
            }
            
            return (q);
        }
        
    };
    
    std::vector<maxrowmatch_thread> tdata;
    
    /** 
     * @param Q the underlying matrix, ONE-based indices
     * @param L the bipartite graph 
     * @param li first index of edges in L 
     * @param lj second index of edges in L, must align with columns of Q
     * 
     */
    maxrowmatch_functor(CRS_Mat *Q, graph *L, double *li_, double *lj_)
    : M(Q->nrow()), N(Q->ncol()), QpVec(Q->nrow()+1), QrVec(Q->nnz()),
        m(L->sVertices), n(L->numVertices-m), nedges(L->numEdges),
        liVec(nedges), ljVec(nedges)
    {
        assert(N == nedges);
        // convert L data to integers
        #pragma omp parallel for
        for (int i=0; i<nedges; ++i) {
            liVec[i] = (int)(li_[i]) - 1;
            ljVec[i] = (int)(lj_[i]) - 1;
        }
        li = &liVec[0];
        lj = &ljVec[0];
        
        // convert Q to zero indexed
        #pragma omp parallel for
        for (int i=0; i<M+1; ++i) {
            QpVec[i] = Q->rowPtr()[i]-1;
        }
        #pragma omp parallel for
        for (int i=0; i<Q->nnz(); ++i) {
            QrVec[i] = Q->colInd()[i]-1;
        }
        int *Qp = &QpVec[0];
        int *Qr = &QrVec[0];
        
        // determine the maximum number of nonzeros in a row
        // this is used to build maps from each row down to
        // a local set of data
        max_row_nonzeros = 0;
        // Seriously? OMP has no "max" function for reduction?
        // Anyway, let's do it ourselves.
        #pragma omp parallel
        {
            int max_row = 0;
            #pragma omp for nowait
            for (int i=0; i<N; ++i) {
                int row_nonzeros = Qp[i+1] - Qp[i];
                if (row_nonzeros > max_row) {
                    max_row = row_nonzeros; 
                }
            }
            #pragma omp critical
            {
                if (max_row > max_row_nonzeros) {
                    max_row_nonzeros = max_row;
                }
            }
        }
        
        // allocate data for each thread
        tdata.resize(omp_get_max_threads());
        for (size_t i=0; i<tdata.size(); ++i) {
            tdata[i].allocate(Qp, Qr, li, lj, m, n, max_row_nonzeros);
        }
    }
    
    /**
     * @param Qv the values array for the matrix Q
     * @param q the matching score for each row
     * @param edgeflag an indicator vector over edges in S
     *   using the same indexing as the CSR representation 
     *   of S
     */
    void apply(double *Qv, double *q, int *edgeflag) {
        #pragma omp parallel for schedule(dynamic,100)
        for (int j=0; j<M; ++j) {
            int tid = omp_get_thread_num();
            q[j] = tdata[tid].match_in_row(j,Qv,edgeflag);
        }
    }
};

double* netAlignMR(CRS_Mat* S, Vec w, graph* L, Vec li, Vec lj, double* wperm, 
    netalign_parameters opts, double* objective)
{
    double alpha = opts.alpha;
    double beta = opts.beta;
    double gamma = opts.gamma;
    int iter = opts.maxiter;
    int stepm = opts.batchrounding;
    chunk = opts.chunk;
    
    assert(S->nrow()==w.length() && gamma >= 0 && gamma <=1);
    
    double time=0,time1=0;
    double lt3=0,lt4=0,lt5=0,lt6=0,lt8=0;
    int size=S->nrow();
    int snz=S->nnz();
    int ns=L->sVertices;
    int nt=L->numVertices-ns;

    int* mate = new int[L->numVertices];
    int* heaviest = new int[L->numVertices];
    int* Q1 = new int[L->numVertices];
    int* Q2 = new int[L->numVertices];
    int* Visited = new int[nt];
    double* RMax = new double[ns];
    double* buffer = new double[2*size];
    
    double *vals=new double[snz];
    double *wbar=new double[size];
    double *x=new double[size];
    double *xbest=new double[size];
    double *U = new double[snz];
    int *tperm = build_perm(S);
    int *jc=S->colInd(), *ic=S->rowPtr();
    double *wi = w.values();
    
    double flower = 0., fupper = 1.e64;
    char itermark;
    
    
    std::vector<int> edgeflagVec(snz);
    int* edgeflag = &edgeflagVec[0];
    maxrowmatch_functor mrmatch(S, L, li.values(), lj.values());
    
    // initialize U
    #pragma omp parallel for
    for (int i=0; i<snz; ++i) { 
        U[i] = 0.;
    }
    
    int lastiter = 0;
    int next_reduction_iteration = stepm;
    for(int k=0;k<iter;k++) {
        // start a timer
        double timestart = timer();
        lastiter= k+1;
        time=timer();
        
        // Line 3
        #pragma omp parallel for
        for(int i=0;i<snz;i++)
            vals[i]=beta/2. + U[i] - U[tperm[i]];   
            
        mrmatch.apply(vals, wbar, edgeflag);
        
        time1=timer(); lt3+=time1-time; time=time1;
        
        // Line 4
        //daxpy(&size, &alpha, w.values(), &incx, wbar, &incy);
        caxpy(size, alpha, w.values(), wbar);
        
        time1=timer(); lt4+=time1-time; time=time1;

        
        // Line 5
        double val=0.;
        if (opts.approx) {
                
           #pragma omp parallel for schedule(dynamic, 1000)
           for(int i=0;i<L->numEdges;i++)
           {
                if(i<size) {
                    buffer[i]=wbar[i];
                } else {
                    buffer[i]=wbar[(int)wperm[i-size]];}
           }
           val = ApproxEdgeWeightedMatching(x,mate, heaviest,Q1, Q2, L, buffer, RMax, Visited);
           val = val*2.; // adjust for upper bound
        } else {
            int *ind = new int[size];
            val = intmatch(mrmatch.m, mrmatch.n, size, 
                        mrmatch.li, mrmatch.lj, wbar, ind);
            #pragma omp parallel for
            for (int i=0; i<size; ++i) {
                x[i] = (double)ind[i];
            }
            delete[] ind;
        }
        
        time1=timer(); lt5+=time1-time; time=time1;
        
        // Line 6 
        // compute the objective
        double overlap = 0;
        double matchval = 0.;
        #pragma omp parallel for schedule(dynamic,1000) \
            reduction(+:overlap) reduction(+:matchval)
        for (int row=0; row<size; ++row) {
            matchval += x[row]*wi[row];
            for (int nzi=ic[row]-1; nzi<ic[row+1]-1; ++nzi) {
                overlap += x[row]*x[jc[nzi]-1];
            }
        }
        overlap*=0.5;
        double obj=alpha*matchval + beta*overlap;
        
        if (val < fupper) {
            fupper = val;
            next_reduction_iteration = k + stepm;
        } 
        if (obj > flower) {
            flower = obj;
            itermark = '*';
            copy(size, wbar, xbest);
        } else {
            itermark = ' ';
        }
        
        time1=timer(); lt6+=time1-time; time=time1;
        
        if (k==next_reduction_iteration) {
            gamma = gamma*0.5;
            if (opts.verbose) {
                printf("%5c   %4c   reducing gamma to %g\n", 
                ' ', ' ', gamma);
            }
            if (gamma < 1e-22) {
                break;
            }
            next_reduction_iteration = k+stepm;
        }
        
        // Line 8/9
        // Transpose Sl in edgeflag, into the vals array;
        #pragma omp parallel for schedule(dynamic,1000)
        for (int i=0; i<snz; ++i) {
            vals[i] = (double)edgeflag[tperm[i]];
        }
        
        #pragma omp parallel for schedule(dynamic,1000)
        for(int i=0;i<size;i++) {
            double xi = x[i];
            for (int nzi=ic[i]-1; nzi<ic[i+1]-1; ++nzi) {
                int j=jc[nzi]-1;
                double xj = x[j];
                double f=U[nzi];
                if (i>j) {
                    // this is the tril portion, ignore it
                } else {
                    // this is a triu position, update it!
                    f += -xi*gamma*(double)edgeflag[nzi] + xj*gamma*vals[nzi];
                    if (f < -0.5) { f = -0.5; }
                    else if (f > 0.5) { f = 0.5; }
                }
                U[nzi] = f;
            }
        }
        
        time1=timer(); lt8+=time1-time; time=time1;
        
        if (!opts.quiet) {
            printf("%5c   %4i   %7g %7g %7g  %7g %7g %7g  %7.1f sec\n", 
                itermark, k+1, flower, fupper, val, obj, matchval, overlap,
                timer() - timestart);
        }
    }
     
    if (opts.verbose) {
        cout << "Timing Report: " << endl;
        cout << "  Line 3: " << lt3/lastiter << "s/iter" << endl;
        cout << "  Line 4: " << lt4/lastiter << "s/iter" << endl;
        cout << "  Line 5: " << lt5/lastiter << "s/iter" << endl;
        cout << "  Line 6: " << lt6/lastiter << "s/iter" << endl;
        cout << "  Line 8: " << lt8/lastiter << "s/iter" << endl;
    }
    return xbest;    
}
