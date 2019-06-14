
#include "netAlignKernel.h"

/**
 * Run a small matching problem
 * n the number of nodes
 * m the number of nodes
 * nedges the number of edges
 * v1 is the source for each of the nedges 
 * v2 is the target for each of the nedges
 * weight is the weight of each of the nedges
 * mi is a vector saying which of v1 and v2 are used, length >= nedges
 * 
 * Written by Ying Wang
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

    // allocate memory for problem
	l1 = new double[n];
	l2 = new double[n+m];
	s = new int[n+m];
	t = new int[n+m];
	offset = new int[n];
	deg = new int[n];
	list = new int[nedges + n];
    index = new int[nedges+n];
	w = new double[nedges + n];
    match1 = new int[n];
	match2 = new int[n+m];
    
    // track modifications to t
    int *tmod, ntmod=0;
    tmod = new int[n+m];

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
    
    delete[] index;
    
    delete[] l1;
    delete[] l2;
    delete[] s;
    delete[] t;
    delete[] offset;
    delete[] deg;
    delete[] list;
    delete[] w;
    delete[] match1;
    delete[] match2;
    delete[] tmod;
    
	return ret;
}

/** Holder function for previous code, should be removed and converted as soon as possible
 */
double exact_match(double* x, double* l1, double* l2, double* w, graph* G)
{
    int m=G->sVertices;
    int n=G->numVertices-m;
    int nedges=G->numEdges;
    
    int *li = new int[nedges];
    int *lj = new int[nedges];
    int *ind = new int[nedges];
    
    #pragma omp parallel for
    for (int i=0; i<nedges; ++i) {
        li[i]=(int)l1[i]-1;
        lj[i]=(int)l2[i]-1;
    }
    
    double val = intmatch(m, n, nedges, li, lj, w, ind);
    
    #pragma omp parallel for
    for (int i=0; i<nedges; ++i) {
        x[i]=(double)ind[i];
    }
    
    delete[] li;
    delete[] lj;
    delete[] ind;
    
    return val;
}
