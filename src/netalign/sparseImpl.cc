#include "sparse.h"
#include<vector>
#include<algorithm>
#include<fstream>
#include "netAlignKernel.h"

#include <armadillo>

using namespace std;
/*****************Vec Class Functions ******************/

Vec::Vec(int nc)
{
    n=nc;
    vec=new double[n];
}
 Vec::Vec(double* a, int nc)
{
    int i;
    n=nc;
    vec=new double[n];
    
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(i=0;i<n;i++)
        vec[i]=a[i];    
}

 int Vec::length()
{
    return n;
}

 double* Vec::values()
{
    return vec;
}

 void Vec::set(int i, double val)
{
    vec[i]=val;    
}

 double Vec::operator()(const int& i)
{
    assert(i<n);
    return vec[i];
}

 double Vec::operator*(Vec & v)
{
    assert(n==v.length());
    int i;
    double sum=0;
    for(i=0;i<n;i++)
        sum+=vec[i]*v(i);
    return sum; 
}

 Vec  Vec::operator*(Matrix & m)
{
    assert(n==m.nrow());
    int i,j,sum,col=m.ncol();
    Vec  v(col);
    for(i=0;i<col;i++)
    {    
        sum=0;
        for(j=0;j<n;j++)
            sum+=vec[j]*m(j,i);
        v.set(i,sum);    
    }    
    return v;    
}



 Vec  Vec::operator+(Vec & v)
{
    assert(n=v.length());
    int i;
    Vec  t(n);
    for(i=0;i<n;i++)
        t.set(i,vec[i]+v(i));
    return t;               
}

 Vec  Vec::operator-(Vec & v)
{
    assert(n=v.length());
    int i;
    Vec  t(n);
    for(i=0;i<n;i++)
        t.set(i,vec[i]-v(i));
    return t;               
}

 Vec  Vec::copy(Vec & v)
{
    int i;
    Vec  t(n);
    for(i=0;i<n;i++)
        t.set(i,v(i));
    return t;               
}

Vec operator*(double scal, Vec& v)
{
    int ln=v.length();
    Vec r(ln);
    
    for(int i=0;i<ln;i++)
        r.set(i,scal*v(i));
    
    return r;
}


/***************** Matrix Class Functions ******************/
 Matrix::Matrix(int nr, int nc)
{
    int i;
    n=nr;
    m=nc;
    mat=new double*[n];
    for(i=0;i<n;i++)
        mat[i]=new double[m];
}

Matrix::Matrix(double** a, int nr, int nc)
{
    int i,j;
    n=nr;
    m=nc;
    mat=new double*[n];
    for(i=0;i<n;i++)
        mat[i]=new double[m];
    for(i=0;i<n;i++)
        for(j=0;j<m;j++)
            mat[i][j]=a[i][j];
}

 int Matrix::nrow()
{
    return n;
}

 int Matrix::ncol()
{
    return m;
}

 Vec  Matrix::operator()(const int& i)
{
    assert(i<n);
    int j;
    Vec v(m);
    for(j=0;j<m;j++)
        v.set(j,mat[i][j]);
    return v;
}

 double Matrix::operator()(const int& i, const int& j)
{
    assert(i<n && j<m);
    return mat[i][j];    
}

 Vec Matrix::operator*(Vec & v)
{
    int row=v.length();
    assert(m==row);
    int i,j;
    Vec t(n);
    for(i=0;i<n;i++)
        t.set(i,0);
    for(j=0;j<m;j++)
        if(v(j)!=0)
            for(i=0;i<n;i++)
                t.set(i,t(i)+v(j)*mat[i][j]);
    return t;   
}

 Matrix Matrix::operator*(Matrix& mt)
{
    int row=mt.nrow();
    int col=mt.ncol();
    
    assert(m==row);
    
    int i,j,k;
    Matrix r(n,col);
    
    // Brute force Matrix Matrix Multiplication
    
    for(i=0;i<n;i++)
        for(j=0;j<col;j++)
        {   
            double sum=0;
            for(k=0;k<m;k++)
                sum+=mat[i][k]*mt(k,j);
            r.set(i,j,sum);
        }       
    return r;   
}


 Matrix Matrix::operator+(Matrix& mt)
{
    int row=mt.nrow();
    int col=mt.ncol();
    
    assert(n==row && m==col);
    
    Matrix r(n,m);
    int i,j;
    
    for(i=0;i<n;i++)
        for(j=0;j<m;j++)
            r.set(i,j,mat[i][j]+mt(i,j));
    
    return r;
}

 Matrix Matrix::operator-(Matrix& mt)
{
    int row=mt.nrow();
    int col=mt.ncol();
    
    assert(n==row && m==col);
    
    Matrix r(n,m);
    int i,j;
    
    for(i=0;i<n;i++)
        for(j=0;j<m;j++)
            r.set(i,j,mat[i][j]-mt(i,j));
    
    return r;
}

 Matrix Matrix::copy(Matrix& mt)
{       
    Matrix r(n,m);
    int i,j;
    
    for(i=0;i<n;i++)
        for(j=0;j<m;j++)
            r.set(i,j,mt(i,j));
    
    return r;
}

Matrix t(Matrix& m)
{
    int row=m.nrow();
    int col=m.ncol();
    int i,j;
    Matrix r(col,row);
    if(row==col)
    {
        for(i=0;i<row-1;i++)
        {   
            r.set(i,i,m(i,i));
            for(j=i+1;j<col;j++)
            {
                r.set(i,j,m(j,i));
                r.set(j,i,m(i,j));
            }
        }
    }
    else
    {
        Vec temp(col);
        for(i=0;i<row;i++)
        {
            temp=m(i);  // The ith row
            for(j=0;j<col;j++) 
                r.set(j,i,temp(j)); // ith row -> ith column
        }
    }
    return r;
}

Matrix operator*(double scal, Matrix& m)
{
    int row=m.nrow();
    int col=m.ncol();
    int i,j;
    Matrix r(row,col);
    for(i=0;i<row;i++)
        for(j=0;j<col;j++)
            r.set(i,j,m(i,j)*scal);
    return r;
}

 void Matrix::set(int i, int j, double val)
{
    mat[i][j]=val;
}

/********************** CRS_Mat Functions *************************/

CRS_Mat::~CRS_Mat()
{

    delete[] vals;
    delete[] cols;
    delete[] rowInds;
} 

CRS_Mat::CRS_Mat(double* va, int* ca, int* ra, int r, int c,
                                       int z, bool zerobase)
{
    //cout<<"Constructor"<<endl;
    n=r;
    m=c;
    nz=z;
    zeroBased=zerobase;
    vals=new double[nz];
    cols=new int[nz];
    rowInds=new int[n+1];
    
    /*int i;

    #pragma omp parallel for
    for(i=0;i<nz;i++)
        vals[i]=va[i];
    
    #pragma omp parallel for
    for(i=0;i<nz;i++)
        cols[i]=ca[i];
    
    #pragma omp parallel for
    for(i=0;i<n+1;i++)
        rowInds[i]=ra[i];*/
    
    #ifdef DYNSCHED
        #pragma omp parallel for schedule(dynamic, CHUNK)
    #else
        #pragma omp parallel for schedule(static, CHUNK)
    #endif
    for(int i=0;i<n;i++)
    {    
        rowInds[i]=ra[i];
        int s=ra[i];
        int t=ra[i+1];
        if(!zeroBased)
        {
            s--;
            t--;
        }
        for(int k=s;k<t;k++)
        {    
            cols[k]=ca[k];
            vals[k]=va[k];
            //#pragma omp atomic
            //count++;
        }

        if(i==(n-1))
            rowInds[i+1]=ra[i+1];
    }
    //rowInds[n]=ra[n];
    //cout<<"CountS: "<<count<<" "<<nz<<" "<<rowInds[n]<<endl;
    
    //cout<<"Constructor"<<" "<<rowInds[n]<<endl;
     
}

 CRS_Mat::CRS_Mat(int r, int c, int z, bool zerobase)
{
    n=r;
    m=c;
    nz=z;
    zeroBased=zerobase;
    if(z>0)
    {
        vals=new double[nz];
        cols=new int[nz];
    }
    
    rowInds=new int[n+1];
    
    #pragma omp parallel for
    for(int i=0;i<n+1;i++)
        rowInds[i]=0;
}

 CRS_Mat::CRS_Mat(double* diag, int z, bool zerobase)
{
    n=z;
    m=z;
    nz=z;
    zeroBased=zerobase;
    vals=new double[nz];
    cols=new int[nz];
    rowInds=new int[n+1];
    
    #pragma omp parallel for
    for(int i=0;i<z;i++)
    {
        vals[i]=diag[i];
        if(zeroBased==1)
        {    
            cols[i]=i;
            rowInds[i]=i;
        }
        else
        {    
            cols[i]=i+1;
            rowInds[i]=i+1;
        }

    }
    rowInds[n]=rowInds[n-1]+1;
}

 CRS_Mat::CRS_Mat(int r, bool zerobase)
{
    n=r;
    m=r;
    nz=r;
    zeroBased=zerobase;
    vals=new double[nz];
    cols=new int[nz];
    rowInds=new int[n+1];
    
    #pragma omp parallel for
    for(int i=0;i<nz;i++)
    {
        vals[i]=1;
        if(zeroBased==1)
        {    
            cols[i]=i;
            rowInds[i]=i;
        }
        else
        {    
            cols[i]=i+1;
            rowInds[i]=i+1;
        }

    }

    rowInds[n]=rowInds[n-1]+1;
}

CRS_Mat::CRS_Mat(arma::sp_mat A) {
    int inp, m1, sym;
    int numRow, numCol, nonZeros, numEdges;
    string s;

	numRow=A.n_rows;
	numCol=A.n_cols;
	nonZeros= A.n_nonzero;
	
	arma::uvec ii(A.n_nonzero);
	arma::uvec jj(A.n_nonzero);
	arma::vec vv(A.n_nonzero);
		
	arma::sp_mat::iterator it     = A.begin();
	arma::sp_mat::iterator it_end = A.end();	
	for(int i = 0; it != it_end; ++it, i ++) {
		vv[i] = (*it);
		ii[i] = it.row();
		jj[i] = it.col();
	}	
	
		
	vector<vector<int> > graphCRSIdx(numRow);
	vector<vector<double> > graphCRSVal(numRow);

    int i,j;
    double v;
	for(int k = 0; k < nonZeros; k++) {
		i = ii(k);
		j = jj(k);
		v = vv(k);
		graphCRSIdx[i].push_back(j);
		graphCRSVal[i].push_back(v);			
	}
        
	numEdges = nonZeros; 
	
	nz=numEdges;
	n=numRow;
	m=numCol;
	zeroBased=true;
	
	rowInds  = new int[numRow+1]; 
	 
	vals = new double[numEdges]; 
	cols = new int[numEdges];    
	
	rowInds[0]=0;
	for(i=0;i<numRow;i++)
	{
		
		copy(graphCRSIdx[i].begin(), graphCRSIdx[i].end(), cols + rowInds[i]);
		copy(graphCRSVal[i].begin(), graphCRSVal[i].end(), vals + rowInds[i]);
		rowInds[i+1] = rowInds[i] + graphCRSIdx[i].size();
	}
	
			
	graphCRSIdx.clear();
	graphCRSVal.clear();
}


CRS_Mat::CRS_Mat(const char* fname)
{
    int count=0,i,j;
    int inp, m1, sym;
    int numRow, numCol, nonZeros, numEdges;
    double f;
    string s;
    ifstream inf;
    inf.open(fname, ios::in);
    if(inf.is_open())
    {
        size_t found1, found2, found3;
        getline(inf,s);
        found1 = s.find("pattern"); 
        if (found1 != string::npos)
            m1 = 2;
        else
            m1 = 3;
        found1 = s.find("symmetric");
        found2 = s.find("hermitian");
        found3 = s.find("skew-symmetric");
        if (found1 != string::npos || found2 != string::npos || found3 != string::npos)
            sym = 1;
        else
            sym = 0;
        while(inf.peek()=='%')
            getline(inf,s);
        
        inf>>inp;
        numRow=inp;
        inf>>inp;
        numCol=inp;
        inf>>inp;
        nonZeros=inp;
        
        count=inp;
        
        
        vector<vector<int> > graphCRSIdx(numRow);
        vector<vector<double> > graphCRSVal(numRow);
        int diag=0;
        while(count>0)
        {
            inf>>i;
            inf>>j;
            graphCRSIdx[i-1].push_back(j-1);
            if(m1==3)
            {
                inf>>f;
                f = f;
                graphCRSVal[i-1].push_back(f);
            }
                    
            if (sym && i != j)
                {
                            
                    graphCRSIdx[j-1].push_back(i-1);
                    if(m1==3)
                    {
                        graphCRSVal[j-1].push_back(f);
                    }
                }
            if(i==j)
                diag++;
            count--;
        }
        inf.close();

        
        numEdges = nonZeros; 
        if(sym == 1) //symmetric matrix
            numEdges = nonZeros*2 - diag;
        
        // reading of the input file ends
        
        //build  the CRS
        
      nz=numEdges;
      n=numRow;
      m=numCol;
      zeroBased=true;
        
        rowInds  = new int[numRow+1];  
        vals = new double[numEdges]; 
        cols = new int[numEdges];    
        
        rowInds[0]=0;
        for(i=0;i<numRow;i++)
        {
            
            copy(graphCRSIdx[i].begin(), graphCRSIdx[i].end(), cols + rowInds[i]);
            copy(graphCRSVal[i].begin(), graphCRSVal[i].end(), vals + rowInds[i]);
            rowInds[i+1] = rowInds[i] + graphCRSIdx[i].size();
        }
        
        /*cout<<numRow<<" "<<numCol<<" "<<numEdges<<endl;
        for(i=0;i<100;i++)
        {
            int s=rowInds[i];
            int t=rowInds[i+1];
            for(j=s;j<t;j++)
                cout<<i<<" "<<cols[j]<<endl;;
        }*/
                
        graphCRSIdx.clear();
        graphCRSVal.clear();
    }
    else
        cout<<"file not open"<<endl;
    //cout<<"Input Processing is done nz="<< nonZeros <<endl;  
}


void CRS_Mat::initialize(float k)
{
    int i;
    if(nz>0)
        for(i=0;i<nz;i++)
            vals[i]=k;
}

 int CRS_Mat::nrow()
{
    return n;
}

 int CRS_Mat::ncol()
{
    return m;
}

 int CRS_Mat::nnz()
{
    return nz;
}

int CRS_Mat::ndiag()
{
    int ndiag=0;
    
    //#pragma omp parallel for
    for(int i=0;i<n;i++)
    {
            int s=rowInds[i];
            int t=rowInds[i+1];
            for(int j=s;j<t;j++)
                if(i==cols[j])
                {
                    //#pragma omp atomic
                    ndiag++;
                }
    }           
    
    return ndiag;
}

 double* CRS_Mat::values()
{
    return vals;
}

 int* CRS_Mat::colInd()
{
    return cols;
}

 int* CRS_Mat::rowPtr()
{
    return rowInds;
}

 void CRS_Mat::setValuesArray(double* val, int len)
{
    assert(len==nz);
    for(int i=0;i<len;i++)
        vals[i]=val[i];
}
 void CRS_Mat::setColIndices(int* ca, int len)
{
    assert(len==nz);
    
    for(int i=0;i<len;i++)
        cols[i]=ca[i];
}
 void CRS_Mat::setRowPointers(int* ra, int len, bool
                                                    zerobased)
{
    assert(len==n+1);
    
    zeroBased=zerobased;
    if(zeroBased==true)
        assert(ra[0]==0);
    else
        assert(ra[0]==1);
    
    for(int i=0;i<len;i++)
        rowInds[i]=ra[i];
}
 void CRS_Mat::updateValues(double val, int r, int c)
{
    assert(r<=n && c<=m);
    int flag=0;
    for(int i=rowInds[r];i<rowInds[i+1];i++)
        if(cols[i]==c)
        {    
            vals[i]=val;
            flag=1;
        }
    if(flag==0)
        std::cout<<"There is no such entry("<<r<<","<<c<<")"<<std::endl;
}

 void CRS_Mat::convertToOneBased()
{
    int i;
    if(zeroBased==true)
    {
        //cout<<"converting"<<endl;
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for(i=0;i<nz;i++)
            cols[i]++;
        
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for(i=0;i<n+1;i++)
            rowInds[i]++;    
        zeroBased = false;
    }
}

 void CRS_Mat::converToZeroBased()
{
    int i;
    if(zeroBased==false)
    {
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for(i=0;i<nz;i++)
            cols[i]--;
        
        #ifdef DYNSCHED
            #pragma omp parallel for schedule(dynamic, CHUNK)
        #else
            #pragma omp parallel for schedule(static, CHUNK)
        #endif
        for(i=0;i<n+1;i++)
            rowInds[i]--;    
        zeroBased = true;
    }
}             

 bool CRS_Mat::isZeroBase()
{
    if(zeroBased==true)
        return true;
    else
        return false;
} 

 

/************************* Cord Matrix Functions ***************/
Cord_Mat::~Cord_Mat()
{
    delete[] ic;
    delete[] jc;
    delete[] vc;
}


Cord_Mat::Cord_Mat(arma::sp_mat L) {
    int inp, m1, sym, count;
    int numRow, numCol, nonZeros, numEdges;
    string s;

	numRow=L.n_rows;
	numCol=L.n_cols;
	nonZeros= L.n_nonzero;
            
	arma::uvec ii(L.n_nonzero);
	arma::uvec jj(L.n_nonzero);
	arma::vec vv(L.n_nonzero);
		
	arma::sp_mat::iterator it     = L.begin();
	arma::sp_mat::iterator it_end = L.end();	
	for(int i = 0; it != it_end; ++it, i ++) {
		vv[i] = (*it);
		ii[i] = it.row();
		jj[i] = it.col();
	}	
	
	vector<vector<int> > graphCRSIdx(numRow);
	vector<vector<double> > graphCRSVal(numRow);

    double v;
    int i,j;
	for(int k = 0; k < nonZeros; k++) {
		i = ii(k);
		j = jj(k);
		v = vv(k);
		graphCRSIdx[i].push_back(j);
		graphCRSVal[i].push_back(v);
	}
	numEdges = nonZeros; 
	
	//build  the Cordinates
	n=numRow;
	m=numCol;
	nz=numEdges;
	ic = new int[numEdges];  
	vc = new double[numEdges]; 
	jc = new int[numEdges];    
	
	count=0;
	for(i=0;i<numRow;i++)
	{
		for(j=0;(size_t)j<graphCRSIdx[i].size();j++)
		{
			ic[count]=i+1;
			jc[count]=graphCRSIdx[i][j]+1;
			vc[count]=graphCRSVal[i][j];
			count++;            
		}   
	}
			
			
	graphCRSIdx.clear();
	graphCRSVal.clear();
}



Cord_Mat::Cord_Mat(const char* fname)
{
    int count=0,i,j;
    int inp, m1, sym;
    int numRow, numCol, nonZeros, numEdges;
    double f;
    string s;
    ifstream inf;
    inf.open(fname, ios::in);
    if(inf.is_open())
    {
        size_t found1, found2, found3;
        getline(inf,s);
        found1 = s.find("pattern"); 
        if (found1 != string::npos)
            m1 = 2;
        else
            m1 = 3;
        found1 = s.find("symmetric");
        found2 = s.find("hermitian");
        found3 = s.find("skew-symmetric");
        if (found1 != string::npos || found2 != string::npos || found3 != string::npos)
            sym = 1;
        else
            sym = 0;
        while(inf.peek()=='%')
            getline(inf,s);
        
        inf>>inp;
        numRow=inp;
        inf>>inp;
        numCol=inp;
        inf>>inp;
        nonZeros=inp;
        
        count=inp;
        
        
        vector<vector<int> > graphCRSIdx(numRow);
        vector<vector<double> > graphCRSVal(numRow);
        int diag=0;
        while(count>0)
        {
            inf>>i;
            inf>>j;
            graphCRSIdx[i-1].push_back(j-1);
            if(m1==3)
            {
                inf>>f;
                f = f;
                graphCRSVal[i-1].push_back(f);
            }
                    
            if (sym && i != j)
                {
                            
                    graphCRSIdx[j-1].push_back(i-1);
                    if(m1==3)
                    {
                        graphCRSVal[j-1].push_back(f);
                    }
                }
            if(i==j)
                diag++;
            count--;
        }
        inf.close();

        
        numEdges = nonZeros; 
        if(sym == 1) //symmetric matrix
            numEdges = nonZeros*2 - diag;
        //cout<<"Diag: "<<diag<<endl;
        // reading of the input file ends
        
                
        //build  the Cordinates
        n=numRow;
        m=numCol;
        nz=numEdges;
        ic = new int[numEdges];  
        vc = new double[numEdges]; 
        jc = new int[numEdges];    
        
        count=0;
        for(i=0;i<numRow;i++)
        {
            for(j=0;(size_t)j<graphCRSIdx[i].size();j++)
            {
                ic[count]=i+1;
                jc[count]=graphCRSIdx[i][j]+1;
                vc[count]=graphCRSVal[i][j];
                count++;            
            }   
        }
        
        /*cout<<numRow<<" "<<numCol<<" "<<numEdges<<endl;
        for(i=0;i<10;i++)
            cout<<ic[i]<<" "<<jc[i]<<" "<<vc[i]<<endl;*/
            
                
        graphCRSIdx.clear();
        graphCRSVal.clear();
    }
    else
        cout<<"file not open"<<endl;
    //cout<<"Input Processing is done nz="<< nonZeros <<endl;
}

 int Cord_Mat::nrow()
{
    return n;
}

 int Cord_Mat::ncol()
{
    return m;
}

 int Cord_Mat::nnz()
{
    return nz;
}

 double* Cord_Mat::values()
{
    return vc;
}

 int* Cord_Mat::colInd()
{
    return jc;
}

 int* Cord_Mat::rowPtr()
{
    return ic;
}


void copy(int l, double* s, double* t)
{
    #pragma omp parallel for
    for(int i=0;i<l;i++)
        t[i]=s[i];
}

void print_vector(double *v, int n, const char *name)
{
    for (int i=0; i<n; ++i) {
        cout << name << "(" << i+1 << ") = " << v[i] << ";" << endl;
    }
}
