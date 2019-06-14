#ifndef _SPARSE_H_
#define _SPARSE_H_

#include<cassert>
#include<iostream>
#include<vector>
#include<algorithm>
#include<fstream>
#include <armadillo>

class Vec;
class Matrix;
class CRS_Mat;
class Cord_Mat;

class Vec
{
    private:
        double* vec;                         // Vector
        int n;							// length
    public:
		Vec(int nc);                    // Constructor
		Vec(double* a, int nc);              // Constructor
		int length();                   // Returns the length of the vector
        double* values();                    // Returns the vector pointer
		void set(int i, double val);         // Set a vector element, set(i,10) <=> v[i]=10
		Vec copy(Vec& v);			// Vector copy, x.copy(y) <=> x=y
		double operator()(const int& i);     // Returns ith element, val=v(i)
		double operator*(Vec& v);         // Inner product x'y, scalar=x*y
		Matrix outer(Vec v);         // Outer Product xy', mat=x.outer(y)
		Vec operator*(Matrix& m);    // vec mat multiplication, v*M
		Vec operator+(Vec& v);    // Vector addition, x+y
		Vec operator-(Vec& v);    // Vector subtraction, x-y
		
		friend Vec operator*(double scal, Vec
        & v);                       // Vector scalar Multiplication, scal*v
};

class Matrix
{
private:
	double** mat;							// Matrix
	int n;								// Number of rows
	int m;								// NUmber of columns
public:
    Matrix(int nr, int nc);				// Constructor
	Matrix(double** a, int nr, int nc);			// Constructor
    int nrow();							// Returns the number of rows
    int ncol();							// Returns the number of columns
	void set(int i, int j, double val);		// Set a Matrix entry M(i,j,val) <=> M(i,j)=val
    Matrix copy(Matrix& mt);			// Matrix copy, M1.copy(M2) <=> M1=M2
	Vec operator()(const int& i);	// Get a row M(i)=returns the ith row vector
	Matrix operator+(Matrix& mt);		// Matrix Addition
	Matrix operator-(Matrix& mt);		// Matrix substraction
	Vec operator*(Vec& v);		// Matvec, M*v
	Matrix operator*(Matrix& mt);		// Matrix Matrix Multiplication, M1*M2

	double operator()(const int& i, const int& j);	                                    // Get a entry, val=M(i,j)
	friend Matrix t(Matrix& mat);	                            // Matrix Transpose
	friend Matrix operator*(double scal, Matrix & mat); // Matrix scalar Multiplication, scal*M
};

class CRS_Mat
{
    private:
        double* vals;
        int* cols;
        int* rowInds;
        int nz;
        int n;
        int m;
        bool zeroBased;
    public:
        CRS_Mat(double* va, int* ca, int* ra, int r, int c, int z, bool zerobase);   // Constructor
        CRS_Mat(int r, int c, int z, bool zerobase);                            // Constructor
        CRS_Mat(double* diag, int n,bool zerobase);                                  // Constructor (Creating adiagonal Matrix)
        CRS_Mat(int n, bool zerobase);                                          // Constructor (Identity matrix)
        CRS_Mat(const char* filename);
        CRS_Mat(arma::sp_mat A);
        
        ~CRS_Mat();
        int nrow();                     // Return the number of rows
        int ncol();                     // Return the number of columns
        int nnz();                      // Return the number of non zeros
        int ndiag();							 // Return the number of diagonal elements	
        double* values();                    // Return the values array
        int* colInd();                  // Return the Column Indices
        int* rowPtr();                  // Return the row Pointers
        void convertToOneBased();       // Convert from zero to one based indexing
        void converToZeroBased();       // Convert from one to zero based              
        bool isZeroBase();              // Check whether it is zero based
        void initialize(float k);       // Initializing/Setting all the values to k
         
        void setValuesArray(double* va, int len);                    // Storing the values array
        void setColIndices(int* ca, int len);                   // Storing Column Indices
        void setRowPointers(int* ra, int len, bool zerobase);   // Storing Row Pointers
        void updateValues(double val, int r, int c);                 // Update a specific value
        
        Vec operator*(Vec& v);                // Matvec, M*v
        CRS_Mat operator+(CRS_Mat& mt);       // Matrix Addition
        CRS_Mat operator-(CRS_Mat& mt);       // Matrix Subtraction
        CRS_Mat operator*(CRS_Mat& mt);       // Matrix Matrix-Matrix Multiplication
        friend CRS_Mat operator*(double scal, CRS_Mat & mat); // Matrix scalar Multiplication, scal*M
};

class Cord_Mat
{
	private:
	int* jc;
	int* ic;
	double* vc;
	int n;
	int m;
	int nz;
	public:
	~Cord_Mat();
	Cord_Mat(const char* fname);
	Cord_Mat(arma::sp_mat L);
	int nrow();
	int ncol();
	int nnz();
	int* rowPtr();
	int* colInd();
	double* values();
};

// Helper functions
void copy(int l, double* s, double* t);
void print_vector(double *v, int n, const char *name);

#endif
