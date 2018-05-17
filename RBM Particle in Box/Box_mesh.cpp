//
//
//   Solve the Dyson Equation by first performing Lanczos iteration on the
//  fw- and bk- parts of the self-energy. Multiple pivots are used.
//
//

using namespace std;

#include <cstdlib>
//#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>


// LAPACK driver for diagonalizaton
extern "C" void dsyevd_(char*, char*, int*, double*, int*, 
                        double*, double*, int*, int*, int*,  int* );



int main() {

  int N_mesh = 1024;
  double x0 = -10.0;
  double Length = 120.0;
  double dx = Length/double(N_mesh-1);
  double dx2=dx*dx;

  double x1, xi, Vi;
  
  int ne, LWopt, LIWopt;

// The stuff needed by the LAPACK eigenvalue pakage:
// For 'DSYEV':
  int INFO;
  int LDA = N_mesh+1;
//int LWORK  = 3*LDA+2;          // For 'DSYEV'  only...
  int LWORK  = 2+(6+2*N_mesh)*(N_mesh);   // For 'DSYEVD' only...
  int LIWORK = 3+5*N_mesh;          // For 'DSYEVD' only...


  cout << " LDA    = " << LDA    << endl;
  cout << " LWORK  = " << LWORK  << endl;
  cout << " LIWORK = " << LIWORK << endl;

  double *MTX, *W, *WORK;
  int    *IWORK;
  W = new double[LDA];
  WORK = new double[LWORK];
  IWORK = new int[LIWORK];   


  MTX = new double[LDA*LDA];
  for(double *ptr1= MTX; ptr1<MTX+(LDA*LDA); ++ptr1) (*ptr1)=0.0;

  for (int i=0; i<N_mesh; ++i) {
    xi = x0 + i * Length/double(N_mesh-1);
    Vi = 0.0;   if (xi < 0.0 || xi > 100.0) Vi=1.e8;
    for (int j=0; j<N_mesh; ++j) {
      x1 = 0.0;
      if (abs(i-j) == 1) x1 = -1.0/dx2;
      if (i == j) x1 = Vi + 2.0/dx2;
      MTX[i*LDA + j] = x1;
    }
  }



// LAPACK library (w/ DSYEVD):
       INFO = 0;
/*408*/	
       dsyevd_("Vectors","Upper",&N_mesh,MTX,&LDA,W,WORK, &LWORK,IWORK,&LIWORK,&INFO);
       if (0 != INFO) cout<< "\nDyson (DSYEV), Wrong value of IFAIL: INFO= "<<INFO<<endl;
       if (0 == INFO) {
         ne = int(WORK[0] + 0.01);
         LWopt = (ne > LWopt) ? ne : LWopt;
         LIWopt = (IWORK[0] > LIWopt) ? IWORK[0] : LIWopt;
/*410*/} else {
        cout << " 'DEEGV' gave INFO = "   << INFO << endl;
        if (INFO < 0) cout << "\n The " <<-INFO <<"-th aggument in line 408 was illegal:\n";
        cout << "\n\n Program has been aborted since IERR != 0."
             <<   "\nAborted at the line 410, Dyson.f!"
             <<   "\n   --> stop.\n\n";
        exit(1);
       }

  
  for (int ne=0; ne<30; ++ne) {
    cout << " ne="<<ne<<" :      " << W[ne] <<   "    -->  " << W[ne]/W[0]<<endl;
  }

  ofstream fout("Psi.dat", ios::trunc|ios::out);
  
  for (int i=0; i<N_mesh; ++i) {
    xi = x0 + i * Length/double(N_mesh-1);

    fout << xi;
    for (ne=0; ne<5; ++ne) fout << "     " << MTX[ne*LDA + i]/sqrt(dx);
    for (ne=0; ne<5; ++ne) {
      if (xi < 0.0 || xi > 100.0) {fout << "     " << 0.0;}
      else {fout << "     " << sqrt(2.0/100.)*sin((ne+1)*3.14*xi/100.0);}
    }
    fout << endl;

  }
  

  return 0;}
