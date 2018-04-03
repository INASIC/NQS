#ifndef NQS_BOX1D_HH
#define NQS_BOX1D_HH

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <complex>
#include <vector>

namespace nqs{

using namespace std;
using namespace Eigen;

// The 1D Particle in a box
// H = -(hbar**2/2m) * \nabla + V(x)

class Box1d{

  // Size of the box
  const double length_;  // This is the length of the bottom of the Well
  double total_length;  // Total integration length, including the edges where V(x)-->\infinity

  //Number of spins
  const int nspins_;
  int n_mesh;  // Number of points in the mesh

  // Starting point in x
  double x0,x1,x2,x3; // points where the V(x) changes
  double Vmax_;  // large value for V(x) outside the edges
  double dx, dx2; // interval and square of the interval

public:

  Box1d(int nspins,double length, double Vmax):nspins_(nspins),length_(length),Vmax_(Vmax){
    //will put the beginning of the well at 0 and the edges will be of length 10.
    x1=0.0;
    x0=x1 - 10.0;
    x2=x1 + length_;
    x3=x2 + 10.0;

    total_length = x3 - x0;

    n_mesh = 1; for (int i=0; i<nspins_; i++) n_mesh *= 2;
    dx = total_length / double(n_mesh-1);
    dx2 = dx*dx;
  }

  // Converts (reversed, i.e. left to right) byte representing x-position into a meshpoint
  int X_to_i(const VectorXd & Xin) {
    int i=0; int pow2n = 1;
    for(int n=0; n<nspins_; n++) {i += int(Xin(n)+0.1) * pow2n; pow2n*=2;}
    return i;
  }

  // Converts meshpoint back into byte (set of 'spins') representing x-position
  void i_to_X(int i, VectorXd & Xout) {
    int bit;
    for (int n=0; n<nspins_; n++) {
      bit = i%2;
      Xout(n) = double(bit);
      i = (i-bit)/2.0;
    } // must fill all n spins, with 0s if i is small
    return;
  }

  //Finds the connected elements of the Hamiltonians
  //i.e. all the X'(k) such that H(X,X'(k))\neq0
  //for this model we have k=0,1,...N_spins
  //where k=0 is the diagonal element X'(0)=X
  //input is:
  //X(i), a binary vector containing the state X
  //output is:
  //mel(k), matrix elements H(X,X'(k))
  //connector(k), for each k contains a list of spins that should be flipped to obtain X'(k)
  //starting from X
  void FindConn(const VectorXd & X,vector<double> & mel,vector<vector<int> > & connectors){

    //Note that here X  is the vector of the spins: X=(0,1,1,0,0,1,0,1...)

    //first get the position
    int i_pos = X_to_i(X);
//    cout << i_pos<<endl;
    int n_mtxels = 3;  // Number of matrix elements
    if ((i_pos <= 0) || (i_pos >= n_mesh-1)) n_mtxels = 2;
//    cout << i_pos<<"   "<<n_mtxels<<endl;

    connectors.clear();
    connectors.resize(n_mtxels);
    mel.resize(n_mtxels);


    double x_pos = x0 + i_pos*dx;

    // Diagonal matrix element: mel[0]
    // If we are outside the box, give matrix element max potential energy
    if ((x_pos < x1) || (x_pos > x2)) { mel[0] = Vmax_;} else {mel[0]=0;} // V(x)
    mel[0] += 2/dx2;  // Add kinetic energy term

    // this is the diagonal element, so no spin flip
    connectors[0].resize(0);

    int i_pos1, n1=0;
    VectorXd X1(nspins_);  // Xprime, <X'|

    // first non-diagonal mtx element:
    if (i_pos > 0) {
      n1++;
      mel[n1] = - 1.0/dx2;  // kienti energy mtx. el. (!!! changed to suit eq 28 in your notes)

      i_pos1 = i_pos - 1;  // i_pos_prime
      i_to_X(i_pos1,X1); // get the spins vector X' for the position i_pos1
      //
      // not fill the vector with the porisiotnof the different spins
      //cout << " i_pos / i_pos1 = " << i_pos << "   " << i_pos1 <<endl;
      for(int i=0;i<nspins_;i++){
        //cout << i <<":   "<<X(i) << "  *and  "  << X1(i);

        // Save which bits need to be flipped to obtain X' from X
        if ( abs(X(i)- X1(i)) > 0.1) {connectors[n1].push_back(i); }  //cout << "       different!!! ";}
        //cout << endl;
      }
        // N.B. X(i) values are either 0.0 ot 1.0 but to be sure we don'e use the '=='
    }

    // second non-diagonal mtx element:
    if (i_pos < n_mesh-1) {
      n1++;
      mel[n1] = - 1.0/dx2;  // kienti energy mtx. el.  (!!! changed to suit eq 28 in your notes)

      i_pos1 = i_pos + 1;
      i_to_X(i_pos1,X1); // get the spins vector X' for the position i_pos1

      // not fill the vector with the porisiotnof the different spins
      for(int i=0;i<nspins_;i++)        {
     //   cout << i <<":   "<<X(i) << "   and  "  << X1(i);

        // Save which bits need to be flipped to obtain X' from X
        if ( abs(X(i)- X1(i)) > 0.1) {
          connectors[n1].push_back(i);
          // cout << "       different!!! ";
        }
       // cout << endl;
      }
    }

//    cout << "  i_pos = "<<i_pos<<"   mel.sz="<<mel.size()<<" : "; for (int i=nspins_ - 1; i>=0; --i) cout <<"  "<<X(i); cout << endl;


  }



};  // end of class Box1d
}  // end of namespace nqs

#endif
