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

  // Parameters
  const double length_;  // Length of the bottom of the Well
  double total_length;  // Total integration length, including edges where V(x)-->\infinity
  const int nspins_;  // Number of spins (i.e. length of byte representing X)
  int n_mesh;  // Number of discretization points in the mesh
  double x0,x1,x2,x3; // Points where the V(x) changes
  double Vmax_;  // Approximately infinite value for V(x) outside the edges
  double dx, dx2; // Interval and square of the interval

public:

  Box1d(int nspins,double length, double Vmax):nspins_(nspins),length_(length),Vmax_(Vmax){
    // Declare dimensions of the box
    x1=0.0;  // Left most wall of the box
    x0=x1 - 10.0;  // Starting point in the integration (outside of box)
    x2=x1 + length_;  // Right most wall
    x3=x2 + 10.0;  // Final integration point
    total_length = x3 - x0;  // Integration region

    // Obtain interval sizes for provided size of input byte --> number of mesh-points
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

  // Finds the connected elements of the Hamiltonian
  // i.e. all the X'(k) such that H(X,X'(k)) \neq 0
  void FindConn(const VectorXd & X,vector<double> & mel,vector<vector<int> > & connectors){
    // Note that here X is the vector of the spins: X=(0,1,1,0,0,1,0,1...)

    // First get the position
    int i_pos = X_to_i(X);
    int n_mtxels = 3;  // Number of matrix elements
    if ((i_pos <= 0) || (i_pos >= n_mesh-1)) n_mtxels = 2;  // Only 2 if at edges of integ. region

    connectors.clear();
    connectors.resize(n_mtxels);
    mel.resize(n_mtxels);

    double x_pos = x0 + i_pos*dx;  // Current location in integration region

    // Diagonal matrix element: mel[0]
    // If we are outside the box, give matrix element max potential energy
    if ((x_pos < x1) || (x_pos > x2)) {mel[0] = Vmax_;} else {mel[0]=0;} // V(x)
    mel[0] -= 2/dx2;  // Add kinetic energy term

    connectors[0].resize(0);  // This is the diagonal element, so no spin flip
    int i_pos1, n1=0;  // ?
    VectorXd X1(nspins_);  // Empty Xprime, <X'|

    // First non-diagonal matrix element:
    if (i_pos > 0) {
      n1++;
      mel[n1] = +1.0/dx2;  // Add kinetic energy matrix element
      i_pos1 = i_pos - 1;  // i_pos_prime
      i_to_X(i_pos1,X1); // Convert mesh-point location into spins vector X'

      // Save which bits need to be flipped to obtain X' from X
      for(int i=0;i<nspins_;i++){
        if (abs(X(i)- X1(i)) > 0.1) {connectors[n1].push_back(i);}
      }
    }

    // Second non-diagonal matrix element:
    if (i_pos < n_mesh-1) {  // ?
      n1++;
      mel[n1] = +1.0/dx2;  // Add kinetic energy matrix element
      i_pos1 = i_pos + 1;  // i_pos_prime
      i_to_X(i_pos1,X1); // Convert mesh-point location into spins vector X'

      // Save which bits need to be flipped to obtain X' from X
      for(int i=0;i<nspins_;i++)        {
        if (abs(X(i)- X1(i)) > 0.1) {connectors[n1].push_back(i);}
      }
    }
  }
};
}
#endif
