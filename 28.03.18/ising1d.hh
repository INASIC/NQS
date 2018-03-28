#ifndef NQS_ISING1D_HH
#define NQS_ISING1D_HH

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <complex>
#include <vector>

namespace nqs{

using namespace std;
using namespace Eigen;

//Transverse-Field Ising model in 1D
//With periodic boundary conditons
//H=-h*\sum_i \sigma^x_i -J*\sum_i \sigma^z_i*sigma^z_{i+1}

class Ising1d{

  //Number of spins
  const int nspins_;

  //Longitudinal coupling constant
  double J_;

  //Transverse-Field coupling constant
  double h_;

public:

  Ising1d(int nspins,double h,double J=1):nspins_(nspins),h_(h),J_(J){
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

    connectors.clear();
    connectors.resize(nspins_+1);
    mel.resize(nspins_+1);

    mel[0]=0;
    connectors[0].resize(0);

    for(int i=0;i<nspins_;i++){
      mel[i+1]=-h_;
      connectors[i+1].push_back(i);

      mel[0]-=J_*(2*X(i)-1.)*(2.*X((i+1)%nspins_)-1.);
    }

  }

};


}

#endif
