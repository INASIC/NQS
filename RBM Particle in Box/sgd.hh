#ifndef NQS_SGD_HH
#define NQS_SGD_HH

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace nqs{

using namespace std;
using namespace Eigen;

class Sgd{

  //decay constant
  double eta_;

  int npar_;

  double l2reg_;

  double dropout_p_;

  double momentum_;

  std::mt19937 rgen_;

public:

  Sgd(double eta,double momentum=0,double l2reg=0,double dropout_p=0):eta_(eta),l2reg_(l2reg),dropout_p_(dropout_p),momentum_(momentum){
    npar_=-1;
  }

  void SetNpar(int npar){
    npar_=npar;
  }

  void Update(const VectorXd & grad,VectorXd & pars){
    assert(npar_>0);

    std::uniform_real_distribution<double> distribution(0,1);
    for(int i=0;i<npar_;i++){
      if(distribution(rgen_)>dropout_p_){
        pars(i)=(1.-momentum_)*pars(i) - (grad(i)+l2reg_*pars(i))*eta_;
      }
    }
  }

  void Reset(){

  }
};


}

#endif
