#ifndef NQS_GIBBS_HH
#define NQS_GIBBS_HH

#include <iostream>
#include <Eigen/Dense>
#include <random>

namespace nqs{

using namespace std;
using namespace Eigen;

//Gibbs sampling for binary Restricted Boltzman Machines
template<class RbmState> class Gibbs{

  //number of visible units
  const int nv_;

  //number of hidden units
  const int nh_;

  std::mt19937 rgen_;

  RbmState & rbm_;


  //states of visible and hidden units
  VectorXd v_;
  VectorXd h_;

  //probabilities for gibbs sampling
  VectorXd probv_;
  VectorXd probh_;

public:

  Gibbs(RbmState & rbm):rbm_(rbm),nv_(rbm.Nvisible()),nh_(rbm.Nhidden()){

    std::random_device rd;
    rgen_.seed(rd());

    v_.resize(nv_);
    h_.resize(nh_);

    probv_.resize(nv_);
    probh_.resize(nh_);

    RandomVals(v_);

    cout<<"# Gibbs sampler is ready "<<endl;

  }


  void Reset(bool initrandom=false){
    if(initrandom){
      RandomVals(v_);
    }
  }

  void Sweep(){
    rbm_.ProbHiddenGivenVisible(v_,probh_);
    RandomValsWithProb(h_,probh_);

    rbm_.ProbVisibleGivenHidden(h_,probv_);
    RandomValsWithProb(v_,probv_);
  }


  VectorXd Visible(){
    return v_;
  }


  VectorXd Hidden(){
    return h_;
  }

  void SetVisible(const VectorXd & v){
    v_=v;
  }

  void SetHidden(const VectorXd & h){
    h_=h;
  }

  void RandomVals(VectorXd & hv){
    std::uniform_int_distribution<int> distribution(0,1);

    for(int i=0;i<hv.size();i++){
      hv(i)=distribution(rgen_);
    }
  }

  int RandomVisible(){
    std::uniform_int_distribution<int> distribution(0,nv_);
    return distribution(rgen_);
  }

  void RandomValsWithProb(VectorXd & hv,const VectorXd & probs){
    std::uniform_real_distribution<double> distribution(0,1);

    for(int i=0;i<hv.size();i++){
      hv(i)=distribution(rgen_)<probs(i);
    }
  }

  RbmState & Rbm(){
    return rbm_;
  }

  VectorXd Acceptance()const{
    return VectorXd::Ones(1);
  }

};


}

#endif
