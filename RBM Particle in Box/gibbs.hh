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
  const int nv_;  //number of visible units
  const int nh_;  //number of hidden units
  std::mt19937 rgen_;  // random number generator
  RbmState & rbm_;
  VectorXd v_;  // visible nodes
  VectorXd h_;  // hidden nodes
  VectorXd probv_;  //probabilities for gibbs sampling
  VectorXd probh_;

public:

  Gibbs(RbmState & rbm):rbm_(rbm),nv_(rbm.Nvisible()),nh_(rbm.Nhidden()){
    std::random_device rd;
    rgen_.seed(rd());
    v_.resize(nv_);
    h_.resize(nh_);
    probv_.resize(nv_);
    probh_.resize(nh_);

    RandomVals(v_);  // Initialize random spins (visible units)
    cout<<"# Gibbs sampler is ready "<<endl;
  }

  // Generate random visible nodes (i.e. 'spins')
  void Reset(bool initrandom=false){
    if(initrandom){RandomVals(v_);}
  }

  // Propose new node configurations according to Gibbs transition
  // probability (bottom of page 12 Beijing)
  // int sweep_iter = 0;
  void Sweep(){
    // sweep_iter += 1;
    // Change hidden nodes
    rbm_.ProbHiddenGivenVisible(v_,probh_);  //  eq 2.12
    RandomValsWithProb(h_,probh_);
    // cout << sweep_iter << "\t";

    // // Print hidden nodes
    // cout << "| h_ = " << "\t";
    // for(int i=0; i<h_.size(); i++){
    //   cout << h_[i];
    // }

    // Change visible nodes
    rbm_.ProbVisibleGivenHidden(h_,probv_);  // eq 2.11
    RandomValsWithProb(v_,probv_);
    //
    // // Print visible nodes
    // cout << "| v_ = " << "\t" ;
    // for(int i=0; i<v_.size(); i++){
    //   cout << v_[i];
    // }

    // cout << endl;

  }

  // Getters and setters
  VectorXd Visible(){return v_;}
  VectorXd Hidden(){return h_;}
  void SetVisible(const VectorXd & v){v_=v;}
  void SetHidden(const VectorXd & h){h_=h;}

  // Random initalization to nodes
  void RandomVals(VectorXd & hv){
    std::uniform_int_distribution<int> distribution(0,1);

    for(int i=0;i<hv.size();i++){
      hv(i)=distribution(rgen_);
    }
  }

  // Initialize visible units
  int RandomVisible(){std::uniform_int_distribution<int> distribution(0,nv_);
    return distribution(rgen_);
  }

  // Change visible/hidden node configurations according to the
  // Gibbs transition probability (page 12, Beijing)
  void RandomValsWithProb(VectorXd & hv,const VectorXd & probs){
    std::uniform_real_distribution<double> distribution(0,1);

    // Step 1-2 / 3-4
    for(int i=0;i<hv.size();i++){  // Loop over each of the nodes
      hv(i)=distribution(rgen_)<probs(i);
    }
  }

  RbmState & Rbm(){return rbm_;}
  // This always returns a value of 1
  VectorXd Acceptance()const{return VectorXd::Ones(1);}
};
}



#endif
