#ifndef NQS_VARIATIONAL_HH
#define NQS_VARIATIONAL_HH

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <random>
#include <complex>
#include <vector>
#include "rbm.hh"

namespace nqs{

using namespace std;
using namespace Eigen;

template<class Hamiltonian,class RbmState,class Sampler,class Optimizer> class Variational{

  Hamiltonian & ham_;
  Sampler & sampler_;
  RbmState & rbm_;

  VectorXd v_;
  vector<vector<int> >  connectors_;
  vector<double> mel_;
  VectorXd logvaldiffs_;

  VectorXd elocs_;
  MatrixXd Ok_;
  MatrixXd vsamp_;
  MatrixXd S_;

  VectorXd grad_;

  double elocmean_;
  int npar_;

  Optimizer opt_;

  int Iter0_;

public:

  Variational(Hamiltonian & ham,Sampler & sampler,Optimizer & opt):ham_(ham),sampler_(sampler),rbm_(sampler.Rbm()),opt_(opt){

    npar_=rbm_.Npar();
    grad_.resize(npar_);
    opt_.SetNpar(npar_);
    Iter0_=0;

  }

  void Sample(int nsweeps){
    sampler_.Reset();
    vsamp_.resize(nsweeps,rbm_.Nvisible());

    for(int i=0;i<nsweeps;i++){
      sampler_.Sweep();
      vsamp_.row(i)=sampler_.Visible();
    }
  }

  // Check this function.
  // grad_ returns zero, why is that? Because Ok = 0.
  void Gradient(){
    const int nsamp=vsamp_.rows();
    elocs_.resize(nsamp);

    Ok_.resize(nsamp,rbm_.Npar());

    for(int i=0;i<nsamp;i++){
      elocs_(i)=Eloc(vsamp_.row(i));

      Ok_.row(i)=rbm_.DerLog(vsamp_.row(i));  // Check here
    }

    elocmean_=elocs_.mean();

    Ok_=Ok_.rowwise() - Ok_.colwise().mean();

    elocs_-=elocmean_*VectorXd::Ones(nsamp);

    grad_=Ok_.transpose()*elocs_/double(nsamp);
    // cout << Ok_.sum() << endl;  // All elements of Ok are zero, therefore grad_ = 0
  }


  double Eloc(const VectorXd & v){

    ham_.FindConn(v,mel_,connectors_);

    assert(connectors_.size()==mel_.size());

    logvaldiffs_=(rbm_.LogValDiff(v,connectors_));

    assert(mel_.size()==logvaldiffs_.size());

    double eloc=0;

    for(int i=0;i<logvaldiffs_.size();i++){
      eloc+=mel_[i]*std::exp(0.5*logvaldiffs_(i));
      //cout <<mel_[i] << "  " << std::exp(0.5*logvaldiffs_(i)) << "   -->  "<<eloc<<endl<<flush;
    }

//    cout << "n.mel = "<<mel_.size()<<endl;
//    for(int i=0;i<mel_.size();i++){
//      cout << connectors_[i].size() << "   ,   mel="<<mel_[i] <<" :"<<flush;
//      for(int n=0; n<connectors_[i].size(); n++) cout << "    " << connectors_[i][n]<<flush;
//      cout << endl <<flush;
//    }

    return eloc;
  }

  double ElocMean(){
    return elocmean_;
  }

  void Run(int nsweeps,int niter){
    opt_.Reset();

    for(int i=0;i<niter;i++){
      Sample(nsweeps);
      Gradient();
      UpdateParameters();
      PrintStats(i);
    }
    Iter0_+=niter;
  }

  void UpdateParameters(){
    auto pars=rbm_.GetParameters();
    opt_.Update(grad_,pars);

    rbm_.SetParameters(pars);
  }

  void PrintStats(int i){
    cout<<i+Iter0_<<"  "<<scientific<<elocmean_<<"   "<<grad_.norm()<<" "<<rbm_.GetParameters().array().abs().maxCoeff()<<" ";
    auto Acceptance=sampler_.Acceptance();
    for(int a=0;a<Acceptance.size();a++){
      cout<<Acceptance(a)<<" ";
    }
    cout<<endl;
  }

  //Debug function to check that the logarithm of the derivative is
  //computed correctly
  void CheckDerLog(double eps=1.0e-4){
    sampler_.Reset(true);

    auto ders=rbm_.DerLog(sampler_.Visible());

    auto pars=rbm_.GetParameters();

    for(int i=0;i<npar_;i++){
      pars(i)+=eps;
      rbm_.SetParameters(pars);
      double valp=rbm_.LogVal(sampler_.Visible());

      pars(i)-=2*eps;
      rbm_.SetParameters(pars);
      double valm=rbm_.LogVal(sampler_.Visible());

      pars(i)+=eps;

      double numder=(-valm+valp)/(eps*2);

      if(std::abs(numder-ders(i))>eps*eps){
        cerr<<"Possible error on parameter "<<i<<". Expected: "<<ders(i)<<" Found: "<<numder<<endl;
      }
    }
  }
};


}

#endif
