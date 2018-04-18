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
// #include "box1d.hh"

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
      // PrintStats(0);
      vsamp_.row(i)=sampler_.Visible();
    }
  }

  void Gradient(){
    const int nsamp=vsamp_.rows();
    elocs_.resize(nsamp);
    Ok_.resize(nsamp,rbm_.Npar());
    for(int i=0;i<nsamp;i++){
      elocs_(i)=Eloc(vsamp_.row(i));
      Ok_.row(i)=rbm_.DerLog(vsamp_.row(i));
    }
    elocmean_=elocs_.mean();
    Ok_=Ok_.rowwise() - Ok_.colwise().mean();
    elocs_-=elocmean_*VectorXd::Ones(nsamp);
    grad_=Ok_.transpose()*elocs_/double(nsamp);
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

  // Converts (reversed, i.e. left to right) byte representing x-position into a meshpoint
  int X_to_i(const VectorXd & Xin) {
    int i=0; int pow2n = 1;
    for(int n=0; n<Xin.size(); n++) {i += int(Xin(n)+0.1) * pow2n; pow2n*=2;}
    return i;
  }

  void PrintStats(int i){
    // cout << i+Iter0_ << '\t';
    // // Print 'spin' states
    // cout << '\t' << "| X = ";
    // for(int i=0; i<sampler_.Visible().size(); i++){
    //   cout << sampler_.Visible()[i];
    // }
    // // Print mesh-point
    // cout << '\t' << "| i = " << X_to_i(sampler_.Visible());
    //
    // // Print x-position
    // float x0=-10.; float dx=120./1024;
    // float x_pos = x0 + X_to_i(sampler_.Visible()) * dx;
    // cout << '\t' << "| x_pos = " << x_pos;

    // // Plotting the wave function at each step of variational iteration
    // // (Eq. 2.3) Beijing notes
    // double first_sum = 0.;
    // double second_sum = 0.;
    // double product_sum = 1.;
    // //
    // Get current parameters of the network
    VectorXd a_, b_, pars; MatrixXd W_;
    VectorXd v_, h_;
    rbm_.GetParameters();
    int nv_ = 10, nh_= 10; int npar_=nv_+nh_+nv_*nh_;
    v_.resize(nv_); h_.resize(nh_);
    a_.resize(nv_); b_.resize(nh_); pars.resize(npar_); W_.resize(nv_, nh_);
    v_ = sampler_.Visible(), h_ = sampler_.Hidden();
    // for(int k=0;k<nv_;k++){  // Visible node bias parameters
    //   a_(k) = pars(k);
    // }
    // for(int k=nv_;k<(nv_+nh_);k++){  // Hidden node biases
    //   b_(k-nv_)= pars(k);
    // }
    // int k=nv_+nh_;
    // for(int i=0;i<nv_;i++){  // Weights between visible and hidden
    //   for(int j=0;j<nh_;j++){
    //     W_(i,j) = pars(k);
    //     k++;
    //   }
    // }
    //
    // // Calculate wave-function using Equation (2.7), page 10, Beijing notes
    // for(int i=0; i<v_.size(); i++){  // First sum inside exponential
    //     first_sum += v_(i) * a_(i);
    // }
    // // Product sum in eq (2.7)
    // for(int j=0; j<h_.size(); j++){  // hidden
    //   second_sum = 0.;  // reset
    //   for(int i=0; i<v_.size(); i++){  // weights
    //     second_sum += W_(i,j) * v_(i);  // square brackets of cosh
    //   }
    //   second_sum += b_(j);
    //   product_sum *= 2. * cosh(second_sum);
    // }
    // double F = exp(first_sum) * product_sum;  // Finalize equation (2.7)
    // double psi = sqrt(F);  // Equation (2.9) Beijing
    // cout << "| psi = " << psi; //<< endl; //<< endl;
    //
    // // // cout << i+Iter0_ << '\t';
    // // //
    // cout << '\t' << "| h_ = ";
    // for(int i=0; i<sampler_.Hidden().size(); i++){
    //   cout << sampler_.Hidden()[i];
    // }
    //
    // // Print 'spin' states
    // cout << '\t' << "| v_ = ";
    // for(int i=0; i<sampler_.Visible().size(); i++){
    //   cout << sampler_.Visible()[i];
    // }
    // cout << '\t' << "| elocmean_ = " << elocmean_; //<< endl;
    // //
    // // Print x-position
    // float x0=-10.; float dx=120./1024;
    // float x_pos = x0 + X_to_i(sampler_.Visible()) * dx;
    // cout << '\t' << "| x_pos = " << x_pos;
    // cout << endl;
    //
    if (i+Iter0_ == 9998){
      cout << '\t' << "| a_ = " << a_ << endl;
      cout << "b_ = " << b_ << endl;
      cout << "W_ = " << W_ << endl;
    };
    //
    // cout<<i+Iter0_<<"  "<<scientific<<elocmean_<<"   "<<grad_.norm()<<" "<<rbm_.GetParameters().array().abs().maxCoeff()<<" ";
    //
    // std::ofstream outfile;
    // outfile.open("iter_elocmean.dat", std::ios_base::app);
    // outfile << i+Iter0_<<";\t" << elocmean_ << '\n';

    auto Acceptance=sampler_.Acceptance();
    for(int a=0;a<Acceptance.size();a++){
      // cout<<Acceptance(a)<<" ";
    }
    // cout<<endl;
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
