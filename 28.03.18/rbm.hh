#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>

#ifndef NQS_RBM_HH
#define NQS_RBM_HH

namespace nqs{

using namespace std;
using namespace Eigen;

class Rbm{

  //number of visible units
  const int nv_;

  //number of hidden units
  const int nh_;

  //number of parameters
  int npar_;

  //weights
  MatrixXd W_;

  //visible units bias
  VectorXd a_;

  //hidden units bias
  VectorXd b_;

  //Auxiliary variables
  VectorXd thetas_;
  VectorXd lnthetas_;
  VectorXd thetasnew_;
  VectorXd lnthetasnew_;

public:

  Rbm(int nv,int nh):nv_(nv),nh_(nh),W_(nv,nh),a_(nv),b_(nh),thetas_(nh),lnthetas_(nh),thetasnew_(nh),lnthetasnew_(nh){

    npar_=nv_+nh_+nv_*nh_;

    cout<<"# RBM Initizialized with nvisible = "<<nv_<<" and nhidden = "<<nh_<<endl;

  }

  int Nvisible()const{
    return nv_;
  }

  int Nhidden()const{
    return nh_;
  }

  int Npar()const{
    return npar_;
  }


  void ProbHiddenGivenVisible(const VectorXd & v,VectorXd & probs){
    logistic(W_.transpose()*v+b_,probs);
  }

  void ProbVisibleGivenHidden(const VectorXd & h,VectorXd & probs){
    logistic(W_*h+a_,probs);
  }

  double Energy(const VectorXd & v,const VectorXd & h){
    return -(h.dot(W_.transpose()*v)+v.dot(a_)+h.dot(b_));
  }

  void InitRandomPars(int seed,double sigma){
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0,sigma);

    for(int i=0;i<nv_;i++){
      a_(i)=distribution(generator);
    }
    for(int j=0;j<nh_;j++){
      b_(j)=distribution(generator);
    }

    for(int i=0;i<nv_;i++){
      for(int j=0;j<nh_;j++){
        W_(i,j)=distribution(generator);
      }

    }

  }

  VectorXd DerLog(const VectorXd & v){
    VectorXd der(npar_);

    for(int k=0;k<nv_;k++){
      der(k)=v(k);
    }

    logistic(W_.transpose()*v+b_,lnthetas_);

    for(int k=nv_;k<(nv_+nh_);k++){
      der(k)=lnthetas_(k-nv_);
    }

    int k=nv_+nh_;
    for(int i=0;i<nv_;i++){
      for(int j=0;j<nh_;j++){
        der(k)=lnthetas_(j)*v(i);
        k++;
      }
    }
    return der;
  }

  VectorXd GetParameters(){

    VectorXd pars(npar_);

    for(int k=0;k<nv_;k++){
      pars(k)=a_(k);
    }
    for(int k=nv_;k<(nv_+nh_);k++){
      pars(k)=b_(k-nv_);
    }

    int k=nv_+nh_;
    for(int i=0;i<nv_;i++){
      for(int j=0;j<nh_;j++){
        pars(k)=W_(i,j);
        k++;
      }
    }

    return pars;
  }

  void SetParameters(const VectorXd & pars){
    for(int k=0;k<nv_;k++){
      a_(k)=pars(k);
    }
    for(int k=(nv_);k<(nv_+nh_);k++){
      b_(k-nv_)=pars(k);
    }
    int k=(nv_+nh_);

    for(int i=0;i<nv_;i++){
      for(int j=0;j<nh_;j++){
        W_(i,j)=pars(k);
        k++;
      }
    }

  }

  //Value of the logarithm of the RBM probability
  double LogVal(const VectorXd & v){
    ln1pexp(W_.transpose()*v+b_,lnthetas_);

    return v.dot(a_)+lnthetas_.sum();
  }

  //Difference between logarithms of values, when one or more visible variables are being flipped
  VectorXd LogValDiff(const VectorXd & v,const vector<vector<int> >  & toflip){

    VectorXd logvaldiffs;

    const int nconn=toflip.size();
    logvaldiffs.resize(nconn);

    thetas_=(W_.transpose()*v+b_);
    ln1pexp(thetas_,lnthetas_);

    double logtsum=lnthetas_.sum();

    for(int k=0;k<nconn;k++){
      logvaldiffs(k)=0;

      if(toflip[k].size()!=0){

        thetasnew_=thetas_;

        for(int s=0;s<toflip[k].size();s++){
          const int sf=toflip[k][s];

          logvaldiffs(k)+=a_(sf)*(1.-2*v(sf));

          thetasnew_+=(1.-2.*v(sf))*W_.row(sf);
        }

        ln1pexp(thetasnew_,lnthetasnew_);
        logvaldiffs(k)+=lnthetasnew_.sum() - logtsum;

      }

    }
    return logvaldiffs;
  }


  void logistic(const VectorXd & x,VectorXd & y){
    for(int i=0;i<x.size();i++){
      y(i)=logistic(x(i));
    }
  }

  inline double logistic(double x)const{
    return 1./(1.+std::exp(-x));
  }

  void ln1pexp(const VectorXd & x,VectorXd & y){
    for(int i=0;i<x.size();i++){
      y(i)=ln1pexp(x(i));
    }
  }

  //log(1+e^x)
  inline double ln1pexp(double x)const{
    if(x>30){
      return x;
    }
    // return std::log(1.+std::exp(x));
    return std::log1p(std::exp(x));
  }


};


}

#endif
