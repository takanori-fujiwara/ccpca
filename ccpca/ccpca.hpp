#ifndef CCPCA_HPP
#define CCPCA_HPP

#include "cpca.hpp"
#include <Eigen/Core>
#include <mutex>
#include <tuple>
#include <vector>

class CCPCA : public CPCA {
public:
  CCPCA(Eigen::Index const nComponents = 2, bool const standardize = true);

  Eigen::MatrixXf
  fitTransform(Eigen::MatrixXf const &K, Eigen::MatrixXf const &R,
               bool const autoAlphaSelection = true, float const alpha = 0.0f,
               float const varThresRatio = 0.5f, bool parallel = true,
               unsigned int const nAlphas = 40, float const maxLogAlpha = 3.0f,
               bool const keepReports = false);
  void fit(Eigen::MatrixXf const &K, Eigen::MatrixXf const &R,
           bool const autoAlphaSelection = true, float const alpha = 0.0f,
           float const varThresRatio = 0.5f, bool parallel = true,
           unsigned int const nAlphas = 40, float const maxLogAlpha = 3.0f,
           bool const keepReports = false);
  void fitWithBestAlpha(Eigen::MatrixXf const &K, Eigen::MatrixXf const &R,
                        float const varThresRatio = 0.5f, bool parallel = true,
                        unsigned int const nAlphas = 40,
                        float const maxLogAlpha = 3.0f,
                        bool const keepReports = false);
  void fitWithManualAlpha(Eigen::MatrixXf const &K, Eigen::MatrixXf const &R,
                          float const alpha = 0.0f);
  Eigen::MatrixXf transform(Eigen::MatrixXf const &X);
  float bestAlpha(Eigen::MatrixXf const &K, Eigen::MatrixXf const &R,
                  float const varThresRatio = 0.5f, bool parallel = true,
                  unsigned int const nAlphas = 40,
                  float const maxLogAlpha = 3.0f,
                  bool const keepReports = false);

  // first loading is the same with featContribs_
  Eigen::VectorXf getFeatContribs() { return featContribs_; };
  Eigen::VectorXf getScaledFeatContribs() {
    float absMax = featContribs_.array().abs().maxCoeff();
    return featContribs_ / absMax;
  };
  Eigen::MatrixXf getComponents() { return cpca_.getComponents(); };
  Eigen::VectorXf getComponent(Eigen::Index const index) {
    return cpca_.getComponent(index);
  };
  Eigen::VectorXf getFirstComponent() { return getComponent(0); };
  Eigen::RowVectorXf getEigenvalues() { return cpca_.getEigenvalues(); };
  float getEigenvalue(Eigen::Index const index) {
    return cpca_.getEigenvalue(index);
  };
  Eigen::MatrixXf getLoadings() { return cpca_.getLoadings(); };
  Eigen::VectorXf getLoading(Eigen::Index const index) {
    return cpca_.getLoading(index);
  };
  float getBestAlpha() { return bestAlpha_; };
  std::vector<std::tuple<float, float, float, Eigen::VectorXf, Eigen::VectorXf,
                         Eigen::VectorXf>>
  getReports() {
    return reports_;
  };

private:
  CPCA cpca_;
  float bestAlpha_;
  Eigen::MatrixXf concatMat_;
  Eigen::VectorXf featContribs_;

  // this tuple is used because pybind11 cannot handle structure well
  // the order is alpha, discrepancy score, var, projK, projR, loadings
  std::vector<std::tuple<float, float, float, Eigen::VectorXf, Eigen::VectorXf,
                         Eigen::VectorXf>>
      reports_;
  std::mutex mtx;
  std::pair<float, float> scaledVar(Eigen::VectorXf const &a,
                                    Eigen::VectorXf const &b);
  float binWidthScott(Eigen::VectorXf const &vals);
  int histIntersect(Eigen::VectorXf const &a, Eigen::VectorXf const &b);
};

#endif // CCPCA_HPP
