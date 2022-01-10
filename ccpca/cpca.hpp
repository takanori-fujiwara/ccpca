#ifndef CPCA_HPP
#define CPCA_HPP

#include <Eigen/Core>
#include <vector>

class CPCA {
public:
  CPCA(Eigen::Index const nComponents = 2, bool const standardize = true);

  void initialize();
  Eigen::MatrixXf
  fitTransform(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
               bool const autoAlphaSelection = true, float const alpha = 0.0f,
               float const eta = 1e-3f, float const convergenceRatio = 1e-2f,
               unsigned int const maxIter = 10, bool const keepReports = false);
  void fit(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
           bool const autoAlphaSelection = true, float const alpha = 0.0f,
           float const eta = 1e-3f, float const convergenceRatio = 1e-2f,
           unsigned int maxIter = 10, bool const keepReports = false);
  void fitWithManualAlpha(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
                          float const alpha);
  void fitWithBestAlpha(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
                        float const initAlpha = 0.0f, float const eta = 1e-3f,
                        float const convergenceRatio = 1e-2f,
                        unsigned int const maxIter = 10,
                        bool const keepReports = false);
  void updateComponents(float const alpha);
  Eigen::MatrixXf transform(Eigen::MatrixXf const &X);
  float bestAlpha(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
                  float const initAlpha = 0.0f, float const eta = 1e-3f,
                  float const convergenceRatio = 1e-2f,
                  unsigned int const maxIter = 10,
                  bool const keepReports = false);
  std::vector<float> logspace(float const start, float const end,
                              unsigned int const num, float const base = 10.0f);
  Eigen::MatrixXf getComponents() { return components_; }
  Eigen::VectorXf getComponent(Eigen::Index const index) {
    return components_.col(index);
  }
  Eigen::RowVectorXf getEigenvalues() { return eigenvalues_; }
  float getEigenvalue(Eigen::Index const index) {
    return eigenvalues_(index);
  }
  Eigen::MatrixXf getLoadings() { return loadings_; }
  Eigen::VectorXf getLoading(Eigen::Index const index) {
    return loadings_.col(index);
  }
  Eigen::MatrixXf getCurrentFg() { return fg_; }
  Eigen::MatrixXf getDiffCov(float const alpha) {
    return fgCov_ - alpha * bgCov_;
  }
  float getBestAlpha() { return bestAlpha_; }
  std::vector<float> getReports() { return reports_; }

private:
  Eigen::Index nComponents_;
  bool standardize_;
  Eigen::MatrixXf components_;
  Eigen::RowVectorXf eigenvalues_;
  Eigen::MatrixXf loadings_;
  Eigen::MatrixXf fg_;
  Eigen::MatrixXf bg_;
  Eigen::MatrixXf fgCov_;
  Eigen::MatrixXf bgCov_;
  Eigen::MatrixXf affinityMat_;
  std::vector<float> reports_;
  float bestAlpha_;

  std::vector<float> findSpectalAlphas(unsigned int const nAlphasToReturn,
                                       unsigned int const nAlphas,
                                       float const maxLogAlpha);
  Eigen::MatrixXf createAffinityMatrix(Eigen::MatrixXf const &X,
                                       unsigned int const nAlphas,
                                       float const maxLogAlpha);
};

#endif // CPCA_HPP
