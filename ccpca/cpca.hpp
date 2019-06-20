#ifndef CPCA_HPP
#define CPCA_HPP

#include <Eigen/Core>
#include <vector>

class CPCA {
public:
  CPCA(Eigen::Index const nComponents = 2, bool const standardize = true);

  void initialize();
  Eigen::MatrixXf fitTransform(Eigen::MatrixXf const &fg,
                               Eigen::MatrixXf const &bg, float const alpha);
  void fit(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
           float const alpha);
  void updateComponents(float const alpha);
  Eigen::MatrixXf transform(Eigen::MatrixXf const &X);
  std::vector<float> logspace(float const start, float const end,
                              unsigned int const num, float const base = 10.0f);
  Eigen::MatrixXf getComponents();
  Eigen::VectorXf getComponent(Eigen::Index const index);
  Eigen::MatrixXf getLoadings();
  Eigen::VectorXf getLoading(Eigen::Index const index);
  Eigen::MatrixXf getCurrentFg();
  Eigen::MatrixXf getDiffCov(float const alpha);

private:
  Eigen::Index nComponents_;
  bool standardize_;
  Eigen::MatrixXf components_;
  Eigen::MatrixXf loadings_;
  Eigen::MatrixXf fg_;
  Eigen::MatrixXf bg_;
  Eigen::MatrixXf fgCov_;
  Eigen::MatrixXf bgCov_;
  Eigen::MatrixXf affinityMat_;

  std::vector<float> findSpectalAlphas(unsigned int const nAlphasToReturn,
                                       unsigned int const nAlphas,
                                       float const maxLogAlpha);
  Eigen::MatrixXf createAffinityMatrix(Eigen::MatrixXf const &X,
                                       unsigned int const nAlphas,
                                       float const maxLogAlpha);
};

#endif // CPCA_HPP
