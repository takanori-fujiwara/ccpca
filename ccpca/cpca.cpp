#include "cpca.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <chrono>
#include <iostream>
#include <utility>

CPCA::CPCA(Eigen::Index const nComponents, bool const standardize)
    : nComponents_(nComponents), standardize_(standardize) {
  initialize();
}

void CPCA::initialize() { components_.resize(0, 0); }

Eigen::MatrixXf CPCA::fitTransform(Eigen::MatrixXf const &fg,
                                   Eigen::MatrixXf const &bg,
                                   float const alpha) {
  fit(fg, bg, alpha);
  return transform(fg_);
}

void CPCA::fit(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
               float const alpha) {
  fg_ = fg;
  bg_ = bg;

  Eigen::Index fgSize = fg_.size();
  Eigen::Index bgSize = bg_.size();

  if (fgSize == 0 && bgSize == 0) {
    std::cerr << "Both target and background matrices are empty." << std::endl;
  } else if (fgSize == 0) {
    // the result will be the same with when alpha is +inf
    fg_ = Eigen::MatrixXf::Zero(1, bg_.cols());
  } else if (bgSize == 0) {
    // the result will be the same with ordinary PCA
    bg_ = Eigen::MatrixXf::Zero(1, fg_.cols());
  }

  Eigen::Index nFeaturesFg = fg_.cols();
  Eigen::Index nFeaturesBg = bg_.cols();

  if (nFeaturesFg != nFeaturesBg) {
    std::cerr << "# of features of foregraound and background must be the same."
              << std::endl;
  }

  fg_ = fg_.rowwise() - fg_.colwise().mean();
  bg_ = bg_.rowwise() - bg_.colwise().mean();

  if (standardize_) {
    Eigen::RowVectorXf fgStd = fg_.array().square().colwise().mean().sqrt();
    Eigen::RowVectorXf bgStd = bg_.array().square().colwise().mean().sqrt();

    fg_ = fg_.array().rowwise() / fgStd.array();
    bg_ = bg_.array().rowwise() / bgStd.array();

    // NaN to 0.0f
    fg_ = fg_.unaryExpr([](float v) { return std::isfinite(v) ? v : 0.0f; });
    bg_ = bg_.unaryExpr([](float v) { return std::isfinite(v) ? v : 0.0f; });
  }

  fgCov_ = (fg_.adjoint() * fg_) /
           std::fmax(float(fg_.rows() - 1), std::numeric_limits<float>::min());
  bgCov_ = (bg_.adjoint() * bg_) /
           std::fmax(float(bg_.rows() - 1), std::numeric_limits<float>::min());

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(fgCov_ - alpha * bgCov_);
  components_ = es.eigenvectors().rightCols(nComponents_).rowwise().reverse();
  Eigen::RowVectorXf eigenvalues =
      es.eigenvalues().real().tail(nComponents_).reverse();
  loadings_ = components_.array().rowwise() * eigenvalues.array().abs().sqrt();
}

void CPCA::updateComponents(float const alpha) {
  if (components_.cols() == 0) {
    std::cerr << "Run fit() at least once before updateComponents()"
              << std::endl;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(fgCov_ - alpha * bgCov_);
  components_ = es.eigenvectors().rightCols(nComponents_).rowwise().reverse();

  Eigen::RowVectorXf eigenvalues =
      es.eigenvalues().real().tail(nComponents_).reverse();
  loadings_ = components_.array().rowwise() * eigenvalues.array().abs().sqrt();
}

Eigen::MatrixXf CPCA::transform(Eigen::MatrixXf const &X) {
  if (components_.cols() == 0) {
    std::cerr << "Run fit() before transform()" << std::endl;
  }
  return X * components_;
}

std::vector<float> CPCA::logspace(float const start, float const end,
                                  unsigned int const num, float const base) {
  float realStart = std::pow(base, start);
  float realBase = std::pow(
      base, (end - start) /
                std::fmax(float(num - 1), std::numeric_limits<float>::min()));

  std::vector<float> result;
  result.reserve(num);
  std::generate_n(
      std::back_inserter(result), num, [=]() mutable throw()->float {
        float val = realStart;
        realStart *= realBase;
        return val;
      });
  return result;
}

// TODO: implement semi-automatic selection of alpha of the original cpca
// (but not necessary for ccPCA)

std::vector<float> CPCA::findSpectalAlphas(unsigned int const nAlphasToReturn,
                                           unsigned int const nAlphas,
                                           float const maxLogAlpha) {
  std::vector<float> alphas;
  Eigen::MatrixXf affinityMat = createAffinityMatrix(fg_, nAlphas, maxLogAlpha);

  // TODO: implement rest of here

  return alphas;
}

Eigen::MatrixXf CPCA::createAffinityMatrix(Eigen::MatrixXf const &X,
                                           unsigned int const nAlphas,
                                           float const maxLogAlpha) {
  std::vector<float> alphas;
  alphas.reserve(nAlphas + 1);
  alphas.push_back(0.0f);

  auto logspaceAlphas = logspace(-1.0f, maxLogAlpha, nAlphas);
  alphas.insert(alphas.end(), logspaceAlphas.begin(), logspaceAlphas.end());

  auto k = alphas.size();
  Eigen::MatrixXf affinityMat =
      0.5 * Eigen::MatrixXf::Identity(Eigen::Index(k), Eigen::Index(k));

  std::vector<Eigen::MatrixXf> subspaces;
  subspaces.reserve(k);
  for (auto const &alpha : alphas) {
    updateComponents(alpha);
    Eigen::MatrixXf proj = transform(X);
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(proj);
    Eigen::MatrixXf Q = qr.householderQ();
    subspaces.push_back(qr.householderQ());
  }

  for (size_t i = 0; i < k; ++i) {
    for (size_t j = i + 1; j < k; ++j) {
      Eigen::BDCSVD<Eigen::MatrixXf> svd(subspaces[i] * subspaces[j],
                                         Eigen::ComputeThinU |
                                             Eigen::ComputeThinV);
      Eigen::VectorXf s = svd.singularValues();
      affinityMat(Eigen::Index(i), Eigen::Index(j)) = s(0) * s(1);
    }
  }

  affinityMat = affinityMat + affinityMat.transpose();
  // NaN to 0.0f
  affinityMat = affinityMat.unaryExpr(
      [](float v) { return std::isfinite(v) ? v : 0.0f; });

  return affinityMat;
}

Eigen::MatrixXf CPCA::getComponents() { return components_; }

Eigen::VectorXf CPCA::getComponent(Eigen::Index const index) {
  return components_.col(index);
}

Eigen::MatrixXf CPCA::getLoadings() { return loadings_; }

Eigen::VectorXf CPCA::getLoading(Eigen::Index const index) {
  return loadings_.col(index);
}

Eigen::MatrixXf CPCA::getCurrentFg() { return fg_; }

Eigen::MatrixXf CPCA::getDiffCov(float const alpha) {
  return fgCov_ - alpha * bgCov_;
}
