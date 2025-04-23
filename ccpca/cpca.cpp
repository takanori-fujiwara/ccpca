#include "cpca.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>

CPCA::CPCA(Eigen::Index const nComponents, bool const standardize)
    : nComponents_(nComponents), standardize_(standardize) {
  initialize();
}

void CPCA::initialize() { components_.resize(0, 0); }

Eigen::MatrixXf
CPCA::fitTransform(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
                   bool const autoAlphaSelection, float const alpha,
                   float const eta, float const convergenceRatio,
                   unsigned int maxIter, bool const keepReports) {
  fit(fg, bg, autoAlphaSelection, alpha, eta, convergenceRatio, maxIter,
      keepReports);
  return transform(fg_);
}

void CPCA::fit(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
               bool const autoAlphaSelection, float const alpha,
               float const eta, float const convergenceRatio,
               unsigned int maxIter, bool const keepReports) {
  if (autoAlphaSelection) {
    fitWithBestAlpha(fg, bg, alpha, eta, convergenceRatio, maxIter,
                     keepReports);
  } else {
    fitWithManualAlpha(fg, bg, alpha);
  }
}

void CPCA::fitWithManualAlpha(Eigen::MatrixXf const &fg,
                              Eigen::MatrixXf const &bg, float const alpha) {
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
  eigenvalues_ = es.eigenvalues().real().tail(nComponents_).reverse();
  totalPosEigenvalue_ = (es.eigenvalues().real().array() < 0)
                            .select(0, es.eigenvalues().real())
                            .sum();
  loadings_ = components_.array().rowwise() * eigenvalues_.array().abs().sqrt();
}

void CPCA::fitWithBestAlpha(Eigen::MatrixXf const &fg,
                            Eigen::MatrixXf const &bg, float const initAlpha,
                            float const eta, float const convergenceRatio,
                            unsigned int const maxIter,
                            bool const keepReports) {
  bestAlpha(fg, bg, initAlpha, eta, convergenceRatio, maxIter, keepReports);
  // updateComponents(bestAlpha_);
  fitWithManualAlpha(fg, bg, bestAlpha_);
}

void CPCA::updateComponents(float const alpha) {
  if (components_.cols() == 0) {
    std::cerr << "Run fit() at least once before updateComponents()"
              << std::endl;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(fgCov_ - alpha * bgCov_);
  components_ = es.eigenvectors().rightCols(nComponents_).rowwise().reverse();

  eigenvalues_ = es.eigenvalues().real().tail(nComponents_).reverse();
  totalPosEigenvalue_ = (es.eigenvalues().real().array() < 0)
                            .select(0, es.eigenvalues().real())
                            .sum();
  loadings_ = components_.array().rowwise() * eigenvalues_.array().abs().sqrt();
}

Eigen::MatrixXf CPCA::transform(Eigen::MatrixXf const &X) {
  if (components_.cols() == 0) {
    std::cerr << "Run fit() before transform()" << std::endl;
  }
  return X * components_;
}

float CPCA::bestAlpha(Eigen::MatrixXf const &fg, Eigen::MatrixXf const &bg,
                      float const initAlpha, float const eta,
                      float const convergenceRatio, unsigned int const maxIter,
                      bool const keepReports) {
  reports_.clear();
  float alpha = initAlpha;
  fit(fg, bg, false, alpha);

  // method 1. discard minor eigenvectors to avoid singular
  // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> esQ(bgCov_);
  // float ratioToKeep = 0.999f;
  // Eigen::RowVectorXf eigenvalues = esQ.eigenvalues().real().reverse();
  // float targetTotalEigenVal = eigenvalues.sum() * ratioToKeep;
  // Eigen::Index nEigenVectorsToKeep = 0;
  // while (targetTotalEigenVal > 0) {
  //     targetTotalEigenVal -= eigenvalues(nEigenVectorsToKeep);
  //     nEigenVectorsToKeep++;
  // }
  // float ratioToKeep = 0.9f;
  // Eigen::Index nEigenVectorsToKeep = Eigen::Index(bg.cols() * ratioToKeep);
  //
  // std::cout << fg.cols() << " " << nEigenVectorsToKeep << std::endl;
  //
  // Eigen::MatrixXf Q =
  //     esQ.eigenvectors().rightCols(nEigenVectorsToKeep).rowwise().reverse();
  // Eigen::MatrixXf fgCovQ = Q.adjoint() * fgCov_ * Q;
  // Eigen::MatrixXf bgCovQ = Q.adjoint() * bgCov_ * Q;
  //
  // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> esU(fgCovQ - alpha *
  // bgCovQ); Eigen::MatrixXf U =
  //     esU.eigenvectors().rightCols(nComponents_).rowwise().reverse();
  //
  // if (keepReports) {
  //   reports_.push_back(alpha);
  // }
  //
  // for (unsigned int i = 0; i < maxIter; ++i) {
  //   float fgTr = (U.adjoint() * fgCovQ * U).trace();
  //   float bgTr = (U.adjoint() * bgCovQ * U).trace();
  //   bgTr = std::fmax(bgTr, std::numeric_limits<float>::min());
  //   alpha = fgTr / bgTr;
  //
  //   // update U
  //   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> esU(fgCovQ - alpha *
  //   bgCovQ); U =
  //   esU.eigenvectors().rightCols(nComponents_).rowwise().reverse();
  //
  //   if (keepReports) {
  //     reports_.push_back(alpha);
  //   }
  // }
  // bestAlpha_ = alpha;

  // method 2: add small constant to diag of bgCov_ to avoid singular
  bgCov_ += Eigen::MatrixXf::Identity(bgCov_.rows(), bgCov_.cols()) * eta;

  if (keepReports) {
    reports_.push_back(alpha);
  }

  for (unsigned int i = 0; i < maxIter; ++i) {
    float fgTr = (components_.adjoint() * fgCov_ * components_).trace();
    float bgTr = (components_.adjoint() * bgCov_ * components_).trace();
    bgTr = std::fmax(bgTr, std::numeric_limits<float>::min());

    float prevAlpha = alpha;
    alpha = fgTr / bgTr;
    updateComponents(alpha);
    if (keepReports) {
      reports_.push_back(alpha);
    }

    if (std::abs(prevAlpha - alpha) / alpha < convergenceRatio)
      break;
  }
  bestAlpha_ = alpha;

  return bestAlpha_;
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
