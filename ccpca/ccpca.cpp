#define EIGEN_NO_DEBUG

#include "ccpca.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>

CCPCA::CCPCA(Eigen::Index const nComponents, bool const standardize)
    : cpca_(CPCA(nComponents, standardize)) {
  featContribs_.resize(0);
  concatMat_.resize(0, 0);
}

Eigen::MatrixXf
CCPCA::fitTransform(Eigen::MatrixXf const &K, Eigen::MatrixXf const &R,
                    bool const autoAlphaSelection, float const alpha,
                    float const varThresRatio, bool parallel,
                    unsigned int const nAlphas, float const maxLogAlpha,
                    bool const keepReports) {
  fit(K, R, autoAlphaSelection, alpha, varThresRatio, parallel, nAlphas,
      maxLogAlpha, keepReports);
  return cpca_.transform(concatMat_);
}

void CCPCA::fit(Eigen::MatrixXf const &K, Eigen::MatrixXf const &R,
                bool const autoAlphaSelection, float const alpha,
                float const varThresRatio, bool parallel,
                unsigned int const nAlphas, float const maxLogAlpha,
                bool const keepReports) {
  if (autoAlphaSelection) {
    fitWithBestAlpha(K, R, varThresRatio, parallel, nAlphas, maxLogAlpha,
                     keepReports);
  } else {
    fitWithManualAlpha(K, R, alpha);
  }
}

void CCPCA::fitWithBestAlpha(Eigen::MatrixXf const &K, Eigen::MatrixXf const &R,
                             float const varThresRatio, bool parallel,
                             unsigned int const nAlphas,
                             float const maxLogAlpha, bool const keepReports) {
  bestAlpha(K, R, varThresRatio, parallel, nAlphas, maxLogAlpha, keepReports);
  cpca_.updateComponents(bestAlpha_);
  featContribs_ = cpca_.getLoading(0);
}

void CCPCA::fitWithManualAlpha(Eigen::MatrixXf const &K,
                               Eigen::MatrixXf const &R, float const alpha) {
  Eigen::Index nSamplesK = K.rows();
  Eigen::Index nSamplesR = R.rows();

  if (K.cols() != R.cols()) {
    std::cerr << "# of rows of K and all matrix in R must be the same."
              << std::endl;
  }

  Eigen::MatrixXf concatMat_(nSamplesK + nSamplesR, K.cols());
  concatMat_ << K, R;

  cpca_.fit(concatMat_, R, alpha);
  featContribs_ = cpca_.getLoading(0);
}

Eigen::MatrixXf CCPCA::transform(Eigen::MatrixXf const &X) {
  return cpca_.transform(X);
}

float CCPCA::bestAlpha(Eigen::MatrixXf const &K, Eigen::MatrixXf const &R,
                       float const varThresRatio, bool parallel,
                       unsigned int const nAlphas, float const maxLogAlpha,
                       bool const keepReports) {
  Eigen::Index nSamplesK = K.rows();
  Eigen::Index nSamplesR = R.rows();

  if (K.cols() != R.cols()) {
    std::cerr << "# of rows of K and all matrix in R must be the same."
              << std::endl;
  }

  Eigen::MatrixXf concatMat_(nSamplesK + nSamplesR, K.cols());
  concatMat_ << K, R;

  cpca_.fit(concatMat_, R, 0.0f);

  Eigen::VectorXf bestProjK = cpca_.transform(K).col(0);
  Eigen::VectorXf bestProjR = cpca_.transform(R).col(0);

  bestAlpha_ = 0.0f;
  auto baseVarK = scaledVar(bestProjK, bestProjR).first;
  auto bestDiscrepancy =
      1.0f / std::max(float(histIntersect(bestProjK, bestProjR)),
                      std::numeric_limits<float>::min());

  reports_.clear();
  if (keepReports) {
    reports_.push_back(std::make_tuple(0.0, float(bestDiscrepancy), baseVarK,
                                       bestProjK, bestProjR,
                                       cpca_.getLoading(0)));
    if (parallel) {
      parallel = false;
      std::cout << "current version keepReports only support non-parallel "
                   "running. parallel is turned off."
                << std::endl;
    }
  }

  auto alphas = cpca_.logspace(-1, maxLogAlpha, nAlphas - 1);

  if (!parallel) {
    // non-parallel ver
    for (auto const &alpha : alphas) {
      cpca_.updateComponents(alpha);

      Eigen::VectorXf tmpProjK = cpca_.transform(K).col(0);
      Eigen::VectorXf tmpProjR = cpca_.transform(R).col(0);

      auto varK = scaledVar(tmpProjK, tmpProjR).first;
      auto discrepancy =
          1.0f / std::max(float(histIntersect(tmpProjK, tmpProjR)),
                          std::numeric_limits<float>::min());

      if (varK >= baseVarK * varThresRatio && discrepancy > bestDiscrepancy) {
        bestDiscrepancy = discrepancy;
        bestAlpha_ = alpha;
      }

      if (keepReports) {
        reports_.push_back(std::make_tuple(alpha, float(bestDiscrepancy), varK,
                                           tmpProjK, tmpProjR,
                                           cpca_.getLoading(0)));
      }
    }
  } else {
    // parallel version
    auto numWorkers = std::max(std::thread::hardware_concurrency(),
                               1u); // obtain max thread num
    std::vector<std::thread> worker;
    auto n = alphas.size();
    std::vector<std::pair<float, float>> varAndDiscpSet(n, {0.0f, 0.0f});

    for (size_t i = 0; i < numWorkers; ++i) {
      worker.emplace_back(
          [&](size_t id) {
            auto r0 = n / numWorkers * id + std::min(n % numWorkers, id);
            auto r1 =
                n / numWorkers * (id + 1) + std::min(n % numWorkers, id + 1);

            for (auto j = r0; j < r1; ++j) {
              Eigen::VectorXf component;
              if (numWorkers > 4) {
                // less mtx locks but more steps
                mtx.lock();
                auto alpha = alphas[j];
                Eigen::MatrixXf diffCov = cpca_.getDiffCov(alpha);
                mtx.unlock();

                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(diffCov);
                component = es.eigenvectors().rightCols(1);
              } else {
                // less steps but more mtx locks
                mtx.lock();
                auto alpha = alphas[j];
                cpca_.updateComponents(alpha);
                component = cpca_.getComponent(0);
                mtx.unlock();
              }

              Eigen::VectorXf tmpProjK = K * component;
              Eigen::VectorXf tmpProjR = R * component;

              auto varK = scaledVar(tmpProjK, tmpProjR).first;
              auto discrepancy =
                  1.0f / std::max(float(histIntersect(tmpProjK, tmpProjR)),
                                  std::numeric_limits<float>::min());
              mtx.lock();
              varAndDiscpSet[j] = {varK, discrepancy};
              mtx.unlock();
            }
          },
          i);
    }
    for (auto &w : worker)
      w.join();

    for (size_t i = 0; i < n; ++i) {
      auto var = varAndDiscpSet[i].first;
      auto discrepancy = varAndDiscpSet[i].second;
      if (var >= baseVarK * varThresRatio && discrepancy > bestDiscrepancy) {
        bestDiscrepancy = discrepancy;
        bestAlpha_ = alphas[i];
      }
    }
  }

  return bestAlpha_;
}

std::pair<float, float> CCPCA::scaledVar(Eigen::VectorXf const &a,
                                         Eigen::VectorXf const &b) {
  float minVal = std::min(a.minCoeff(), b.minCoeff());
  float maxVal = std::max(a.maxCoeff(), b.maxCoeff());
  float range = std::max(maxVal - minVal, std::numeric_limits<float>::min());

  float varA = ((a.array() - a.mean()) / range).array().square().mean();
  float varB = ((b.array() - b.mean()) / range).array().square().mean();

  return {varA, varB};
}

float CCPCA::binWidthScott(Eigen::VectorXf const &vals) {
  auto n = vals.size();
  float sd = 0.0f;
  if (n > 1) {
    sd = std::sqrt((vals.array() - vals.mean()).square().sum() / float(n - 1));
  }
  float denom = std::max(std::pow(float(n), 1.0f / 3.0f),
                         std::numeric_limits<float>::min());
  float binWidth = 3.5f * sd / denom;

  return binWidth;
}

int CCPCA::histIntersect(Eigen::VectorXf const &a, Eigen::VectorXf const &b) {
  float minVal = std::min(a.minCoeff(), b.minCoeff());
  float maxVal = std::max(a.maxCoeff(), b.maxCoeff());
  float range = std::max(maxVal - minVal, std::numeric_limits<float>::min());
  auto nA = a.size();
  auto nB = b.size();

  Eigen::VectorXf tmpA = (a.array() - minVal) / range;
  Eigen::VectorXf tmpB = (b.array() - minVal) / range;
  Eigen::VectorXf tmpAB(nA + nB);
  tmpAB << tmpA, tmpB;

  float binW =
      std::max(binWidthScott(tmpAB), std::numeric_limits<float>::min());
  unsigned int nBins = static_cast<unsigned int>(1.0f / binW) + 1;

  std::vector<int> countsA(nBins, 0);
  std::vector<int> countsB(nBins, 0);

  for (Eigen::Index i = 0; i < nA; ++i) {
    int binIndex = int(tmpA(i) / binW);
    countsA[binIndex]++;
  }
  for (Eigen::Index i = 0; i < nB; ++i) {
    int binIndex = int(tmpB(i) / binW);
    countsB[binIndex]++;
  }

  int histIntersect = 0;
  for (unsigned int i = 0; i < nBins; ++i) {
    histIntersect += std::min(countsA[i], countsB[i]);
  }

  return histIntersect;
}
