/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "cuda.h"
#include "cuda_util.h"
#include "io_manager.h"
#include "rng.h"
#include "rpucuda_powstep_device.h"
#include "rpucuda_pulsed.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

namespace {

using namespace RPU;

class RPUCudaPowStepTestFixture : public ::testing::TestWithParam<float> {
public:
  void SetUp() {

    context = &context_container;

    x_size = 100;
    d_size = 110;
    repeats = 100;
    K = 10;
    m_batch = 100; // for the batched versions
    dim3 = 4;

    num_t bmin = -0.7;
    num_t bmax = 0.7;

    PulsedMetaParameter<num_t> p;
    PowStepRPUDeviceMetaParameter<num_t> dp;
    IOMetaParameter<num_t> p_io;

    p_io.out_noise = 0.0; // no noise in output;
    dp.dw_min_std = 0.0;

    p.up.desired_BL = K;
    p.up.pulse_type = PulseType::DeterministicImplicit;
    p.up.d_res_implicit = 0.01;
    p.up.x_res_implicit = 0.01;

    dp.w_max = 1;
    dp.w_min = -1;
    dp.w_min_dtod = 0.0;
    dp.w_max_dtod = 0.0;

    dp.dw_min = 0.01;

    // peripheral circuits specs
    p_io.inp_res = -1;
    p_io.inp_sto_round = false;
    p_io.out_res = -1;

    p_io.noise_management = NoiseManagementType::AbsMax;
    p_io.bound_management = BoundManagementType::None;

    p.f_io = p_io;
    p.b_io = p_io;

    p.up.update_management = true;
    p.up.update_bl_management = true;

    dp.lifetime = 0;
    dp.lifetime_dtod = 0;

    dp.diffusion = 0.00;
    dp.diffusion_dtod = 0.0;

    // slope
    dp.ps_gamma = GetParam();
    dp.ps_gamma_dtod = 0.01;
    dp.ps_gamma_up_down = 0.1;
    dp.ps_gamma_up_down_dtod = 0.01;

    // dp.print();

    rx.resize(x_size * m_batch);
    rd.resize(d_size * m_batch);

    num_t lr = 0.4;

    layer_pulsed = RPU::make_unique<RPUPulsed<num_t>>(x_size, d_size);
    layer_pulsed->populateParameter(&p, &dp);
    layer_pulsed->setLearningRate(lr);
    layer_pulsed->setWeightsUniformRandom(bmin, bmax);

    // layer_pulsed->disp();

    // culayer
    culayer_pulsed = RPU::make_unique<RPUCudaPulsed<num_t>>(context, *layer_pulsed);
    // culayer_pulsed->disp();

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1., 1.);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weigt matrix
    for (int i = 0; i < x_size * m_batch; i++)
      rx[i] = (num_t)urnd();

    for (int j = 0; j < d_size * m_batch; j++) {
      rd[j] = (num_t)urnd();
    }

    x_cuvec = RPU::make_unique<CudaArray<num_t>>(context, x_size);
    x_vec.resize(x_size);

    d_cuvec = RPU::make_unique<CudaArray<num_t>>(context, d_size);
    d_vec.resize(d_size);

    x_cuvec_batch = RPU::make_unique<CudaArray<num_t>>(context, m_batch * x_size);
    x_vec_batch.resize(m_batch * x_size);

    d_cuvec_batch = RPU::make_unique<CudaArray<num_t>>(context, m_batch * d_size);
    d_vec_batch.resize(m_batch * d_size);
  }

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  std::unique_ptr<RPUPulsed<num_t>> layer_pulsed;
  std::unique_ptr<RPUCudaPulsed<num_t>> culayer_pulsed;
  std::vector<num_t> x_vec, x_vec_batch, d_vec, d_vec_batch, rx, rd;
  std::unique_ptr<CudaArray<num_t>> x_cuvec, x_cuvec_batch;
  std::unique_ptr<CudaArray<num_t>> d_cuvec, d_cuvec_batch;
  int x_size;
  int d_size;
  int repeats;
  int K;
  int dim3;
  int m_batch;

  num_t noise_value;
};

// types
INSTANTIATE_TEST_CASE_P(GammaTest, RPUCudaPowStepTestFixture, ::testing::Values(1.2, 1.0));

#define RPU_TEST_UPDATE(CUFUN, FUN, NLOOP)                                                         \
  this->context->synchronizeDevice();                                                              \
                                                                                                   \
  num_t **refweights = Array_2D_Get<num_t>(this->d_size, this->x_size);                            \
  num_t **w = this->layer_pulsed->getWeights();                                                    \
  for (int i = 0; i < this->d_size; i++) {                                                         \
    for (int j = 0; j < this->x_size; j++) {                                                       \
      refweights[i][j] = w[i][j];                                                                  \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  int nloop = NLOOP;                                                                               \
  double cudur = 0, dur = 0;                                                                       \
  int count_diff = 0;                                                                              \
  auto start_time = std::chrono::high_resolution_clock::now();                                     \
  for (int loop = 0; loop < nloop; loop++) {                                                       \
                                                                                                   \
    start_time = std::chrono::high_resolution_clock::now();                                        \
    this->culayer_pulsed->CUFUN;                                                                   \
    this->context->synchronizeDevice();                                                            \
    auto end_time = std::chrono::high_resolution_clock::now();                                     \
    cudur += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(); \
                                                                                                   \
    start_time = std::chrono::high_resolution_clock::now();                                        \
    this->layer_pulsed->FUN;                                                                       \
    end_time = std::chrono::high_resolution_clock::now();                                          \
    dur += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();   \
                                                                                                   \
    num_t **cuweights = this->culayer_pulsed->getWeights();                                        \
    num_t **weights = this->layer_pulsed->getWeights();                                            \
    this->context->synchronizeDevice();                                                            \
    count_diff = 0;                                                                                \
    for (int i = 0; i < this->d_size; i++) {                                                       \
      for (int j = 0; j < this->x_size; j++) {                                                     \
        ASSERT_NEAR((float)weights[i][j], (float)cuweights[i][j], TOLERANCE);                      \
        count_diff += weights[i][j] != refweights[i][j];                                           \
      }                                                                                            \
    }                                                                                              \
    ASSERT_TRUE(count_diff > 0);                                                                   \
    if (loop % 3 == 0) {                                                                           \
      this->culayer_pulsed->setWeights(refweights[0]);                                             \
      this->context->synchronizeDevice();                                                          \
      this->layer_pulsed->setWeights(refweights[0]);                                               \
    }                                                                                              \
  }                                                                                                \
  std::cout << BOLD_ON << "\nCUDA Updates done in: " << (float)cudur / 1000. / nloop << " msec. "  \
            << BOLD_OFF << std::endl;                                                              \
  std::cout << BOLD_ON << "RPU Updates done in: " << (float)dur / 1000. / nloop << " msec.\n "     \
            << BOLD_OFF << std::endl;                                                              \
                                                                                                   \
  Array_2D_Free(refweights);

TEST_P(RPUCudaPowStepTestFixture, UpdateVector) {

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  this->context->synchronizeDevice();

  RPU_TEST_UPDATE(
      update(this->x_cuvec->getData(), this->d_cuvec->getData(), false, 1),
      update(this->x_vec.data(), this->d_vec.data(), false, 1), this->repeats);
}

TEST_P(RPUCudaPowStepTestFixture, UpdateMatrixBatch) {

  this->x_cuvec_batch->assign(this->rx.data());
  this->x_vec_batch = this->rx;

  this->d_cuvec_batch->assign(this->rd.data());
  this->d_vec_batch = this->rd;

  RPU_TEST_UPDATE(
      update(this->x_cuvec_batch->getData(), this->d_cuvec_batch->getData(), false, this->m_batch),
      update(this->x_vec_batch.data(), this->d_vec_batch.data(), false, this->m_batch), 10);
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
