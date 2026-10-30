#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, __gm__ float* X, __gm__ float* Y) {
  float x[16];
  float rsum[2];
  float rsq[2];
  float sq[2];
  float mu[2];
  float var[2];
  float inv[2];
  #pragma unroll
  for (int32_t i = 0; i < 16; ++i) {
    float condval;
    if (((((i & 7) * 16) + (((int32_t)threadIdx.x) >> 3)) < 125)) {
      condval = X[((((((int32_t)block_idx) * 2000) + ((i >> 3) * 1000)) + ((i & 7) * 128)) + ((int32_t)threadIdx.x))];
    } else {
      condval = float(0x0p+0f/*0.000000e+00*/);
    }
    x[i] = condval;
  }
  float broadcast_var = float(0x0p+0f/*0.000000e+00*/);
  *(float2*)(rsum + 0) = make_float2(broadcast_var, broadcast_var);
  float broadcast_var_1 = float(0x0p+0f/*0.000000e+00*/);
  *(float2*)(rsq + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
    rsum[(i_1 >> 3)] = (rsum[(i_1 >> 3)] + x[i_1]);
    rsq[(i_1 >> 3)] = (rsq[(i_1 >> 3)] + (x[i_1] * x[i_1]));
  }
  for (int32_t __finred_0 = 0; __finred_0 < 2; ++__finred_0) {
    asc_syncthreads();
    rsq[__finred_0] = tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(rsq[__finred_0], (&(((__ubuf__ float*)buf_dyn_shmem)[2000])));
  }
  *(float2*)(sq + 0) = *(float2*)(rsq + 0);
  for (int32_t __finred_0_1 = 0; __finred_0_1 < 2; ++__finred_0_1) {
    asc_syncthreads();
    rsum[__finred_0_1] = tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(rsum[__finred_0_1], (&(((__ubuf__ float*)buf_dyn_shmem)[2128])));
  }
  *(float2*)(mu + 0) = *(float2*)(rsum + 0);
  float broadcast_var_2 = float(0x1.f4p+9f/*1.000000e+03*/);
  float2 _;
    float2 v_ = *(float2*)(mu + 0);
    float2 v__1 = make_float2(broadcast_var_2, broadcast_var_2);
    _.x = (v_.x/v__1.x);
    _.y = (v_.y/v__1.y);
  *(float2*)(mu + 0) = _;
  float broadcast_var_3 = float(0x1.f4p+9f/*1.000000e+03*/);
  float2 __1;
    float2 v__2 = *(float2*)(sq + 0);
    float2 v__3 = make_float2(broadcast_var_3, broadcast_var_3);
    __1.x = (v__2.x/v__3.x);
    __1.y = (v__2.y/v__3.y);
  *(float2*)(var + 0) = (__1 - (*(float2*)(mu + 0) * *(float2*)(mu + 0)));
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 2; ++i_2) {
    inv[i_2] = rsqrtf((max(var[i_2], float(0x0p+0f/*0.000000e+00*/)) + float(0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/)));
  }
  #pragma unroll
  for (int32_t i_3 = 0; i_3 < 16; ++i_3) {
    if ((((i_3 & 7) * 16) + (((int32_t)threadIdx.x) >> 3)) < 125) {
      Y[((((((int32_t)block_idx) * 2000) + ((i_3 >> 3) * 1000)) + ((i_3 & 7) * 128)) + ((int32_t)threadIdx.x))] = ((((x[i_3] - mu[(i_3 >> 3)]) * inv[(i_3 >> 3)]) * ((__ubuf__ float*)buf_dyn_shmem)[(((i_3 & 7) * 128) + ((int32_t)threadIdx.x))]) + ((__ubuf__ float*)buf_dyn_shmem)[((((i_3 & 7) * 128) + ((int32_t)threadIdx.x)) + 1000)]);
    }
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* B, __gm__ float* W, __gm__ float* X, __gm__ float* Y) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(W[0]))), 1, 4000, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4000, 4000);
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[1000]))), (__gm__ uint8_t*)((&(B[0]))), 1, 4000, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4000, 4000);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(128), buf_dyn_shmem, X, Y);
}

