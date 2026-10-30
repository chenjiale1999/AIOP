#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, __gm__ float* X, __gm__ float* Y) {
  float x[16];
  float rmax[2];
  float mx[2];
  float rsum[2];
  float sm[2];
  #pragma unroll
  for (int32_t i = 0; i < 16; ++i) {
    float condval;
    if (((((i & 7) * 16) + (((int32_t)threadIdx.x) >> 3)) < 125)) {
      condval = X[((((((int32_t)block_idx) * 2000) + ((i >> 3) * 1000)) + ((i & 7) * 128)) + ((int32_t)threadIdx.x))];
    } else {
      condval = float(-0x1.c363cbf21f28ap+127f/*-3.000000e+38*/);
    }
    x[i] = condval;
  }
  float broadcast_var = tl::limits::kFloatNInf;
  *(float2*)(rmax + 0) = make_float2(broadcast_var, broadcast_var);
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
    rmax[(i_1 >> 3)] = max(rmax[(i_1 >> 3)], x[i_1]);
  }
  for (int32_t __finred_0 = 0; __finred_0 < 2; ++__finred_0) {
    asc_syncthreads();
    rmax[__finred_0] = tl::AscendAllReduce<tl::MaxOp, 128, 1, 0>::run(rmax[__finred_0], (&(((__ubuf__ float*)buf_dyn_shmem)[0])));
  }
  *(float2*)(mx + 0) = *(float2*)(rmax + 0);
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 16; ++i_2) {
    float condval_1;
    if (((((i_2 & 7) * 16) + (((int32_t)threadIdx.x) >> 3)) < 125)) {
      condval_1 = expf((x[i_2] - mx[(i_2 >> 3)]));
    } else {
      condval_1 = float(0x0p+0f/*0.000000e+00*/);
    }
    x[i_2] = condval_1;
  }
  float broadcast_var_1 = float(0x0p+0f/*0.000000e+00*/);
  *(float2*)(rsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  #pragma unroll
  for (int32_t i_3 = 0; i_3 < 16; ++i_3) {
    rsum[(i_3 >> 3)] = (rsum[(i_3 >> 3)] + x[i_3]);
  }
  for (int32_t __finred_0_1 = 0; __finred_0_1 < 2; ++__finred_0_1) {
    asc_syncthreads();
    rsum[__finred_0_1] = tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(rsum[__finred_0_1], (&(((__ubuf__ float*)buf_dyn_shmem)[128])));
  }
  *(float2*)(sm + 0) = *(float2*)(rsum + 0);
  #pragma unroll
  for (int32_t i_4 = 0; i_4 < 16; ++i_4) {
    if ((((i_4 & 7) * 16) + (((int32_t)threadIdx.x) >> 3)) < 125) {
      Y[((((((int32_t)block_idx) * 2000) + ((i_4 >> 3) * 1000)) + ((i_4 & 7) * 128)) + ((int32_t)threadIdx.x))] = (x[i_4] / sm[(i_4 >> 3)]);
    }
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* X, __gm__ float* Y) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(128), buf_dyn_shmem, X, Y);
}

