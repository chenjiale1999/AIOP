#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, __gm__ float* X, __gm__ float* Y) {
  float x[64];
  float rsq[2];
  float sq[2];
  float var[2];
  float inv[2];
  #pragma unroll
  for (int32_t i = 0; i < 64; ++i) {
    x[i] = X[(((((int32_t)block_idx) * 8192) + (i * 128)) + ((int32_t)threadIdx.x))];
  }
  float broadcast_var = float(0x0p+0f/*0.000000e+00*/);
  *(float2*)(rsq + 0) = make_float2(broadcast_var, broadcast_var);
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 64; ++i_1) {
    rsq[(i_1 >> 5)] = (rsq[(i_1 >> 5)] + (x[i_1] * x[i_1]));
  }
  for (int32_t __finred_0 = 0; __finred_0 < 2; ++__finred_0) {
    asc_syncthreads();
    rsq[__finred_0] = tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(rsq[__finred_0], (&(((__ubuf__ float*)buf_dyn_shmem)[4096])));
  }
  *(float2*)(sq + 0) = *(float2*)(rsq + 0);
  *(float2*)(var + 0) = *(float2*)(sq + 0);
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 2; ++i_2) {
    inv[i_2] = rsqrtf(((sq[i_2] / float(0x1p+12f/*4.096000e+03*/)) + float(0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/)));
  }
  #pragma unroll
  for (int32_t i_3 = 0; i_3 < 64; ++i_3) {
    Y[(((((int32_t)block_idx) * 8192) + (i_3 * 128)) + ((int32_t)threadIdx.x))] = ((x[i_3] * inv[(i_3 >> 5)]) * ((__ubuf__ float*)buf_dyn_shmem)[(((i_3 & 31) * 128) + ((int32_t)threadIdx.x))]);
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* W, __gm__ float* X, __gm__ float* Y) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(W[0]))), 1, 16384, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16384, 16384);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(128), buf_dyn_shmem, X, Y);
}

