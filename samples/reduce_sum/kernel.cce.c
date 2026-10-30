#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void main_kernel_simt_vf_0(__gm__ float* X, __gm__ float* Y) {
  float x[16];
  float acc[2];
  float res[2];
  __ubuf__ float *workspace = (__ubuf__ float *)0;
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
  *(float2*)(acc + 0) = make_float2(broadcast_var, broadcast_var);
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
    acc[(i_1 >> 3)] = (acc[(i_1 >> 3)] + x[i_1]);
  }
  for (int32_t __finred_0 = 0; __finred_0 < 2; ++__finred_0) {
    asc_syncthreads();
    acc[__finred_0] = tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(acc[__finred_0], (&(workspace[0])));
  }
  *(float2*)(res + 0) = *(float2*)(acc + 0);
  if (((int32_t)threadIdx.x) == 0) {
    *(__gm__ float2*)(Y + (((int32_t)block_idx) * 2)) = *(float2*)(res + 0);
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* X, __gm__ float* Y) {
  asc_init();
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(128), X, Y);
}

