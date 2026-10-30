#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/gemm.h>
__simt_vf__ __launch_bounds__(128) inline void main_kernel_simt_vf_0(__ubuf__ float* C_ub) {
  #pragma unroll
  for (int32_t i = 0; i < 112; ++i) {
    float broadcast_var = float(0x0p+0f/*0.000000e+00*/);
    float2 _;
      float2 v_ = *(__ubuf__ float2*)(C_ub + ((i * 256) + (((int32_t)threadIdx.x) * 2)));
      float2 v__1 = make_float2(broadcast_var, broadcast_var);
      _.x = max(v_.x, v__1.x);
      _.y = max(v_.y, v__1.y);
    *(__ubuf__ float2*)(C_ub + ((i * 256) + (((int32_t)threadIdx.x) * 2))) = _;
  }
}

extern "C" __global__ __mix__(1, 2) void main_kernel(__gm__ bfloat16_t* A, __gm__ bfloat16_t* B, __gm__ float* C) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ float *C_l0c = (__cc__ float *)0;
  __ubuf__ float *C_ub = (__ubuf__ float *)0;
  if ASC_IS_AIC {
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(224), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 32768), (&(B[0])), 256, static_cast<asc_load_l2_cache_mode>(4), 224, 128, 0, 0);
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(256), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1), (&(A[0])), 256, static_cast<asc_load_l2_cache_mode>(0), 256, 128, 0, 0);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
    ascend_gemm_l1<256, 128, 224, 64, true, bfloat16_t>(((__cc__ float*)C_l0c), ((__cbuf__ bfloat16_t*)buf_dyn_l1), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 32768), (bool)1, 0, 0);
    asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
    asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2ub((&(C_ub[0])), ((__cc__ float*)C_l0c), 224, 256, 224, 256, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
    asc_sync_intra_arrive(PIPE_FIX, 0);
    asc_sync_intra_arrive(PIPE_FIX, 16);
  }
  if ASC_IS_AIV {
    int32_t sid = asc_get_sub_block_id();
    asc_sync_intra_wait(PIPE_V, 0);
    asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(128), C_ub);
    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    asc_copy_ub2gm_align((__gm__ uint8_t*)((&(C[(sid * 28672)]))), (__ubuf__ uint8_t*)((&(C_ub[0]))), 1, 114688, static_cast<asc_store_l2_cache_mode>(5), 114688, 114688);
  }
}

