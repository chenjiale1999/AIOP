#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t step) {
  #pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((step & 1) * 4096) + (i * 512)) + (((int32_t)threadIdx.x) * 2)) + 16384)) = (*(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + ((((step & 1) * 4096) + (i * 512)) + (((int32_t)threadIdx.x) * 2))) + *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((step & 1) * 4096) + (i * 512)) + (((int32_t)threadIdx.x) * 2)) + 8192)));
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* A, __gm__ float* B, __gm__ float* C) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  for (int32_t step = 0; step < 4; ++step) {
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((step & 1)));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[((step & 1) * 4096)]))), (__gm__ uint8_t*)((&(A[((step * 262144) + (((int32_t)block_idx) * 4096))]))), 1, 16384, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16384, 16384);
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((step & 1) * 4096) + 8192)]))), (__gm__ uint8_t*)((&(B[((step * 262144) + (((int32_t)block_idx) * 4096))]))), 1, 16384, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16384, 16384);
    asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((step & 1)));
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((step & 1)));
    asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((step & 1)));
    asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(256), buf_dyn_shmem, step);
    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((step & 1)));
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((step & 1)));
    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>((step & 1)));
    asc_copy_ub2gm_align((__gm__ uint8_t*)((&(C[((step * 262144) + (((int32_t)block_idx) * 4096))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((step & 1) * 4096) + 16384)]))), 1, 16384, static_cast<asc_store_l2_cache_mode>(4), 16384, 16384);
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((step & 1)));
  }
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
}

