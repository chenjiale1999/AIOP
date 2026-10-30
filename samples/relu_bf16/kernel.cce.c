#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t step) {
  #pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    bfloat16_t c_local_cast[4];
    bfloat16_t a_local_cast_1[4];
    *(uint2*)(a_local_cast_1 + 0) = *(__ubuf__ uint2*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + ((((step & 1) * 8192) + (i * 1024)) + (((int32_t)threadIdx.x) * 4)));
    for (int32_t vec = 0; vec < 2; ++vec) {
      float broadcast_var = float(0x0p+0f/*0.000000e+00*/);
      uint1 _;
        float2 __1;
          float2 __2;
            uint1 v_ = *(uint1*)(a_local_cast_1 + (vec * 2));
            ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<bfloat16x2_t*>(&v_))[0]);
          float2 v__1 = make_float2(broadcast_var, broadcast_var);
          __1.x = max(__2.x, v__1.x);
          __1.y = max(__2.y, v__1.y);
        (reinterpret_cast<bfloat16x2_t*>(&_))[0] = __float22bfloat162_rn(((float2*)(&__1))[0]);
      *(uint1*)(c_local_cast + (vec * 2)) = _;
    }
    *(__ubuf__ uint2*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + (((((step & 1) * 8192) + (i * 1024)) + (((int32_t)threadIdx.x) * 4)) + 16384)) = *(uint2*)(c_local_cast + 0);
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ bfloat16_t* X, __gm__ bfloat16_t* Y) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  for (int32_t step = 0; step < 8; ++step) {
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((step & 1)));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((step & 1) * 8192)]))), (__gm__ uint8_t*)((&(X[((step * 524288) + (((int32_t)block_idx) * 8192))]))), 1, 16384, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16384, 16384);
    asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((step & 1)));
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((step & 1)));
    asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((step & 1)));
    asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(256), buf_dyn_shmem, step);
    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((step & 1)));
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((step & 1)));
    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>((step & 1)));
    asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Y[((step * 524288) + (((int32_t)block_idx) * 8192))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((step & 1) * 8192) + 16384)]))), 1, 16384, static_cast<asc_store_l2_cache_mode>(4), 16384, 16384);
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((step & 1)));
  }
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
}

