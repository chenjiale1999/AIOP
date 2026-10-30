#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/gemm.h>
extern "C" __global__ __cube__ void main_kernel(__gm__ bfloat16_t* A, __gm__ bfloat16_t* B, __gm__ float* C) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ float *C_l0c = (__cc__ float *)0;
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
  for (int32_t kk = 0; kk < 2; ++kk) {
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kk));
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(224), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((kk * 28672) + 65536)), (&(B[((((int32_t)block_idx) * 57344) + (kk * 128))])), 512, static_cast<asc_load_l2_cache_mode>(4), 224, 128, 0, 0);
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(256), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + (kk * 32768)), (&(A[(kk * 128)])), 512, static_cast<asc_load_l2_cache_mode>(0), 256, 128, 0, 0);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kk));
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kk));
    ascend_gemm_l1<256, 128, 224, 64, true, bfloat16_t>(((__cc__ float*)C_l0c), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + (kk * 32768)), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((kk * 28672) + 65536)), (kk == 0), 0, 0);
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kk));
    if (kk == 1) {
      asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
    }
  }
  asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  asc_set_l0c_copy_nz_para(1, 0, 0);
  asc_copy_l0c2gm((&(C[(((int32_t)block_idx) * 224)])), ((__cc__ float*)C_l0c), 224, 256, 448, 256, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
}

