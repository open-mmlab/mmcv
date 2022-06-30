/*************************************************************************
 * Copyright (C) 2021 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef UTILS_H_
#define UTILS_H_

#define NFU_ALIGN_SIZE 128          // Byte
#define REM_FOR_STACK (128 * 1024)  // 128KB reserved for cncc

#ifdef __BANG_ARCH__
#define MAX_NRAM_SIZE \
  (__MLU_NRAM_SIZE__ * 1024 - REM_FOR_STACK)  // 128KB reserved for cncc
#define MAX_SRAM_SIZE \
  (__MLU_SRAM_SIZE__ * 1024 - REM_FOR_STACK)  // 128KB reserved for cncc
#else
#define MAX_NRAM_SIZE (384 * 1024)   // 384KB,  initialization value
#define MAX_SRAM_SIZE (1920 * 1024)  // 1920KB, initialization value
#endif

#ifndef PAD_UP
#define PAD_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))
#endif

#ifndef PAD_DOWN
#define PAD_DOWN(x, y) (((x) / (y)) * (y))
#endif

#define CEIL_ALIGN(x, y) (((x) + (y)-1) / (y) * (y))

template <typename T>
__mlu_func__ void loadStr2D(T *dst, T *src, const int size, const int dst_str,
                            const int src_str, const int seg_num) {
  if (dst_str == src_str && size == src_str) {
    __memcpy(dst, src, src_str * seg_num * sizeof(T), GDRAM2NRAM);
  } else if ((size == src_str || src_str <= dst_str) &&
             src_str * sizeof(T) <= 512) {
    // IO efficiency is best when datasize gather than 512bytes
    T *tmp = (T *)dst + (dst_str - src_str) * seg_num;
    __memcpy(tmp, src, (src_str * (seg_num - 1) + size) * sizeof(T),
             GDRAM2NRAM);
    if (dst_str != src_str) {
      __memcpy(dst, tmp, size * sizeof(T), NRAM2NRAM, dst_str * sizeof(T),
               src_str * sizeof(T), seg_num - 1);
    }
  } else {
    __memcpy(dst, src, size * sizeof(T), GDRAM2NRAM, dst_str * sizeof(T),
             src_str * sizeof(T), seg_num - 1);
  }
}

template <typename T>
__mlu_func__ void loadStr3D(T *dst, T *src, const int size,
                            const int seg_num_in, const int seg_num_out,
                            const int dst_str_in, const int dst_str_out,
                            const int src_str_in, const int src_str_out) {
  T *tmp_dst = dst;
  T *tmp_src = src;

  for (int i = 0; i < seg_num_out; ++i) {
    loadStr2D(tmp_dst, tmp_src, size, dst_str_in, src_str_in, seg_num_in);
    tmp_src += src_str_out;
    tmp_dst += dst_str_out;
  }
}

template <typename T>
__mlu_func__ void storeStr2D(T *dst, T *src, const int size, const int seg_num,
                             const int dst_str, const int src_str) {
  if ((size == dst_str && dst_str <= src_str) && dst_str * sizeof(T) <= 512) {
    // IO efficiency is best when datasize gather than 512bytes
    if (dst_str != src_str) {
      __memcpy(src, src, size * sizeof(T), NRAM2NRAM, dst_str * sizeof(T),
               src_str * sizeof(T), seg_num - 1);
    }
    __memcpy(dst, src, size * seg_num * sizeof(T), NRAM2GDRAM);
  } else {
    __memcpy(dst, src, size * sizeof(T), NRAM2GDRAM, dst_str * sizeof(T),
             src_str * sizeof(T), seg_num - 1);
  }
}

template <typename T>
__mlu_func__ void storeStr3D(T *dst, T *src, const int size,
                             const int seg_num_in, const int seg_num_out,
                             const int dst_str_in, const int dst_str_out,
                             const int src_str_in, const int src_str_out) {
  T *tmp_dst = dst;
  T *tmp_src = src;
  for (int i = 0; i < seg_num_out; ++i) {
    storeStr2D(tmp_dst, tmp_src, size, seg_num_in, dst_str_in, src_str_in);
    tmp_src += src_str_out;
    tmp_dst += dst_str_out;
  }
}

#endif  // UTILS_H_
