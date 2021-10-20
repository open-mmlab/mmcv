// Copyright (c) 2021, SenseTime.

#ifndef MLU_UTILS_H_
#define MLU_UTILS_H_

#define NFU_ALIGN_SIZE 128          // Byte
#define REM_FOR_STACK (128 * 1024)  // 128KB reserved for cncc

#ifdef __BANG_ARCH__
#define MAX_NRAM_SIZE                                                          \
    (__MLU_NRAM_SIZE__ * 1024 - REM_FOR_STACK)  // 128KB reserved for cncc
#else
#define MAX_NRAM_SIZE (384 * 1024)  // 384KB, initialization value
#endif

#define PAD_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))  // . NOLINT

#define PAD_DOWN(x, y) (((x) / (y)) * (y))

#endif  // MLU_UTILS_H_
