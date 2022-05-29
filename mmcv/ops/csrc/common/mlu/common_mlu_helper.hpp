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

#define CEIL_ALIGN(x, y) (((x) + (y) - 1) / (y) * (y))

#endif  // UTILS_H_
