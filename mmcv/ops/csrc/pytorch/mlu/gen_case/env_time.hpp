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
#ifndef ENV_TIME_HPP_
#define ENV_TIME_HPP_

#if !defined(PLATFORM_POSIX)
#if defined(_WIN32)
#define PLATFORM_WINDOWS
#else
#define PLATFORM_POSIX
#endif
#endif  // !defined(PLATFORM_POSIX)

#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace cnnl {
namespace platform {

// An interface used by the implementation to access timer related operations.
class EnvTime {
 public:
  static constexpr uint64_t kMicrosToPicos   = 1000ULL * 1000ULL;
  static constexpr uint64_t kMicrosToNanos   = 1000ULL;
  static constexpr uint64_t kMillisToMicros  = 1000ULL;
  static constexpr uint64_t kMillisToNanos   = 1000ULL * 1000ULL;
  static constexpr uint64_t kSecondsToMillis = 1000ULL;
  static constexpr uint64_t kSecondsToMicros = 1000ULL * 1000ULL;
  static constexpr uint64_t kSecondsToNanos  = 1000ULL * 1000ULL * 1000ULL;

  EnvTime() {}
  virtual ~EnvTime() = default;

  // Returns a default impl suitable for the current operating system.
  // The result of Default() belongs to this library and must never be deleted.
  static EnvTime *Default();

  // Returns the number of nano-seconds since the Unix epoch.
  virtual uint64_t NowNanos() const = 0;

  // Returns the number of micro-seconds since the Unix epoch.
  virtual uint64_t NowMicros() const { return NowNanos() / kMicrosToNanos; }

  // Returns the number of seconds since the Unix epoch.
  virtual uint64_t NowSeconds() const { return NowNanos() / kSecondsToNanos; }
};

class PosixEnvTime : public EnvTime {
 public:
  PosixEnvTime() {}

  uint64_t NowNanos() const override {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64_t>(ts.tv_nsec));
  }
};

#if defined(PLATFORM_POSIX)
EnvTime *EnvTime::Default() {
  static EnvTime *default_env_time = new (std::nothrow) PosixEnvTime;
  return default_env_time;
}
#endif  // defined(PLATFORM_POSIX)

}  // namespace platform
}  // namespace cnnl

#endif  // ENV_TIME_HPP_
