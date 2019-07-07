#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;

void FlowWarp(double* img, double* flow1, double* out, const int height,
              const int width, const int channels, const int filling_value,
              const int interpolateMode);

void BilinearInterpolate(const double* img, int width, int height, int channels,
                         double x, double y, double* out);

void NNInterpolate(const double* img, int width, int height, int channels,
                   double x, double y, double* out);

template <typename T>
inline T __min__(T a, T b) {
  return a > b ? b : a;
}

template <typename T>
inline T __max__(T a, T b) {
  return (a < b) ? b : a;
}

template <typename T>
inline T EnforceRange(const T x, const int MaxValue) {
  return __min__(__max__(x, 0), MaxValue);
}
