#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;

void flowWarp(double* img, double* flow1, double* out, const int height,
              const int width, const int channels, const int filling_value,
              const int interpolateMode);

void BilinearInterpolate(const double* img, int width, int height, int channels,
                         double x, double y, double* out);

void NNInterpolate(const double* img, int width, int height, int channels,
                   double x, double y, double* out);

template <typename T>
inline T __min(T a, T b) {
  return a > b ? b : a;
}

template <typename T>
inline T __max(T a, T b) {
  return (a < b) ? b : a;
}

template <typename T>
inline T EnforceRange(const T x, const int MaxValue) {
  return __min(__max(x, 0), MaxValue);
}
