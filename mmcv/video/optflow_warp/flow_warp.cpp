#include "flow_warp.hpp"

void FlowWarp(double* img, double* flow, double* out, const int height,
              const int width, const int channels, const int filling_value = 0,
              const int interpolateMode = 0) {
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int offset_cur = h * width + w;
      int offset_img = offset_cur * channels;
      int offset_flow = offset_cur * 2;
      double x, y;
      x = h + flow[offset_flow + 1];
      y = w + flow[offset_flow];

      if (x < 0 || x >= height - 1 || y < 0 || y >= width - 1) {
        for (int k = 0; k < channels; k++) {
          out[offset_img + k] = filling_value;
        }
        continue;
      }

      if (interpolateMode == 0)
        BilinearInterpolate(img, width, height, channels, x, y,
                            out + offset_img);
      else if (interpolateMode == 1)
        NNInterpolate(img, width, height, channels, x, y, out + offset_img);
      else
        throw "Not Implemented Interpolation Method";
    }
  }
}

void BilinearInterpolate(const double* img, int width, int height, int channels,
                         double x, double y, double* out) {
  int xx, yy, m, n, u, v, offset, offset_img, l;
  xx = x;
  yy = y;

  double dx, dy, s;

  dx = __max__(__min__(x - xx, double(1)), double(0));
  dy = __max__(__min__(y - yy, double(1)), double(0));

  for (m = 0; m <= 1; m++)
    for (n = 0; n <= 1; n++) {
      u = EnforceRange(yy + n, width);
      v = EnforceRange(xx + m, height);
      offset = v * width + u;
      offset_img = offset * channels;
      s = fabs(1 - m - dx) * fabs(1 - n - dy);
      for (l = 0; l < channels; l++) out[l] += img[offset_img + l] * s;
    }
}

void NNInterpolate(const double* img, int width, int height, int channels,
                   double x, double y, double* out) {
  int xx, yy, m, n, u, v, offset, offset_img, l;
  xx = x;
  yy = y;

  double dx, dy;

  dx = __max__(__min__(x - xx, double(1)), double(0));
  dy = __max__(__min__(y - yy, double(1)), double(0));

  m = (dx < 0.5) ? 0 : 1;
  n = (dy < 0.5) ? 0 : 1;

  u = EnforceRange(yy + n, width);
  v = EnforceRange(xx + m, height);
  offset = v * width + u;
  offset_img = offset * channels;

  for (l = 0; l < channels; l++) out[l] = img[offset_img + l];
}
