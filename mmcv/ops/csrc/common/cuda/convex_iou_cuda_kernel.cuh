// Copyright (c) OpenMMLab. All rights reserved
#ifndef CONVEX_IOU_CUDA_KERNEL_CUH
#define CONVEX_IOU_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#define maxn 100
const double eps = 1E-8;
int const threadsPerBlock = 512;

__device__ inline int sig(double d) { return int(d > eps) - int(d < -eps); }

struct Point {
  double x, y;
  __device__ Point() {}
  __device__ Point(double x, double y) : x(x), y(y) {}
};

__device__ inline bool point_same(Point& a, Point& b) {
  return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

__device__ inline void swap1(Point* a, Point* b) {
  Point temp;
  temp.x = a->x;
  temp.y = a->y;

  a->x = b->x;
  a->y = b->y;

  b->x = temp.x;
  b->y = temp.y;
}

__device__ inline void reverse1(Point* a, const int n) {
  Point temp[maxn];
  for (int i = 0; i < n; i++) {
    temp[i].x = a[i].x;
    temp[i].y = a[i].y;
  }
  for (int i = 0; i < n; i++) {
    a[i].x = temp[n - 1 - i].x;
    a[i].y = temp[n - 1 - i].y;
  }
}

__device__ inline double cross(Point o, Point a, Point b) {
  return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

__device__ inline double dis(Point a, Point b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
__device__ inline double area(Point* ps, int n) {
  ps[n] = ps[0];
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
  }
  return res / 2.0;
}

__device__ inline int lineCross(Point a, Point b, Point c, Point d, Point& p) {
  double s1, s2;
  s1 = cross(a, b, c);
  s2 = cross(a, b, d);
  if (sig(s1) == 0 && sig(s2) == 0) return 2;
  if (sig(s2 - s1) == 0) return 0;
  p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
  p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
  return 1;
}

__device__ inline void polygon_cut(Point* p, int& n, Point a, Point b) {
  Point pp[maxn];
  int m = 0;
  p[n] = p[0];
  for (int i = 0; i < n; i++) {
    if (sig(cross(a, b, p[i])) > 0) {
      pp[m] = p[i];
      m++;
    }
    if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1]))) {
      lineCross(a, b, p[i], p[i + 1], pp[m]);
      m++;
    }
  }
  n = 0;
  for (int i = 0; i < m; i++) {
    if (!i || !(point_same(pp[i], pp[i - 1]))) {
      p[n] = pp[i];
      n++;
    }
  }

  while (n > 1 && point_same(p[n - 1], p[0])) n--;
}

__device__ inline double intersectArea(Point a, Point b, Point c, Point d) {
  Point o(0, 0);
  int s1 = sig(cross(o, a, b));
  int s2 = sig(cross(o, c, d));
  if (s1 == 0 || s2 == 0) return 0.0;
  if (s1 == -1) {
    Point* i = &a;
    Point* j = &b;
    swap1(i, j);
  }
  if (s2 == -1) {
    Point* i = &c;
    Point* j = &d;
    swap1(i, j);
  }
  Point p[10] = {o, a, b};
  int n = 3;

  polygon_cut(p, n, o, c);
  polygon_cut(p, n, c, d);
  polygon_cut(p, n, d, o);
  double res = area(p, n);
  if (s1 * s2 == -1) res = -res;
  return res;
}
__device__ inline double intersectAreaO(Point* ps1, int n1, Point* ps2,
                                        int n2) {
  if (area(ps1, n1) < 0) reverse1(ps1, n1);
  if (area(ps2, n2) < 0) reverse1(ps2, n2);
  ps1[n1] = ps1[0];
  ps2[n2] = ps2[0];
  double res = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
    }
  }
  return res;
}

// convex_find and get the polygen_index_box_index
__device__ inline void Jarvis_and_index(Point* in_poly, int& n_poly,
                                        int* points_to_convex_ind) {
  int n_input = n_poly;
  Point input_poly[20];
  for (int i = 0; i < n_input; i++) {
    input_poly[i].x = in_poly[i].x;
    input_poly[i].y = in_poly[i].y;
  }
  Point p_max, p_k;
  int max_index, k_index;
  int Stack[20], top1, top2;
  double sign;
  Point right_point[10], left_point[10];

  for (int i = 0; i < n_poly; i++) {
    if (in_poly[i].y < in_poly[0].y ||
        in_poly[i].y == in_poly[0].y && in_poly[i].x < in_poly[0].x) {
      Point* j = &(in_poly[0]);
      Point* k = &(in_poly[i]);
      swap1(j, k);
    }
    if (i == 0) {
      p_max = in_poly[0];
      max_index = 0;
    }
    if (in_poly[i].y > p_max.y ||
        in_poly[i].y == p_max.y && in_poly[i].x > p_max.x) {
      p_max = in_poly[i];
      max_index = i;
    }
  }
  if (max_index == 0) {
    max_index = 1;
    p_max = in_poly[max_index];
  }

  k_index = 0, Stack[0] = 0, top1 = 0;
  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top1]], in_poly[i], p_k);
      if ((sign > 0) || ((sign == 0) && (dis(in_poly[Stack[top1]], in_poly[i]) >
                                         dis(in_poly[Stack[top1]], p_k)))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top1++;
    Stack[top1] = k_index;
  }
  for (int i = 0; i <= top1; i++) {
    right_point[i] = in_poly[Stack[i]];
  }
  k_index = 0, Stack[0] = 0, top2 = 0;

  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top2]], in_poly[i], p_k);
      if ((sign < 0) || (sign == 0) && (dis(in_poly[Stack[top2]], in_poly[i]) >
                                        dis(in_poly[Stack[top2]], p_k))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top2++;
    Stack[top2] = k_index;
  }

  for (int i = top2 - 1; i >= 0; i--) {
    left_point[i] = in_poly[Stack[i]];
  }

  for (int i = 0; i < top1 + top2; i++) {
    if (i <= top1) {
      in_poly[i] = right_point[i];
    } else {
      in_poly[i] = left_point[top2 - (i - top1)];
    }
  }
  n_poly = top1 + top2;
  for (int i = 0; i < n_poly; i++) {
    for (int j = 0; j < n_input; j++) {
      if (point_same(in_poly[i], input_poly[j])) {
        points_to_convex_ind[i] = j;
        break;
      }
    }
  }
}

__device__ inline float devrIoU(float const* const p, float const* const q) {
  Point ps1[maxn], ps2[maxn];
  Point convex[maxn];
  for (int i = 0; i < 9; i++) {
    convex[i].x = (double)p[i * 2];
    convex[i].y = (double)p[i * 2 + 1];
  }
  int n_convex = 9;
  int points_to_convex_ind[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  Jarvis_and_index(convex, n_convex, points_to_convex_ind);
  int n1 = n_convex;
  for (int i = 0; i < n1; i++) {
    ps1[i].x = (double)convex[i].x;
    ps1[i].y = (double)convex[i].y;
  }
  int n2 = 4;
  for (int i = 0; i < n2; i++) {
    ps2[i].x = (double)q[i * 2];
    ps2[i].y = (double)q[i * 2 + 1];
  }
  double inter_area = intersectAreaO(ps1, n1, ps2, n2);
  double S_pred = area(ps1, n1);
  double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;
  double iou = inter_area / union_area;
  return (float)iou;
}

__global__ void convex_iou_cuda_kernel(const int ex_n_boxes,
                                       const int gt_n_boxes,
                                       const float* ex_boxes,
                                       const float* gt_boxes, float* iou) {
  const int ex_start = blockIdx.x;
  const int ex_size =
      min(ex_n_boxes - ex_start * threadsPerBlock, threadsPerBlock);

  if (threadIdx.x < ex_size) {
    const int cur_box_idx = threadsPerBlock * ex_start + threadIdx.x;
    const float* cur_box = ex_boxes + cur_box_idx * 18;
    for (int i = 0; i < gt_n_boxes; i++) {
      iou[cur_box_idx * gt_n_boxes + i] = devrIoU(cur_box, gt_boxes + i * 8);
    }
  }
}
#endif  // CONVEX_IOU_CUDA_KERNEL_CUH
