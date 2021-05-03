// It is modified from https://github.com/whai362/PSENet
#include <iostream>
#include <queue>

#include "pytorch_cpp_helper.hpp"

using namespace std;

class Point2d {
 public:
  int x;
  int y;

  Point2d() : x(0), y(0) {}
  Point2d(int _x, int _y) : x(_x), y(_y) {}
};

void growing_text_line(const uint8_t *data, IntArrayRef data_shape,
                       const int *label_map,

                       IntArrayRef label_shape, int &label_num, int &min_area,
                       vector<vector<int>> &text_line) {
  std::vector<int> area(label_num + 1);
  for (int x = 0; x < label_shape[0]; ++x) {
    for (int y = 0; y < label_shape[1]; ++y) {
      int label = label_map[x * label_shape[1] + y];
      if (label == 0) continue;
      area[label] += 1;
    }
  }

  queue<Point2d> queue, next_queue;
  for (int x = 0; x < label_shape[0]; ++x) {
    vector<int> row(label_shape[1]);
    for (int y = 0; y < label_shape[1]; ++y) {
      int label = label_map[x * label_shape[1] + y];
      if (label == 0) continue;
      if (area[label] < min_area) continue;

      Point2d point(x, y);
      queue.push(point);
      row[y] = label;
    }
    text_line.emplace_back(row);
  }

  int dx[] = {-1, 1, 0, 0};
  int dy[] = {0, 0, -1, 1};

  for (int kernel_id = data_shape[0] - 2; kernel_id >= 0; --kernel_id) {
    while (!queue.empty()) {
      Point2d point = queue.front();
      queue.pop();
      int x = point.x;
      int y = point.y;
      int label = text_line[x][y];

      bool is_edge = true;
      for (int d = 0; d < 4; ++d) {
        int tmp_x = x + dx[d];
        int tmp_y = y + dy[d];

        if (tmp_x < 0 || tmp_x >= (int)text_line.size()) continue;
        if (tmp_y < 0 || tmp_y >= (int)text_line[1].size()) continue;
        int kernel_value = data[kernel_id * data_shape[1] * data_shape[2] +
                                tmp_x * data_shape[2] + tmp_y];
        if (kernel_value == 0) continue;
        if (text_line[tmp_x][tmp_y] > 0) continue;

        Point2d point(tmp_x, tmp_y);
        queue.push(point);
        text_line[tmp_x][tmp_y] = label;
        is_edge = false;
      }

      if (is_edge) {
        next_queue.push(point);
      }
    }
    swap(queue, next_queue);
  }
}

std::vector<std::vector<int>> contour_expand(Tensor kernel_mask,
                                             Tensor internal_kernel_label,
                                             int min_kernel_area,
                                             int kernel_num) {
  kernel_mask = kernel_mask.contiguous();
  internal_kernel_label = internal_kernel_label.contiguous();
  assert(kernel_mask.dim() == 3);
  assert(internal_kernel_label.dim() == 2);
  assert(kernel_mask.size(1) == internal_kernel_label.size(0));
  assert(kernel_mask.size(2) == internal_kernel_label.size(1));
  CHECK_CPU_INPUT(kernel_mask);
  CHECK_CPU_INPUT(internal_kernel_label);
  auto ptr_data = kernel_mask.data_ptr<uint8_t>();
  IntArrayRef data_shape = kernel_mask.sizes();

  auto data_label_map = internal_kernel_label.data_ptr<int32_t>();
  IntArrayRef label_map_shape = internal_kernel_label.sizes();
  vector<vector<int>> text_line;

  growing_text_line(ptr_data, data_shape, data_label_map, label_map_shape,
                    kernel_num, min_kernel_area, text_line);

  return text_line;
}
