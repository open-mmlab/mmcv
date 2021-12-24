# Rotated ops

- [x] box_iou_rotated (Support `CW`.)
- [x] nms_rotated (Support `CW`.)
- [x] RoIAlignRotated (Support `CW` & `CCW`. Defaults to `CCW`)
- [x] RiRoIAlignRotated (Support `CW` & `CCW`. Defaults to `CCW`)


## Definition of coordinate system
Assume we have a horizontal box B `(x_center, y_center, width, height)`,
where width is along the x-axis and height is along the y-axis.
The rotated box B_rot `(x_center, y_center, width, height, angle)` can be seen
as:

1. When `angle == 0`: B_rot == B
2. When `angle > 0`: B_rot is obtained by rotating B w.r.t its center
by `|angle|` degrees CCW;
3. When `angle < 0`: B_rot is obtained by rotating B w.r.t its center
by `|angle|` degrees CW.


Mathematically, since the right-handed coordinate system for image space
is (y, x), where y is top->down and x is left->right, the 4 vertices of the
rotated rectangle `(yr_i, xr_i)` (i = 1, 2, 3, 4) can be obtained from
the vertices of the horizontal rectangle `(y_i, x_i)` (i = 1, 2, 3, 4)
in the following way (`theta = angle * np.pi / 180` is the angle in radians,
`(y_c, x_c)` is the center of the rectangle):
```
        yr_i = cos(theta) * (y_i - y_c) - sin(theta) * (x_i - x_c) + y_c,
        xr_i = sin(theta) * (y_i - y_c) + cos(theta) * (x_i - x_c) + x_c,
```
which is the standard rigid-body rotation transformation. Intuitively, the
angle is (1) the rotation angle from y-axis in image space
to the height vector (top->down in the box's local coordinate system)
of the box in CCW, and (2) the rotation angle from x-axis in image space
to the width vector (left->right in the box's local coordinate system)
of the box in CCW.

More intuitively, consider the following horizontal box ABCD represented
in `(x1, y1, x2, y2)`: (3, 2, 7, 4), covering the [3, 7] x [2, 4] region of the
continuous coordinate system which looks like this:
```
O--------> x
|
|  A---B
|  |   |
|  D---C
|
v y
```

Note that each capital letter represents one 0-dimensional geometric point
instead of a 'square pixel' here.
In the example above, using `(x, y)` to represent a point we have:
```
O = (0, 0), A = (3, 2), B = (7, 2), C = (7, 4), D = (3, 4)
```
We name `vector AB = vector DC` as the width vector in box's local coordinate system, and
`vector AD = vector BC` as the height vector in box's local coordinate system. Initially,
when `angle = 0` degree, they're aligned with the positive directions of x-axis and y-axis
in the image space, respectively.
For better illustration, we denote the center of the box as E,
```
O--------> x
|
|  A---B
|  | E |
|  D---C
|
v y
```
where the center `E = ((3+7)/2, (2+4)/2) = (5, 3)`. Also,`width = |AB| = |CD| = 7 - 3 = 4`,
`height = |AD| = |BC| = 4 - 2 = 2`.
Therefore, the corresponding representation for the same shape in rotated box in
`(x_center, y_center, width, height, angle)` format is: (5, 3, 4, 2, 0),
Now, let's consider (5, 3, 4, 2, 90), which is rotated by 90 degrees
CCW (counter-clockwise) by definition. It looks like this:
```
O--------> x
|   B-C
|   | |
|   |E|
|   | |
|   A-D
v y
```
The center E is still located at the same point (5, 3), while the vertices
ABCD are rotated by 90 degrees CCW with regard to E:
`A = (4, 5), B = (4, 1), C = (6, 1), D = (6, 5)`
Here, 90 degrees can be seen as the CCW angle to rotate from y-axis to
vector AD or vector BC (the top->down height vector in box's local coordinate system),
or the CCW angle to rotate from x-axis to vector AB or vector DC (the left->right
width vector in box's local coordinate system).`width = |AB| = |CD| = 5 - 1 = 4`,
`height = |AD| = |BC| = 6 - 4 = 2`.
Next, how about (5, 3, 4, 2, -90), which is rotated by 90 degrees CW (clockwise)
by definition? It looks like this:
```
O--------> x
|   D-A
|   | |
|   |E|
|   | |
|   C-B
v y
```
The center E is still located at the same point (5, 3), while the vertices
ABCD are rotated by 90 degrees CW with regard to E:
```
A = (6, 1), B = (6, 5), C = (4, 5), D = (4, 1)
width = |AB| = |CD| = 5 - 1 = 4,
height = |AD| = |BC| = 6 - 4 = 2.
```
This covers exactly the same region as (5, 3, 4, 2, 90) does, and their IoU
will be 1. However, these two will generate different RoI Pooling results and
should not be treated as an identical box.
On the other hand, it's easy to see that `(X, Y, W, H, A)` is identical to
`(X, Y, W, H, A+360N)`, for any integer N. For example (5, 3, 4, 2, 270) would be
identical to (5, 3, 4, 2, -90), because rotating the shape 270 degrees CCW is
equivalent to rotating the same shape 90 degrees CW.
We could rotate further to get (5, 3, 4, 2, 180), or (5, 3, 4, 2, -180):
```
O--------> x
|
|  C---D
|  | E |
|  B---A
|
v y
```
```
A = (7, 4), B = (3, 4), C = (3, 2), D = (7, 2),
width = |AB| = |CD| = 7 - 3 = 4,
height = |AD| = |BC| = 4 - 2 = 2.
```
Finally, this is a very inaccurate (heavily quantized) illustration of
how (5, 3, 4, 2, 60) looks like in case anyone wonders:
```
O--------> x
|     B\
|    /  C
|   /E /
|  A  /
|   `D
v y
```
It's still a rectangle with center of (5, 3), width of 4 and height of 2,
but its angle (and thus orientation) is somewhere between
(5, 3, 4, 2, 0) and (5, 3, 4, 2, 90).
