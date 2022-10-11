## 算子

MMCV 提供了检测、分割等任务中常用的算子

| Device                       | CPU | CUDA | MLU | MPS |
| ---------------------------- | --- | ---- | --- | --- |
| ActiveRotatedFilter          | √   | √    |     |     |
| AssignScoreWithK             |     | √    |     |     |
| BallQuery                    |     | √    |     |     |
| BBoxOverlaps                 |     | √    | √   | √   |
| BorderAlign                  |     | √    |     |     |
| BoxIouRotated                | √   | √    |     |     |
| CARAFE                       |     | √    | √   |     |
| ChamferDistance              |     | √    |     |     |
| CrissCrossAttention          |     | √    |     |     |
| ContourExpand                | √   |      |     |     |
| ConvexIoU                    |     | √    |     |     |
| CornerPool                   |     | √    |     |     |
| Correlation                  |     | √    |     |     |
| Deformable Convolution v1/v2 | √   | √    |     |     |
| Deformable RoIPool           |     | √    | √   |     |
| DiffIoURotated               |     | √    |     |     |
| DynamicScatter               |     | √    |     |     |
| FurthestPointSample          |     | √    |     |     |
| FurthestPointSampleWithDist  |     | √    |     |     |
| FusedBiasLeakyrelu           |     | √    |     |     |
| GatherPoints                 |     | √    |     |     |
| GroupPoints                  |     | √    |     |     |
| Iou3d                        |     | √    |     |     |
| KNN                          |     | √    |     |     |
| MaskedConv                   |     | √    | √   |     |
| MergeCells                   |     | √    |     |     |
| MinAreaPolygon               |     | √    |     |     |
| ModulatedDeformConv2d        | √   | √    |     |     |
| MultiScaleDeformableAttn     |     | √    |     |     |
| NMS                          | √   | √    | √   |     |
| NMSRotated                   | √   | √    |     |     |
| PixelGroup                   | √   |      |     |     |
| PointsInBoxes                | √   | √    |     |     |
| PointsInPolygons             |     | √    |     |     |
| PSAMask                      | √   | √    | √   |     |
| RotatedFeatureAlign          | √   | √    |     |     |
| RoIPointPool3d               |     | √    | √   |     |
| RoIPool                      |     | √    | √   |     |
| RoIAlignRotated              | √   | √    | √   |     |
| RiRoIAlignRotated            |     | √    |     |     |
| RoIAlign                     | √   | √    | √   |     |
| RoIAwarePool3d               |     | √    |     |     |
| SAConv2d                     |     | √    |     |     |
| SigmoidFocalLoss             |     | √    | √   |     |
| SoftmaxFocalLoss             |     | √    |     |     |
| SoftNMS                      |     | √    |     |     |
| Sparse Convolution           |     | √    |     |     |
| Synchronized BatchNorm       |     | √    |     |     |
| ThreeInterpolate             |     | √    |     |     |
| ThreeNN                      |     | √    | √   |     |
| TINShift                     |     | √    | √   |     |
| UpFirDn2d                    |     | √    |     |     |
| Voxelization                 | √   | √    |     |     |
| PrRoIPool                    |     | √    |     |     |
