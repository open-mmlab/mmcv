## ops

We implement common ops used in detection, segmentation, etc.

| Device                       | CPU | CUDA | MLU | MPS | Ascend |
| ---------------------------- | --- | ---- | --- | --- | ------ |
| ActiveRotatedFilter          | √   | √    |     |     |        |
| AssignScoreWithK             |     | √    |     |     |        |
| BallQuery                    |     | √    |     |     |        |
| BBoxOverlaps                 |     | √    | √   | √   |        |
| BorderAlign                  |     | √    |     |     |        |
| BoxIouRotated                | √   | √    |     |     |        |
| BoxIouQuadri                 | √   | √    |     |     |        |
| CARAFE                       |     | √    | √   |     |        |
| ChamferDistance              |     | √    |     |     |        |
| CrissCrossAttention          |     | √    |     |     |        |
| ContourExpand                | √   |      |     |     |        |
| ConvexIoU                    |     | √    |     |     |        |
| CornerPool                   |     | √    |     |     |        |
| Correlation                  |     | √    |     |     |        |
| Deformable Convolution v1/v2 | √   | √    |     |     | √      |
| Deformable RoIPool           |     | √    | √   |     | √      |
| DiffIoURotated               |     | √    |     |     |        |
| DynamicScatter               |     | √    |     |     |        |
| FurthestPointSample          |     | √    |     |     |        |
| FurthestPointSampleWithDist  |     | √    |     |     |        |
| FusedBiasLeakyrelu           |     | √    |     |     | √      |
| GatherPoints                 |     | √    |     |     |        |
| GroupPoints                  |     | √    |     |     |        |
| Iou3d                        |     | √    | √   |     |        |
| KNN                          |     | √    |     |     |        |
| MaskedConv                   |     | √    | √   |     | √      |
| MergeCells                   |     | √    |     |     |        |
| MinAreaPolygon               |     | √    |     |     |        |
| ModulatedDeformConv2d        | √   | √    |     |     | √      |
| MultiScaleDeformableAttn     |     | √    | √   |     |        |
| NMS                          | √   | √    | √   |     | √      |
| NMSRotated                   | √   | √    |     |     |        |
| NMSQuadri                    | √   | √    |     |     |        |
| PixelGroup                   | √   |      |     |     |        |
| PointsInBoxes                | √   | √    |     |     |        |
| PointsInPolygons             |     | √    |     |     |        |
| PSAMask                      | √   | √    | √   |     | √      |
| RotatedFeatureAlign          | √   | √    |     |     |        |
| RoIPointPool3d               |     | √    | √   |     |        |
| RoIPool                      |     | √    | √   |     | √      |
| RoIAlignRotated              | √   | √    | √   |     |        |
| RiRoIAlignRotated            |     | √    |     |     |        |
| RoIAlign                     | √   | √    | √   |     |        |
| RoIAwarePool3d               |     | √    | √   |     |        |
| SAConv2d                     |     | √    |     |     |        |
| SigmoidFocalLoss             |     | √    | √   |     | √      |
| SoftmaxFocalLoss             |     | √    |     |     | √      |
| SoftNMS                      |     | √    |     |     |        |
| Sparse Convolution           |     | √    |     |     |        |
| Synchronized BatchNorm       |     | √    |     |     |        |
| ThreeInterpolate             |     | √    |     |     |        |
| ThreeNN                      |     | √    | √   |     |        |
| TINShift                     |     | √    | √   |     |        |
| UpFirDn2d                    |     | √    |     |     |        |
| Voxelization                 | √   | √    |     |     |        |
| PrRoIPool                    |     | √    |     |     |        |
| BezierAlign                  | √   | √    |     |     |        |
