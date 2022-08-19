## Introduction

MindSpor 3D(ms3d) is an open source 3D Video toolbox based on MindSpore.

## Major features

- Modular Design

We decompose the 3D framework into different components and one can easily construct a customized 3D framework by combining different modules.

### Supported models

- [x] [DGCNN]() for Point Clouds classification and Feature Extraction.
- [x] [PointNet]() for Point Clouds classification and Feature Extraction.
- [x] [Pointnet++]() for Point Clouds classification and Feature Extraction.
- [x] [PointPillars]() for 3D Object Detection.
- [x] [Group-Free-3D]() for 3D Object Detection.
- [x] [Point Transformer]() for 3D Semantic segmentation.
- [x] [VIBE]() for 3D Body Pose Estimation.
- [x] [SMPLify-X]() for 3D Body Pose Estimation.
- [x] [SynergyNet]() for 3D Head Reconstruction.

and where resource is:

- [x] **PointPillars**. Fast Encoders for Object Detection from Point Clouds. PointPillars local paper [address](./point_pillars/1812.05784v2.pdf) for 3D Object Detection, and source paper [link](https://paperswithcode.com/paper/pointpillars-fast-encoders-for-object).
- [x] **Point Transformer**. Point Transformer. Point Transformer local paper [address](./sscns/1711.10275v1.pdf) for 3D Semantic segmentation, and source paper [link](https://arxiv.org/abs/2012.09164).
- [x] **Group-Free-3D**. Group-Free 3D Object Detection via Transformers. Group-Free-3D local paper [address](./groupfree_3d/2104.00678v2.pdf) for 3D Object Detection, and source paper [link](https://paperswithcode.com/paper/group-free-3d-object-detection-via).
- [x] **DGCNN**. Dynamic Graph CNN for Learning on Point Clouds. DGCNN local paper [address](./dgcnn/1801.07829v2.pdf) for Point Clouds classification and Point Clouds Feature Extraction, and source paper [link](https://paperswithcode.com/paper/dynamic-graph-cnn-for-learning-on-point).
- [x] **PointNet**. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. PointNet local paper [address](./pointnet/1612.00593v2.pdf) for Point Clouds classification and Point Clouds Feature Extraction., and source paper [link](https://paperswithcode.com/paper/pointnet-deep-learning-on-point-sets-for-3d).
- [x] **Pointnet++**. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. Pointnet++ local paper [address](./pointnet++/1706.02413v1.pdf) for 3D Semantic segmentation, and source paper [link](https://paperswithcode.com/paper/pointnet-deep-hierarchical-feature-learning).
- [x] **VIBE**. VIBE: Video Inference for Human Body Pose and Shape Estimation. VIBE local paper [address](./vibe/1912.05656v3.pdf) for 3D Body Pose Estimation, and source paper [link](https://paperswithcode.com/paper/vibe-video-inference-for-human-body-pose-and).
- [x] **SMPLify-X**. Expressive Body Capture: 3D Hands, Face, and Body from a Single Image. SMPLify-X local paper [address](./smplify_x/1904.05866v1.pdf) for 3D Body Pose Estimation, and source paper [link](https://paperswithcode.com/paper/expressive-body-capture-3d-hands-face-and).
- [x] **SynergyNet**. Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry. SynergyNet local paper [address](./h3dnet/2107.12512v1.pdf) for 3D Head Reconstruction, and source paper [link](https://arxiv.org/abs/2110.09772).