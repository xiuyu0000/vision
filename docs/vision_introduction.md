# MindSpore Vision套件框架设计

- [MindSpore Vision套件框架设计](#MindSpore Vision套件框架设计)
    - [整体框架设计](#整体框架设计)
    - [配置文件 config](#配置文件-config)
    - [注册机制 registry](#注册机制-registry)
        - [什么是注册机制](#什么是注册机制)
        - [举例](#举例)
    - [数据集 dataset](#数据集-dataset)
        - [数据解析](#数据解析)
        - [注册数据集解析](#注册数据集解析)
        - [配置数据集解析](#配置数据集解析)
        - [数据增强](#数据增强)
        - [注册数据增强](#注册数据增强)
        - [数据增强配置](#数据增强配置)
    - [模型 models](#模型-models)
    - [内部函数 internals](#内部函数-internals)

## 整体框架设计

MindSpore Vision算法套件将基于MindSpore的模型，分解成backbone，neck，head等模块，通过注册机制将模块和对应的参数，通过配置文件+注册的方式，生成各个模块。模块之间尽可能的解耦，使得模块可以根据配置来生成，对比单个模型固定流程的方式灵活性大幅增加。

- tools:  训练和评估和推理的脚本
- utils:  注册和解析配置文件
- configs：各种模型的配置文件
- dataset：数据集加载相关的代码，dataloader，
- Models：
  1. internals：anchor，bbox，learning schedule，optimizer等模型内部相关非Cell类的模块
  2. detector：检测器，比如FasterRCNN，Yolo等模型的入口
  3. backbone：ResNet系列，DarkNet, MobileNet系列等
  4. neck：fpn等
  5. head：模型的head，dense_head, roi_head等

tools下面的train.py为训练入口，主要的调用关系如下：

- 通过build_detector构建detector模型，在构建detector时，会相应的构建detector内部的模块。
- 构建dataset，包含数据集加载和数据增强
- 构建lr_schedule, optimizer等剩余的模块

  以FasterRCNN为例调用关系如下：

  ![image-20210726150505686](https://i.loli.net/2021/07/26/3iYe8sH1GTPELZo.png)

## 配置文件 config

MindSpore Vision整体采用模块化以及继承设计，所有的网络模型均可以采用配置进行组合设计，而且配置文件也可以各自继承，方便进行各种实验。当前MindSpore Vision的配置文件采用yaml格式。配置文件是MindSpore Vision套件模型区别于ModelZoo上模型的主要特点。这种机制将各个模块（类）的配置通过build装配成一个一个具体模块，实现模型，优化器，数据集增强等模块参数的可配。

具体配置文件设计参考yaml_config.md

## 注册机制 registry

注册机制是检测套件为将网络模型模块化而设计；例如，检测器中的backbone、neck、head等。在检测套件中，数据处理以及网络模型相关的模块均是采用注册机制进行模块化管理。

### 什么是注册机制

在MindSpore Vision套件中，注册机制可以理解为是一个将类映射到字符串的映射关系。每一个映射表是功能类似的类的集合。而开发者在使用时则可以通过对应的字符串查找相关类并实例化该类。而注册机制通常和配置文件一起使用，在使用检测套件实现一个网络时，开发者通过配置创建检测模型以及数据集，进而可以进行模型训练或推理。

在检测套件中，主要有两个类实现这层映射关系：`ClassFactory`和`ModuleType`。

其中`Class ModuleType`是声明了MindSpore Vision套件中支持的注册表的类别。

```python
class ModuleType:
    """Class module type"""
    DATASET = 'dataset'                     # 数据集
    DATASET_LOADER = 'dataset_loader'       # 数据加载
    DATASET_SAMPLER = 'dataset_sampler'     # 数据集采样器
    PIPELINE = 'pipeline'                   # 数据增强流水线
    BACKBONE = 'backbone'                   # 主干网络
    DETECTOR = 'detector'                   # 检测器
    HEAD = 'head'                           # 检测头
    NECK = 'neck'                           # 连接颈部
    LOSS = 'loss'                           # 损失函数
    OPTIMIZER = 'optimizer'                 # 优化器
    ANCHOR_GENERATOR = 'anchor generator'   # 锚框生成器
    WRAPPER = 'wrapper'                     # 训练封包器

    # bbox
    BBOX_ASSIGNERS = 'bbox_assigner'
    BBOX_SAMPLERS = 'bbox_sampler'          # bbox 采样
    BBOX_CODERS = 'bbox_coder'              # bbox 编解码器

    GENERAL = 'general'                     # 通用表
```

若当前的注册表类别不能满足开发者诉求，开发者可以自行在上述类中增加所需类型字符即可。

`Class ClassFactory`实现了注册表的管理功能，主要实现了注册表注册以及获取注册表中的类并实例化的功能。

1. 注册模块有两种方式：一是通过调用`ClassFactory.register()`装饰器就行注册，二是显示的调用`ClassFactory.register_cls()`函数进行注册。
2. 获取注册表中的类实例：一种配合配置文件，通过配置信息获取一个实例`ClassFactory.get_instance_from_cfg()`，二是通过函数传参的方式获取一个实例`ClassFactory.get_instance()`。

### 举例

例如在MindSpore Vision分类套件中实现一个主干网络的注册和实例化。

套件中注册主干网络network1，例如，`bacbone/network1.py`：

```python
from mindspore import nn

from utils.class_factory import ClassFactory, ModuleType
@ClassFactory.register(ModuleType.BACKBONE)
class Network1(nn.Cell):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def construct(self, x):
        out = self.a * x + b
        return out
```

上述代码中则是调用了注册机制中的注册装饰器，将主干网络映射为字符串：'Network1' -> <class 'Network1'>

经过上面的注册过程之后，则可以通过从配置文件中来实例化network1，则只需要调用套件中实例化主干网络的API即可：

```python
config = Config(**dict(type='Network1', a='0.9', b='0.01'))
network1 = build_backbone(config)
```

其中，config 为主干网络network的配置信息，build_backbone是封装了注册模块中实例化的接口：

```python
def build_backbone(cfg):
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.BACKBONE)
```

## 数据集 dataset

MindSpore Vision套件的数据集加载模块主要实现了数据集解析和数据集增强流水线功能。通过将数据处理的各个功能最小化模块化，从而实现了数据集处理可配置。

### 数据解析

在套件中，数据集解析主要是调用MindSpore内置的数据集[`APIs`](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/mindspore.dataset.html)；在套件中数据集解析有[`mindspore.dataset.CocoDataset`](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/dataset/mindspore.dataset.CocoDataset.html#mindspore.dataset.CocoDataset)、[`mindspore.dataset.VOCDataset`](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/dataset/mindspore.dataset.VOCDataset.html#mindspore.dataset.VOCDataset)、[`mindspore.dataset.MindDataset`](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/dataset/mindspore.dataset.MindDataset.html#mindspore.dataset.MindDataset)、[`mindspore.dataset.GeneratorDataset`](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)等。

### 注册数据集解析

对于MindSpore的数据集解析接口，通过`register_ms_builtin_dataset()`注册到注册表中，这样开发者就能在配置中直接配置使用。在`register_ms_builtin_dataset()`主要是完成对`MindSpore.dataset()`中所有的类进行遍历而后注册到注册表中。

```python
def register_ms_builtin_dataset():
    """ register MindSpore builtin dataset class. """
    for module_name in dir(ms.dataset):
        if module_name.startswith('__'):
            continue
        dataset = getattr(ms.dataset, module_name)
        if inspect.isclass(dataset):
            ClassFactory.register_cls(dataset, ModuleType.DATASET)
```

### 配置数据集解析

在训练、验证和推理时，只需要在配置文件中配置已经注册的数据集解析类即可。例如，训练配置：

```yaml
train:
    dataset:
        type: MindDataset
        dataset_file: ["/home/lid/workspace/mindvision/mindvision/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord0",
                        "/home/lid/workspace/mindvision/mindvision/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord1",
                        "/home/lid/workspace/mindvision/mindvision/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord2",
                        "/home/lid/workspace/mindvision/mindvision/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord3",
                        "/home/lid/workspace/mindvision/mindvision/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord4",
                        "/home/lid/workspace/mindvision/mindvision/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord5",
                        "/home/lid/workspace/mindvision/mindvision/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord6",
                        "/home/lid/workspace/mindvision/mindvision/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord7"]
        columns_list: ["image", "annotation"]
        num_shards: 1
        shard_id: 0
        num_parallel_workers: 1
        shuffle: False
```

由于数据集类型是MindSpore自研数据格式MindRecord，所以需要选择MindDataset进行数据解析，type关键字下面的配置则是MindDataset类的参数配置。

### 数据增强

在套件中，开发者在使用时只需要新增新的数据增强处理算子和配置数据增强处理配置文件即可。

套件的数据增强处理是调用MindSpore的map()和batch()算子进行数据增强的流水线构建。map将指定的函数或算子作用于数据集的指定列数据，实现数据映射操作。用户可以自定义映射函数，也可以直接使用`c_transforms`或`py_transforms`中的算子针对图像、文本数据进行数据增强。batch将数据集分批，并且可以对批数据进行数据增强。

数据增强是数据集解析完后对图像数据进行缩放、旋转、填充等变换操作的过程，经历了数据增强后的图像数据就会被输入到网络中进行计算。套件将数据增强设计为流水线形式，每一个数据变换则为流水线的一个处理单元，进而可以将不同网络的数据增强操作统一且灵活可配置。

![数据增强流水线](https://i.loli.net/2021/07/26/MdjoL329avP87s1.png)

上图中的op1、op2、op3、...、opN是一个个数据增强的算子，MindSpore Vision套件通过配置参数中将算子按照配置顺序串联得到最终的数据增强的operations，作为参数传递给MindSpore的map和batch算子。

### 注册数据增强

为了实现可配置，在MindSpore Vision套件中需要注册各个数据增强的算法。例如，在检测套件中实现了Resize的功能：

```python
@ClassFactory.register(ModuleType.PIPELINE)
class Resize:
    """resize operation for image"""

    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, results):
        img = results.get("image")
        gt_bboxes = results.get("bboxes")

        img_data = img
        img_data, w_scale, h_scale = mmcv.imresize(
            img_data, (self.img_width, self.img_height), return_scale=True)
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        img_shape = (self.img_height, self.img_width, 1.0)
        img_shape = np.asarray(img_shape, dtype=np.float32)

        gt_bboxes = gt_bboxes * scale_factor

        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

        results['image'] = img_data
        results['image_shape'] = img_shape
        results['bboxes'] = gt_bboxes

        return results
```

检测套件中通过`@ClassFactory.register(ModuleType.PIPELINE)`装饰器将Resize类注册到pipeline中，开发者在使用时则只需要在数据增强流水线配置Resize类以及其所需要的参数；例如：

```yaml
map:
    operations:
        - type: Resize
          img_height: 768
          img_width: 128
```

### 数据增强配置

map算子的数据增强配置是配置map算子的operation参数，例如，配置fasterrcnn的数据处理：

```yaml
map:
            operations:
                - type: Format
                  pad_max_number: 128
                - type: Decode
                  decode_mode: C
                - type: ImgRgbToBgr
                - type: RandomExpand
                  # mean: (0, 0, 0) # TODO
                  to_rgb: True
                  expand_ratio: 1.0
                  # ratio_range: (1, 4) # TODO
                - type: Resize
                  img_height: 768
                  img_width: 1280
                - type: Normalize
                  mean: [123.675, 116.28, 103.53]
                  std: [58.395, 57.12, 57.375]
                  to_rgb: True
                - type: RandomFlip
                  flip_ratio: 0.5
                - type: Transpose
                  # perm: (2, 0, 1) # TODO
                - type: Collect
                  output_orders: ["image", "image_shape", "bboxes", "labels", "valid_num"]
                  output_type_dict:
                      image: float32
                      image_shape: float32
                      bboxes: float32
                      labels: int32
                      valid_num: bool

            input_columns: ["image", "annotation"]
            output_columns: ["image", "image_shape", "bboxes", "labels", "valid_num"]
            column_order: ["image", "image_shape", "bboxes", "labels", "valid_num"]
            python_multiprocessing: False
            num_parallel_workers: 1
```

在上述的配置中，fasterrcnn配置了Format、Decode、ImgRgbToBgr、RandomExpand、Resize、Normalize、RandomFlip、Collect共8个算子进行数据增强操作，在训练网络时数据增强的算子按照配置中算子出现的顺序执行。

其余的input_columns、output_columns、column_order、python_multiprocessing、num_parallel_workers等也是map算子的参数。

batch算子的数据配置是配置per_batch_map的参数，例如在YOLOv4的数据增强配置：

```yaml
batch:
            per_batch_map:
                type: PerBatchMap
                out_orders: ["image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3"]
                multi_scales: [ [ 416, 416 ],
                                      [ 448, 448 ],
                                      [ 480, 480 ],
                                      [ 512, 512 ],
                                      [ 544, 544 ],
                                      [ 576, 576 ],
                                      [ 608, 608 ],
                                      [ 640, 640 ],
                                      [ 672, 672 ],
                                      [ 704, 704 ],
                                      [ 736, 736 ] ]
                pipeline:
                    - type: PerBatchCocoCollect
                    - type: ResizeWithinMultiScales
                      max_boxes: 90
                      jitter: 0.3
                      max_trial: 10
                    - type: RandomPilFlip
                    - type: ConvertGrayToColor
                    - type: ColorDistortion
                      hue: 0.1
                      saturation: 1.5
                      value: 1.5
                    - type: Normalize
                      mean: [0.485, 0.456, 0.406]
                      std: [0.229, 0.224, 0.225]
                    - type: Transpose
                    - type: YoloBboxPreprocess
                      anchors: [[12, 16],
                                [19, 36],
                                [40, 28],
                                [36, 75],
                                [76, 55],
                                [72, 146],
                                [142, 110],
                                [192, 243],
                                [459, 401]]
                      anchor_mask: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                      num_classes: 80
                      label_smooth: 0
                      label_smooth_factor: 0.1
                      iou_threshold: 0.213
                      max_boxes: 90

            input_columns: [ "image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3" ]
            output_columns: [ "image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3" ]
            num_parallel_workers: 8
            drop_remainder: True
            batch_size: 8
```

训练时配置了PerBatchMap类进行per_batch_map的数据增强算子的配置；而在PerBatchMap类的参数配置中的pipeline则是数据增强的操作，配置了PerBatchCocoCollect、ResizeWithinMultiScales、RandomPilFlip、ConvertGrayToColor、ColorDistortion、Normalize、Transpose、YoloBboxPreprocess共8个算子。

其余的input_columns、output_columns、num_parallel_workers、drop_remainder、batch_size等也是batch算子的参数。

## 模型 models

models中包含了所有有关模型的代码，没一个模块都用继承了MindSpore的Cell。在__init__函数中构造各个模块。在contruct/construct_train/construct_test中执行构建好的模块和算子。

1. 顶层架构 meta_arch

MindVision的Detection套件包含了一个一个顶层设计模块，目前将检测类的模型抽象成一阶段（one_stage) ，二阶段（two_stage）的两类

2. construct_train, construct_test

   代码中我们将train和test的代码分开，训练时只用到contruct_train的代码，推理时只用construct_test的代码

   ```python
       def construct_train(self, x, proposal, proposal_mask, gt_bboxes, gt_labels, gt_valids):
           """train construct of roi head"""
           gt_labels = self.cast(gt_labels, mstype.int32)
           gt_valids = self.cast(gt_valids, mstype.int32)
           bboxes_tuple = ()
           deltas_tuple = ()
           labels_tuple = ()
           mask_tuple = ()
           for i in range(self.batch_size):
               gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])

               gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
               gt_labels_i = self.cast(gt_labels_i, mstype.uint8)

               gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
               gt_valids_i = self.cast(gt_valids_i, mstype.bool_)

               bboxes, deltas, labels, mask = self.bbox_assigner_sampler_for_rcnn(gt_bboxes_i,
                                                                                  gt_labels_i,
                                                                                  proposal_mask[i],
                                                                                  proposal[i][::, 0:4:1],
                                                                                  gt_valids_i)
               bboxes_tuple += (bboxes,)
               deltas_tuple += (deltas,)
               labels_tuple += (labels,)
               mask_tuple += (mask,)

           bbox_targets = self.concat(deltas_tuple)
           rcnn_labels = self.concat(labels_tuple)
           bbox_targets = F.stop_gradient(bbox_targets)
           rcnn_labels = F.stop_gradient(rcnn_labels)
           rcnn_labels = self.cast(rcnn_labels, mstype.int32)

           if self.batch_size > 1:
               bboxes_all = self.concat(bboxes_tuple)
           else:
               bboxes_all = bboxes_tuple[0]
           rois = self.concat_1((self.roi_align_index_tensor, bboxes_all))

           rois = self.cast(rois, mstype.float32)
           rois = F.stop_gradient(rois)

           roi_feats = self.roi_align(rois, self.cast(x[0], mstype.float32), self.cast(x[1], mstype.float32),
                                      self.cast(x[2], mstype.float32), self.cast(x[3], mstype.float32))

           roi_feats = self.cast(roi_feats, self.ms_type)
           rcnn_masks = self.concat(mask_tuple)
           rcnn_masks = F.stop_gradient(rcnn_masks)
           rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))
           _, rcnn_cls_loss, rcnn_reg_loss, _ = self.rcnn(roi_feats, bbox_targets, rcnn_labels, rcnn_mask_squeeze)
           output = rcnn_cls_loss, rcnn_reg_loss

           return output

       def construct_test(self, x, img_metas, proposal, proposal_mask):
           """construct test of roi_head"""
           bboxes_tuple = ()
           mask_tuple = ()
           mask_tuple += proposal_mask
           bbox_targets = proposal_mask
           rcnn_labels = proposal_mask
           for p_i in proposal:
               bboxes_tuple += (p_i[::, 0:4:1],)

           if self.test_batch_size > 1:
               bboxes_all = self.concat(bboxes_tuple)
           else:
               bboxes_all = bboxes_tuple[0]
           rois = self.concat_1((self.roi_align_index_test_tensor, bboxes_all))

           rois = F.stop_gradient(rois)

           roi_feats = self.roi_align_test(rois,
                                           self.cast(x[0], mstype.float32),
                                           self.cast(x[1], mstype.float32),
                                           self.cast(x[2], mstype.float32),
                                           self.cast(x[3], mstype.float32))
           roi_feats = self.cast(roi_feats, self.ms_type)
           rcnn_masks = self.concat(mask_tuple)
           rcnn_masks = F.stop_gradient(rcnn_masks)
           rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))
           _, rcnn_cls_loss, rcnn_reg_loss, _ = self.rcnn(roi_feats, bbox_targets, rcnn_labels, rcnn_mask_squeeze)
           output = self.get_det_bboxes(rcnn_cls_loss, rcnn_reg_loss, rcnn_masks, bboxes_all, img_metas)

           return output
   ```

3. backbone, neck, head, losses

   在两阶段construct的流程中，图片分别通过backbone，neck，rpn_head， roi_head 等流程最终计算loss求和。

   ```python
   class TwoStageDetector(BaseDetector):
       def __init__(self, config, backbone, neck, rpn_head, roi_head, train_cfg, test_cfg):
           super().__init__()
           # backbone
           self.backbone = build_backbone(backbone)
           # fpn
           if neck is not None:
               self.neck = build_neck(neck)
           # rpn and rpn loss
           if train_cfg is not None:
               rpn_head.update(train_cfg=train_cfg)#将train_cfg 加入到rpn_head的配置中
               roi_head.update(train_cfg=train_cfg)#将train_cfg 加入到roi_head的配置中
           if test_cfg is not None:
               rpn_head.update(test_cfg=test_cfg)#将test_cfg 加入到rpn_head的配置中
               roi_head.update(test_cfg=test_cfg)#将test_cfg 加入到roi_head的配置中

           self.rpn_head = build_head(rpn_head)
           # proposal
           # self.proposal_generator = build_proposal(config)
           self.roi_head = build_head(roi_head)

       def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids):
           """Construct of two stage detector."""
           x = self.backbone(img_data)#执行backbone
           if self.has_neck:
               x = self.neck(x)#如果有neck，执行neck

           if self.training:
               rpn_cls_loss, rpn_reg_loss, proposal, proposal_mask = self.rpn_head.construct_train(x, img_metas,
                                                                                                   gt_bboxes, gt_valids)#执行训练的rpn流程
               roi_cls_loss, roi_reg_loss = self.roi_head.construct_train(x, proposal, proposal_mask, gt_bboxes, gt_labels, gt_valids)#执行训练的roi_head
               return rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss
           proposal, proposal_mask = self.rpn_head.construct_test(x)# 执行推理的rpn_head
           output = self.roi_head.construct_test(x, img_metas, proposal, proposal_mask)# 执行推理的roi_head
           return output
   ```

## 内部函数 internals

internals包含了模型中间需要的一些内部函数，比如assigner，sampler。通过这些函数配合models的各种module，组合成模型的主体

- anchor:

  当前anchor模块主要包含了anchor generator。通过配置生成anchor_generator再通过get_anchors产生anchor。

  ```python
      def get_anchors(self, featmap_sizes):
          """Get anchors according to feature map sizes.

          Args:
              featmap_sizes (list[tuple]): Multi-level feature map sizes.
              img_metas (list[dict]): Image meta info.

          Returns:
              tuple: anchors of each image, valid flags of each image.
          """
          num_levels = len(featmap_sizes)

          # since feature map sizes of all images are the same, we only compute
          # anchors for one time
          multi_level_anchors = ()
          for i in range(num_levels):
              anchors = self.grid_anchors(
                  featmap_sizes[i], self.base_sizes[i])
              multi_level_anchors += (Tensor(anchors.astype(np.float32)),)

          return multi_level_anchors
  ```

- bbox

  跟bounding box相关的模块都放在bbox文件夹下，目前有assigner,sampler：根据IOU标记前背景的模块，并且采样。iou_caculator，多种计算iou的方式。和其他模块一样有builder，通过配置构建各个子模块。

  ```python
  class Iou(nn.Cell):
      """Calculate the iou of boxes"""
      def __init__(self):
          super().__init__()
          self.min = ops.Minimum()
          self.max = ops.Maximum()

      def construct(self, box1, box2):
          """
          box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
          box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
          convert to topLeft and rightDown
          """
          box1_xy = box1[:, :, :, :, :, :2]
          box1_wh = box1[:, :, :, :, :, 2:4]
          box1_mins = box1_xy - box1_wh / F.scalar_to_array(2.0) # topLeft
          box1_maxs = box1_xy + box1_wh / F.scalar_to_array(2.0) # rightDown

          box2_xy = box2[:, :, :, :, :, :2]
          box2_wh = box2[:, :, :, :, :, 2:4]
          box2_mins = box2_xy - box2_wh / F.scalar_to_array(2.0)
          box2_maxs = box2_xy + box2_wh / F.scalar_to_array(2.0)

          intersect_mins = self.max(box1_mins, box2_mins)
          intersect_maxs = self.min(box1_maxs, box2_maxs)
          intersect_wh = self.max(intersect_maxs - intersect_mins, F.scalar_to_array(0.0))
          # ops.squeeze: for effiecient slice
          intersect_area = ops.Squeeze(-1)(intersect_wh[:, :, :, :, :, 0:1]) * \
                           ops.Squeeze(-1)(intersect_wh[:, :, :, :, :, 1:2])
          box1_area = ops.Squeeze(-1)(box1_wh[:, :, :, :, :, 0:1]) * ops.Squeeze(-1)(box1_wh[:, :, :, :, :, 1:2])
          box2_area = ops.Squeeze(-1)(box2_wh[:, :, :, :, :, 0:1]) * ops.Squeeze(-1)(box2_wh[:, :, :, :, :, 1:2])
          iou = intersect_area / (box1_area + box2_area - intersect_area)
          # iou : [batch, gx, gy, anchors, maxboxes]
          return iou
  ```

- lr_schedule: 学习率的调整策略

  Detection可以使用学习率调整策略来调整在训练过程中的学习率，所有学习率调整的策略都在lr_schedule.py中。目前包含的学习率调整策略有'exponential',  'cosine_annealing',  'cosine_annealing_V2',  'cosine_annealing_sample',  'dynamic_lr'。用户也可以自行设计学习率调整策略。

  主要包含warmup的学习率策略，中间steps（epochs）的学习率策略，和最后结束训练前的学习率策略，最终将每个step的学习率转化为一个numpy array或者list。

  ```python
  def warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
      """Warmup step learning rate."""
      base_lr = lr
      warmup_init_lr = 0
      total_steps = int(max_epoch * steps_per_epoch)
      warmup_steps = int(warmup_epochs * steps_per_epoch)
      milestones = lr_epochs
      milestones_steps = []
      for milestone in milestones:
          milestones_step = milestone * steps_per_epoch
          milestones_steps.append(milestones_step)

      lr_each_step = []
      lr = base_lr
      milestones_steps_counter = Counter(milestones_steps)
      for i in range(total_steps):
          if i < warmup_steps:
              lr = linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, warmup_init_lr)
          else:
              lr = lr * gamma**milestones_steps_counter[i]
          lr_each_step.append(lr)

      return np.array(lr_each_step).astype(np.float32)
  ```

- optimizer：优化器的注册模块

  优化器目前我们采用注册的方式将默认的optimizer都注册进到我们的ClassFactory，这样就可以通过配置的方式进行配置。默认的optimizer有'Momentum', 'LARS', 'Adam', 'AdamWeightDecay', 'LazyAdam', 'AdamOffload', 'Lamb', 'SGD', 'FTRL', 'RMSProp', 'ProximalAdagrad', 'Adagrad', 'thor'等。

  ```python
  def register_optimizers():
      for module_name in dir(ms.nn):
          if module_name.startswith('__'):
              continue
          opt = getattr(ms.nn, module_name)
          if inspect.isclass(opt) and issubclass(opt, ms.nn.Optimizer):
              ClassFactory.register_cls(opt, ModuleType.OPTIMIZER
  ```