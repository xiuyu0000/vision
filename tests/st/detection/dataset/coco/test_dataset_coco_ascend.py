# # Copyright 2022 Huawei Technologies Co., Ltd
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ============================================================================
# """Test COCO dataset operators."""

# import os
# import pytest

# import mindspore.dataset as ds

# from mindvision.msdetection.datasets import COCO

# data_dir = "/home/workspace/mindspore_dataset/mindvision_data/coco/"
# anno_train_dir = "/home/workspace/mindspore_dataset/mindvision_data/coco/annotations/instances_train2017.json"
# anno_val_dir = "/home/workspace/mindspore_dataset/mindvision_data/coco/annotations/instances_val2017.json"
# cur_path = os.path.dirname(os.path.abspath(__file__))


# @pytest.mark.level0
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.env_onecard
# def test_coco_split():
#     """
#     Feature: test class of COCO.
#     Description: test the split parameters of COCO.
#     Expectation: success.
#     """
#     data_train = COCO(path=data_dir,
#                       anno_file=anno_train_dir,
#                       split="train")
#     data_train = data_train.run()
#     assert isinstance(data_train, ds.RepeatDataset)

#     data_eval = COCO(path=data_dir,
#                      anno_file=anno_val_dir,
#                      split="val")
#     data_eval = data_eval.run()
#     assert isinstance(data_eval, ds.RepeatDataset)

#     data_infer = COCO(path=cur_path,
#                       anno_file=anno_val_dir,
#                       split="infer",
#                       batch_size=1)
#     data_infer = data_infer.run()
#     assert isinstance(data_infer, ds.RepeatDataset)


# @pytest.mark.level0
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.env_onecard
# def test_coco_batch_size():
#     """
#     Feature: test class of COCO.
#     Description: test the batch_size parameters of COCO.
#     Expectation: success.
#     """
#     data_train = COCO(path=data_dir,
#                       anno_file=anno_train_dir,
#                       batch_size=64)
#     data_train = data_train.run()
#     step_size = data_train.get_dataset_size()
#     assert data_train.get_batch_size() == 64
#     assert step_size == 1832


# @pytest.mark.level0
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.env_onecard
# def test_coco_repeat():
#     """
#     Feature: test class of COCO.
#     Description: test the repeat parameters of COCO.
#     Expectation: success.
#     """
#     data_train = COCO(path=data_dir,
#                       anno_file=anno_train_dir,
#                       repeat_num=2)
#     data_train = data_train.run()
#     step_size = data_train.get_dataset_size()
#     assert step_size == 3664


# @pytest.mark.level0
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.env_onecard
# def test_coco_resize():
#     """
#     Feature: test class of COCO.
#     Description: test the resize parameters of COCO.
#     Expectation: success.
#     """
#     data_train = COCO(path=data_dir,
#                       anno_file=anno_train_dir,
#                       resize=64)
#     data_train = data_train.run()
#     data_iter = next(data_train.create_dict_iterator(output_numpy=True))
#     images = data_iter["image"]
#     assert images.shape == (64, 3, 64, 64)
