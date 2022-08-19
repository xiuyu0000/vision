# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" group parameters for ConvNeXt """


def get_group(network,
              init_lr,
              warmup_epochs,
              epoch_size,
              lr_scheduler,
              step_per_epoch,
              assigner=None,
              weight_decay=1e-5,
              skip_list=None):
    """
    Get parameters group by parameter's name, each group with different learning rate and weight decay value.

    Args:
        network(nn.Cell): The MindSpore network.
        init_lr(float): Init learning rate.
        warmup_epochs(int): Epoch numbers for warmup training.
        epoch_size(int): Total epoch size.
        lr_scheduler: Learning rate schedule function.
        step_per_epoch(int): Number of steps for one training epoch.
        assigner: Learning rate scale assigner function. Default: None
        weight_decay(float): weight decay init value. Default: 1e-5
        skip_list(List(str)): The list of parameters' names for skipping the group process. Default: None

    Returns:
        List, the groups of network's parameters.
    """

    if skip_list is None:
        skip_list = []
    param_groups = {}

    for (name, param) in network.parameters_and_names():
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list:
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay

        layer_id = get_param_id(name)
        lr_scale = 1.
        if assigner is not None:
            lr_scale = assigner.get_lr_scale(layer_id)
        group_name = "layer_%d_%s" % (layer_id, group_name)
        if group_name not in param_groups:
            param_groups[group_name] = {'params': [],
                                        'weight_decay': this_weight_decay,
                                        'lr': lr_scheduler(lr=init_lr * lr_scale,
                                                           steps_per_epoch=step_per_epoch,
                                                           warmup_epochs=warmup_epochs,
                                                           max_epoch=epoch_size,
                                                           t_max=150,
                                                           eta_min=0)}

        param_groups[group_name]['params'].append(param)

    return list(param_groups.values())


def get_param_id(name):
    """
    Get parameter id number from parameter's name.
    The id range comes from 0 to 13.

    Args:
        name: the parameter's name.

    Returns:
        int, the id of the given parameter.
    """

    name_split = name.split('.')

    layer_id = 13
    if name_split[0] == 'backbone':
        if name_split[1] == 'start_cell':
            layer_id = 0
        elif name_split[1] == 'block1':
            layer_id = 1
        elif name_split[1] == 'down_sample_blocks':
            if name_split[2] in ['0', '1']:
                layer_id = 2
            elif name_split[2] == '2':
                layer_id = 3
            elif name_split[2] == '3':
                layer_id = 3 + int(name_split[3]) // 3
            elif name_split[2] in ['4', '5']:
                layer_id = 12

    return layer_id


class ParamLRValueAssigner:
    """
    For given layer_id, get relative lr scale value.

    Args:
        values: param decay values with length 14 for 14 levels

    Returns:
        float, lr scale value for given layer_id
    """

    def __init__(self, values):
        self.values = values

    def get_lr_scale(self, layer_id):
        """
        get lr scale value for given layer_id
        """
        return self.values[layer_id]
