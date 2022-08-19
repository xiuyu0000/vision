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
"""S3FD Detection Engine. """

import numpy as np
from mindspore import ops
from mindspore import Tensor
from mindvision.msdetection.utils.bbox import nms, batch_decode


class S3FDDetectionEngine:
    """Detection engine."""

    def __init__(self, olist, nms_thresh, imgs_num, variances):
        self.olist = olist
        self.nms_thresh = nms_thresh
        self.imgs_num = imgs_num
        self.variances = variances

    def detect(self):
        """Detect face rectangle"""
        olist = self.olist
        bboxlists = []
        for i in range(len(olist) // 2):
            olist[i * 2] = ops.Softmax(axis=1)(olist[i * 2])
        olist = [oelem for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2**(i + 2)
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for _, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[:, 1, int(hindex), int(windex)]
                loc = Tensor(oreg[:, :, int(hindex), int(windex)]).copy().view(self.imgs_num, 1, 4)
                priors = Tensor([axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]).view(1, 1, 4)
                box = batch_decode(loc, priors, self.variances)
                box = box[:, 0] * 1.0
                bboxlists.append(ops.Concat(1)([box, ops.ExpandDims()(score, 1)]).asnumpy())
        bboxlists = np.array(bboxlists)
        if not bboxlists:
            bboxlists = np.zeros((1, self.imgs_num, 5))
        keeps = [nms(bboxlists[:, i, :], self.nms_thresh) for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.5] for bboxlist in bboxlists]

        results = []
        for i, d in enumerate(bboxlists):
            if not d:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)
            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results
