"""coco eval for fasterrcnn"""
import json

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_init_value = np.array(0.0)
summary_init = {
    'Precision/mAP': _init_value,
    'Precision/mAP@.50IOU': _init_value,
    'Precision/mAP@.75IOU': _init_value,
    'Precision/mAP (small)': _init_value,
    'Precision/mAP (medium)': _init_value,
    'Precision/mAP (large)': _init_value,
    'Recall/AR@1': _init_value,
    'Recall/AR@10': _init_value,
    'Recall/AR@100': _init_value,
    'Recall/AR@100 (small)': _init_value,
    'Recall/AR@100 (medium)': _init_value,
    'Recall/AR@100 (large)': _init_value,
}


def coco_eval(result_files, result_types, coco, max_dets=None, single_result=False):
    """coco eval for fasterrcnn"""
    anns = json.load(open(result_files['bbox']))
    if not anns:
        return summary_init

    if not max_dets:
        max_dets = (100, 300, 1000)

    if isinstance(coco, str):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    summary_metrics = {}
    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        gt_img_ids = coco.getImgIds()
        det_img_ids = coco_dets.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        eval_coco = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            eval_coco.params.useCats = 0
            eval_coco.params.maxDets = list(max_dets)

        tgt_ids = gt_img_ids if not single_result else det_img_ids

        if single_result:
            res_dict = dict()
            for id_i in tgt_ids:
                eval_coco = COCOeval(coco, coco_dets, iou_type)
                if res_type == 'proposal':
                    eval_coco.params.useCats = 0
                    eval_coco.params.maxDets = list(max_dets)

                eval_coco.params.imgIds = [id_i]
                eval_coco.evaluate()
                eval_coco.accumulate()
                eval_coco.summarize()
                res_dict.update({coco.imgs[id_i]['file_name']: eval_coco.stats[1]})

        eval_coco = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            eval_coco.params.useCats = 0
            eval_coco.params.maxDets = list(max_dets)

        eval_coco.params.imgIds = tgt_ids
        eval_coco.evaluate()
        eval_coco.accumulate()
        eval_coco.summarize()

        summary_metrics = {
            'Precision/mAP': eval_coco.stats[0],
            'Precision/mAP@.50IOU': eval_coco.stats[1],
            'Precision/mAP@.75IOU': eval_coco.stats[2],
            'Precision/mAP (small)': eval_coco.stats[3],
            'Precision/mAP (medium)': eval_coco.stats[4],
            'Precision/mAP (large)': eval_coco.stats[5],
            'Recall/AR@1': eval_coco.stats[6],
            'Recall/AR@10': eval_coco.stats[7],
            'Recall/AR@100': eval_coco.stats[8],
            'Recall/AR@100 (small)': eval_coco.stats[9],
            'Recall/AR@100 (medium)': eval_coco.stats[10],
            'Recall/AR@100 (large)': eval_coco.stats[11],
        }

    return summary_metrics


def xyxy2xywh(bbox):
    """bbox xyxy to xywh."""
    boxes = bbox.tolist()
    return [boxes[0],
            boxes[1],
            boxes[2] - boxes[0] + 1,
            boxes[3] - boxes[1] + 1,
            ]


def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        result = [np.zeros((0, 5), dtype=np.float32)] * (num_classes - 1)
    else:
        result = [bboxes[labels == i, :] for i in range(num_classes - 1)]
    return result


def proposal2json(dataset, results):
    """convert proposal to json mode"""
    img_ids = dataset.getImgIds()
    json_results = []
    dataset_len = dataset.get_dataset_size() * 2
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    """convert det to json mode"""
    cat_ids = dataset.getCatIds()
    img_ids = dataset.getImgIds()
    json_results = []
    dataset_len = len(img_ids)
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        if idx == len(results):
            break
        result = results[idx]
        for label, result_label in enumerate(result):
            bboxes = result_label
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    """convert segm to json mode"""
    cat_ids = dataset.getCatIds()
    img_ids = dataset.getImgIds()
    bbox_json_results = []
    segm_json_results = []

    dataset_len = len(img_ids)
    # assert dataset_len == len(results)
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        if idx == len(results): break
        det, seg = results[idx]
        for label, det_label in enumerate(det):
            bboxes = det_label
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = cat_ids[label]
                bbox_json_results.append(data)

            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def results2json(dataset, results, out_file):
    """convert result convert to json mode"""
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        with open(result_files['bbox'], 'w') as fp:
            json.dump(json_results, fp)
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        with open(result_files['bbox'], 'w') as fp:
            json.dump(json_results[0], fp)
        with open(result_files['segm'], 'w') as fp:
            json.dump(json_results[1], fp)
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        with open(result_files['proposal'], 'w') as fp:
            json.dump(json_results, fp)
    else:
        raise TypeError('Invalid type of results.')
    return result_files


def get_seg_masks(mask_pred, det_bboxes, det_labels, img_meta, rescale, num_classes, mask_thr_binary):
    """Get segmentation masks from mask_pred and bboxes"""
    mask_pred = mask_pred.astype(np.float32)
    cls_segms = [[] for _ in range(num_classes - 1)]
    bboxes = det_bboxes[:, :4]
    labels = det_labels + 1

    ori_shape = bbox = img_meta[:2].astype(np.int32)
    scale_factor = img_meta[2:].astype(np.int32)

    if rescale:
        img_h, img_w = ori_shape[:2]
        img_h = img_h.asnumpy()
        img_w = img_w.asnumpy()
    else:
        img_h = np.round(ori_shape[0] * scale_factor[0]).astype(np.int32)
        img_w = np.round(ori_shape[1] * scale_factor[1]).astype(np.int32)

    for i in range(bboxes.shape[0]):
        bbox = (bboxes[i, :] / 1.0).astype(np.int32)
        label = labels[i]
        w = max(bbox[2] - bbox[0] + 1, 1)
        h = max(bbox[3] - bbox[1] + 1, 1)
        w = min(w, img_w - bbox[0])
        h = min(h, img_h - bbox[1])
        if w <= 0 or h <= 0:
            print("there is invalid proposal bbox, index={} bbox={} w={} h={}".format(i, bbox, w, h))
            w = max(w, 1)
            h = max(h, 1)
        mask_pred_ = mask_pred[i, :, :]
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        bbox_mask = mmcv.imresize(mask_pred_, (w, h))
        bbox_mask = (bbox_mask > mask_thr_binary).astype(np.uint8)
        im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

        rle = maskUtils.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]
        cls_segms[label - 1].append(rle)

    return cls_segms


def xywh2xyxy(bbox):
    """xywh convert into xyxy format."""
    x_min = bbox[0]
    y_min = bbox[1]
    w = bbox[2]
    h = bbox[3]
    return [x_min, y_min, x_min + w, y_min + h]
