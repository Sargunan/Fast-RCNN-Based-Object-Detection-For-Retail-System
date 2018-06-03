from abc import ABCMeta
from abc import abstractmethod
import collections
import logging
import numpy as np

from object_detection.core import standard_fields
from object_detection.utils import label_map_util
from object_detection.utils import metrics
from object_detection.utils import per_image_evaluation


class DetectionEvaluator(object):

  __metaclass__ = ABCMeta

  def __init__(self, categories):

    self._categories = categories

  @abstractmethod
  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):

    pass

  @abstractmethod
  def add_single_detected_image_info(self, image_id, detections_dict):

    pass

  @abstractmethod
  def evaluate(self):
    pass

  @abstractmethod
  def clear(self):
    pass


class ObjectDetectionEvaluator(DetectionEvaluator):

  def __init__(self,
               categories,
               matching_iou_threshold=0.5,
               evaluate_corlocs=False,
               metric_prefix=None,
               use_weighted_mean_ap=False,
               evaluate_masks=False,
               group_of_weight=0.0):
    super(ObjectDetectionEvaluator, self).__init__(categories)
    self._num_classes = max([cat['id'] for cat in categories])
    if min(cat['id'] for cat in categories) < 1:
      raise ValueError('Classes should be 1-indexed.')
    self._matching_iou_threshold = matching_iou_threshold
    self._use_weighted_mean_ap = use_weighted_mean_ap
    self._label_id_offset = 1
    self._evaluate_masks = evaluate_masks
    self._group_of_weight = group_of_weight
    self._evaluation = ObjectDetectionEvaluation(
        num_groundtruth_classes=self._num_classes,
        matching_iou_threshold=self._matching_iou_threshold,
        use_weighted_mean_ap=self._use_weighted_mean_ap,
        label_id_offset=self._label_id_offset,
        group_of_weight=self._group_of_weight)
    self._image_ids = set([])
    self._evaluate_corlocs = evaluate_corlocs
    self._metric_prefix = (metric_prefix + '_') if metric_prefix else ''

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    if image_id in self._image_ids:
      raise ValueError('Image with id {} already added.'.format(image_id))

    groundtruth_classes = (
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_classes] -
        self._label_id_offset)
    # If the key is not present in the groundtruth_dict or the array is empty
    # (unless there are no annotations for the groundtruth on this image)
    # use values from the dictionary or insert None otherwise.
    if (standard_fields.InputDataFields.groundtruth_difficult in
        groundtruth_dict.keys() and
        (groundtruth_dict[standard_fields.InputDataFields.groundtruth_difficult]
         .size or not groundtruth_classes.size)):
      groundtruth_difficult = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_difficult]
    else:
      groundtruth_difficult = None
      if not len(self._image_ids) % 1000:
        logging.warn(
            'image %s does not have groundtruth difficult flag specified',
            image_id)
    groundtruth_masks = None
    if self._evaluate_masks:
      if (standard_fields.InputDataFields.groundtruth_instance_masks not in
          groundtruth_dict):
        raise ValueError('Instance masks not in groundtruth dictionary.')
      groundtruth_masks = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_instance_masks]
    self._evaluation.add_single_ground_truth_image_info(
        image_key=image_id,
        groundtruth_boxes=groundtruth_dict[
            standard_fields.InputDataFields.groundtruth_boxes],
        groundtruth_class_labels=groundtruth_classes,
        groundtruth_is_difficult_list=groundtruth_difficult,
        groundtruth_masks=groundtruth_masks)
    self._image_ids.update([image_id])

  def add_single_detected_image_info(self, image_id, detections_dict):
    detection_classes = (
        detections_dict[standard_fields.DetectionResultFields.detection_classes]
        - self._label_id_offset)
    detection_masks = None
    if self._evaluate_masks:
      if (standard_fields.DetectionResultFields.detection_masks not in
          detections_dict):
        raise ValueError('Detection masks not in detections dictionary.')
      detection_masks = detections_dict[
          standard_fields.DetectionResultFields.detection_masks]
    self._evaluation.add_single_detected_image_info(
        image_key=image_id,
        detected_boxes=detections_dict[
            standard_fields.DetectionResultFields.detection_boxes],
        detected_scores=detections_dict[
            standard_fields.DetectionResultFields.detection_scores],
        detected_class_labels=detection_classes,
        detected_masks=detection_masks)

  def evaluate(self):
    (per_class_ap, mean_ap, _, _, per_class_corloc, mean_corloc) = (
        self._evaluation.evaluate())
    pascal_metrics = {
        self._metric_prefix +
        'Precision/mAP@{}IOU'.format(self._matching_iou_threshold):
            mean_ap
    }
    if self._evaluate_corlocs:
      pascal_metrics[self._metric_prefix + 'Precision/meanCorLoc@{}IOU'.format(
          self._matching_iou_threshold)] = mean_corloc
    category_index = label_map_util.create_category_index(self._categories)
    for idx in range(per_class_ap.size):
      if idx + self._label_id_offset in category_index:
        display_name = (
            self._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                self._matching_iou_threshold,
                category_index[idx + self._label_id_offset]['name']))
        pascal_metrics[display_name] = per_class_ap[idx]

        # Optionally add CorLoc metrics.classes
        if self._evaluate_corlocs:
          display_name = (
              self._metric_prefix + 'PerformanceByCategory/CorLoc@{}IOU/{}'
              .format(self._matching_iou_threshold,
                      category_index[idx + self._label_id_offset]['name']))
          pascal_metrics[display_name] = per_class_corloc[idx]

    return pascal_metrics

  def clear(self):
    self._evaluation = ObjectDetectionEvaluation(
        num_groundtruth_classes=self._num_classes,
        matching_iou_threshold=self._matching_iou_threshold,
        use_weighted_mean_ap=self._use_weighted_mean_ap,
        label_id_offset=self._label_id_offset)
    self._image_ids.clear()


class PascalDetectionEvaluator(ObjectDetectionEvaluator):

  def __init__(self, categories, matching_iou_threshold=0.5):
    super(PascalDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='PascalBoxes',
        use_weighted_mean_ap=False)


class WeightedPascalDetectionEvaluator(ObjectDetectionEvaluator):
  def __init__(self, categories, matching_iou_threshold=0.5):
    super(WeightedPascalDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='WeightedPascalBoxes',
        use_weighted_mean_ap=True)


class PascalInstanceSegmentationEvaluator(ObjectDetectionEvaluator):

  def __init__(self, categories, matching_iou_threshold=0.5):
    super(PascalInstanceSegmentationEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='PascalMasks',
        use_weighted_mean_ap=False,
        evaluate_masks=True)


class WeightedPascalInstanceSegmentationEvaluator(ObjectDetectionEvaluator):
  def __init__(self, categories, matching_iou_threshold=0.5):
    super(WeightedPascalInstanceSegmentationEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='WeightedPascalMasks',
        use_weighted_mean_ap=True,
        evaluate_masks=True)


class OpenImagesDetectionEvaluator(ObjectDetectionEvaluator):
  def __init__(self,
               categories,
               matching_iou_threshold=0.5,
               evaluate_corlocs=False,
               metric_prefix='OpenImagesV2',
               group_of_weight=0.0):
    super(OpenImagesDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold,
        evaluate_corlocs,
        metric_prefix=metric_prefix,
        group_of_weight=group_of_weight)

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    if image_id in self._image_ids:
      raise ValueError('Image with id {} already added.'.format(image_id))

    groundtruth_classes = (
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_classes] -
        self._label_id_offset)
    # If the key is not present in the groundtruth_dict or the array is empty
    # (unless there are no annotations for the groundtruth on this image)
    # use values from the dictionary or insert None otherwise.
    if (standard_fields.InputDataFields.groundtruth_group_of in
        groundtruth_dict.keys() and
        (groundtruth_dict[standard_fields.InputDataFields.groundtruth_group_of]
         .size or not groundtruth_classes.size)):
      groundtruth_group_of = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_group_of]
    else:
      groundtruth_group_of = None
      if not len(self._image_ids) % 1000:
        logging.warn(
            'image %s does not have groundtruth group_of flag specified',
            image_id)
    self._evaluation.add_single_ground_truth_image_info(
        image_id,
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_boxes],
        groundtruth_classes,
        groundtruth_is_difficult_list=None,
        groundtruth_is_group_of_list=groundtruth_group_of)
    self._image_ids.update([image_id])


class OpenImagesDetectionChallengeEvaluator(OpenImagesDetectionEvaluator):
  def __init__(self,
               categories,
               matching_iou_threshold=0.5,
               evaluate_corlocs=False,
               group_of_weight=1.0):
    super(OpenImagesDetectionChallengeEvaluator, self).__init__(
        categories,
        matching_iou_threshold,
        evaluate_corlocs,
        metric_prefix='OpenImagesChallenge2018',
        group_of_weight=group_of_weight)

    self._evaluatable_labels = {}

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    super(OpenImagesDetectionChallengeEvaluator,
          self).add_single_ground_truth_image_info(image_id, groundtruth_dict)
    groundtruth_classes = (
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_classes] -
        self._label_id_offset)
    self._evaluatable_labels[image_id] = np.unique(
        np.concatenate(((groundtruth_dict.get(
            standard_fields.InputDataFields.verified_labels,
            np.array([], dtype=int)) - self._label_id_offset),
                        groundtruth_classes)))

  def add_single_detected_image_info(self, image_id, detections_dict):
    if image_id not in self._image_ids:
      # Since for the correct work of evaluator it is assumed that groundtruth
      # is inserted first we make sure to break the code if is it not the case.
      self._image_ids.update([image_id])
      self._evaluatable_labels[image_id] = np.array([])

    detection_classes = (
        detections_dict[standard_fields.DetectionResultFields.detection_classes]
        - self._label_id_offset)
    allowed_classes = np.where(
        np.isin(detection_classes, self._evaluatable_labels[image_id]))
    detection_classes = detection_classes[allowed_classes]
    detected_boxes = detections_dict[
        standard_fields.DetectionResultFields.detection_boxes][allowed_classes]
    detected_scores = detections_dict[
        standard_fields.DetectionResultFields.detection_scores][allowed_classes]

    self._evaluation.add_single_detected_image_info(
        image_key=image_id,
        detected_boxes=detected_boxes,
        detected_scores=detected_scores,
        detected_class_labels=detection_classes)

  def clear(self):
    """Clears stored data."""

    super(OpenImagesDetectionChallengeEvaluator, self).clear()
    self._evaluatable_labels.clear()


ObjectDetectionEvalMetrics = collections.namedtuple(
    'ObjectDetectionEvalMetrics', [
        'average_precisions', 'mean_ap', 'precisions', 'recalls', 'corlocs',
        'mean_corloc'
    ])


class ObjectDetectionEvaluation(object):

  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_iou_threshold=1.0,
               nms_max_output_boxes=10000,
               use_weighted_mean_ap=False,
               label_id_offset=0,
               group_of_weight=0.0):
    if num_groundtruth_classes < 1:
      raise ValueError('Need at least 1 groundtruth class for evaluation.')

    self.per_image_eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes=num_groundtruth_classes,
        matching_iou_threshold=matching_iou_threshold,
        nms_iou_threshold=nms_iou_threshold,
        nms_max_output_boxes=nms_max_output_boxes,
        group_of_weight=group_of_weight)
    self.group_of_weight = group_of_weight
    self.num_class = num_groundtruth_classes
    self.use_weighted_mean_ap = use_weighted_mean_ap
    self.label_id_offset = label_id_offset

    self.groundtruth_boxes = {}
    self.groundtruth_class_labels = {}
    self.groundtruth_masks = {}
    self.groundtruth_is_difficult_list = {}
    self.groundtruth_is_group_of_list = {}
    self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=float)
    self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

    self._initialize_detections()

  def _initialize_detections(self):
    self.detection_keys = set()
    self.scores_per_class = [[] for _ in range(self.num_class)]
    self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
    self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
    self.average_precision_per_class = np.empty(self.num_class, dtype=float)
    self.average_precision_per_class.fill(np.nan)
    self.precisions_per_class = []
    self.recalls_per_class = []
    self.corloc_per_class = np.ones(self.num_class, dtype=float)

  def clear_detections(self):
    self._initialize_detections()

  def add_single_ground_truth_image_info(self,
                                         image_key,
                                         groundtruth_boxes,
                                         groundtruth_class_labels,
                                         groundtruth_is_difficult_list=None,
                                         groundtruth_is_group_of_list=None,
                                         groundtruth_masks=None):
    if image_key in self.groundtruth_boxes:
      logging.warn(
          'image %s has already been added to the ground truth database.',
          image_key)
      return

    self.groundtruth_boxes[image_key] = groundtruth_boxes
    self.groundtruth_class_labels[image_key] = groundtruth_class_labels
    self.groundtruth_masks[image_key] = groundtruth_masks
    if groundtruth_is_difficult_list is None:
      num_boxes = groundtruth_boxes.shape[0]
      groundtruth_is_difficult_list = np.zeros(num_boxes, dtype=bool)
    self.groundtruth_is_difficult_list[
        image_key] = groundtruth_is_difficult_list.astype(dtype=bool)
    if groundtruth_is_group_of_list is None:
      num_boxes = groundtruth_boxes.shape[0]
      groundtruth_is_group_of_list = np.zeros(num_boxes, dtype=bool)
    self.groundtruth_is_group_of_list[
        image_key] = groundtruth_is_group_of_list.astype(dtype=bool)

    self._update_ground_truth_statistics(
        groundtruth_class_labels,
        groundtruth_is_difficult_list.astype(dtype=bool),
        groundtruth_is_group_of_list.astype(dtype=bool))

  def add_single_detected_image_info(self, image_key, detected_boxes,
                                     detected_scores, detected_class_labels,
                                     detected_masks=None):
   
    if (len(detected_boxes) != len(detected_scores) or
        len(detected_boxes) != len(detected_class_labels)):
      raise ValueError('detected_boxes, detected_scores and '
                       'detected_class_labels should all have same lengths. Got'
                       '[%d, %d, %d]' % len(detected_boxes),
                       len(detected_scores), len(detected_class_labels))

    if image_key in self.detection_keys:
      logging.warn(
          'image %s has already been added to the detection result database',
          image_key)
      return

    self.detection_keys.add(image_key)
    if image_key in self.groundtruth_boxes:
      groundtruth_boxes = self.groundtruth_boxes[image_key]
      groundtruth_class_labels = self.groundtruth_class_labels[image_key]
      # Masks are popped instead of look up. The reason is that we do not want
      # to keep all masks in memory which can cause memory overflow.
      groundtruth_masks = self.groundtruth_masks.pop(
          image_key)
      groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[
          image_key]
      groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[
          image_key]
    else:
      groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
      groundtruth_class_labels = np.array([], dtype=int)
      if detected_masks is None:
        groundtruth_masks = None
      else:
        groundtruth_masks = np.empty(shape=[0, 1, 1], dtype=float)
      groundtruth_is_difficult_list = np.array([], dtype=bool)
      groundtruth_is_group_of_list = np.array([], dtype=bool)
    scores, tp_fp_labels, is_class_correctly_detected_in_image = (
        self.per_image_eval.compute_object_detection_metrics(
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detected_class_labels,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_class_labels=groundtruth_class_labels,
            groundtruth_is_difficult_list=groundtruth_is_difficult_list,
            groundtruth_is_group_of_list=groundtruth_is_group_of_list,
            detected_masks=detected_masks,
            groundtruth_masks=groundtruth_masks))

    for i in range(self.num_class):
      if scores[i].shape[0] > 0:
        self.scores_per_class[i].append(scores[i])
        self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
    (self.num_images_correctly_detected_per_class
    ) += is_class_correctly_detected_in_image

  def _update_ground_truth_statistics(self, groundtruth_class_labels,
                                      groundtruth_is_difficult_list,
                                      groundtruth_is_group_of_list):

    for class_index in range(self.num_class):
      num_gt_instances = np.sum(groundtruth_class_labels[
          ~groundtruth_is_difficult_list
          & ~groundtruth_is_group_of_list] == class_index)
      num_groupof_gt_instances = self.group_of_weight * np.sum(
          groundtruth_class_labels[groundtruth_is_group_of_list] == class_index)
      self.num_gt_instances_per_class[
          class_index] += num_gt_instances + num_groupof_gt_instances
      if np.any(groundtruth_class_labels == class_index):
        self.num_gt_imgs_per_class[class_index] += 1

  def evaluate(self):
    if (self.num_gt_instances_per_class == 0).any():
      logging.warn(
          'The following classes have no ground truth examples: %s',
          np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0)) +
          self.label_id_offset)

    if self.use_weighted_mean_ap:
      all_scores = np.array([], dtype=float)
      all_tp_fp_labels = np.array([], dtype=bool)
    for class_index in range(self.num_class):
      if self.num_gt_instances_per_class[class_index] == 0:
        continue
      if not self.scores_per_class[class_index]:
        scores = np.array([], dtype=float)
        tp_fp_labels = np.array([], dtype=float)
      else:
        scores = np.concatenate(self.scores_per_class[class_index])
        tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
      if self.use_weighted_mean_ap:
        all_scores = np.append(all_scores, scores)
        all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
      precision, recall = metrics.compute_precision_recall(
          scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])
      self.precisions_per_class.append(precision)
      self.recalls_per_class.append(recall)
      average_precision = metrics.compute_average_precision(precision, recall)
      self.average_precision_per_class[class_index] = average_precision

    self.corloc_per_class = metrics.compute_cor_loc(
        self.num_gt_imgs_per_class,
        self.num_images_correctly_detected_per_class)

    if self.use_weighted_mean_ap:
      num_gt_instances = np.sum(self.num_gt_instances_per_class)
      precision, recall = metrics.compute_precision_recall(
          all_scores, all_tp_fp_labels, num_gt_instances)
      mean_ap = metrics.compute_average_precision(precision, recall)
    else:
      mean_ap = np.nanmean(self.average_precision_per_class)
    mean_corloc = np.nanmean(self.corloc_per_class)
    return ObjectDetectionEvalMetrics(
        self.average_precision_per_class, mean_ap, self.precisions_per_class,
        self.recalls_per_class, self.corloc_per_class, mean_corloc)
