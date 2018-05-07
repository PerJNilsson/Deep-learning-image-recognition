import pandas as pd
import numpy as np

def intersection_over_union(pred, gt):
    # Coordinates of intersecting rectangle
    xA = max(pred[0], gt[0])
    yA = max(pred[1], gt[1])
    xB = min(pred[2], gt[2])
    yB = min(pred[3], gt[3])

    interArea = (xB - xA) * (yB - yA)

    predArea = (pred[2] - pred[0]) * (pred[3] - pred[1])
    gtArea = (gt[2] - gt[0]) * (gt[3] - gt[1])

    iou = interArea / (predArea + gtArea - interArea)
    return iou


def removeFromMap(filename, index_to_remove, gt_map):
    old_sign_list = gt_map[filename]
    if len(old_sign_list) == 1:
        del gt_map[filename]
    else:
        new_sign_list = []
        for i in range(0, len(old_sign_list)):
            if i != index_to_remove:
                new_sign_list.append(old_sign_list[i])
        gt_map[filename] = new_sign_list

def correct_pred(pred, gt_map): #TODO - would be nice to compute number of wrong classifications
    false_positive = np.zeros(43, dtype=int)
    true_positive = np.zeros(43, dtype=int)
    wrong_class = np.zeros(43, dtype=int)
    for i in range(0, len(pred[1])):
        file = pred[0][i]
        bbox = [pred[1][i], pred[2][i], pred[3][i], pred[4][i]]
        sign_class = pred[5][i]
        if file in gt_map:
            gt_signs = gt_map[file]
            for j in range(0, len(gt_signs)):
                iou = intersection_over_union(bbox, gt_signs[j][0])
                if iou > 0.5 and sign_class == gt_signs[j][1]:
                    true_positive[sign_class] = true_positive[sign_class] + 1
                    removeFromMap(file, j, gt_map)
                    break
                if j == len(gt_signs)-1:
                    false_positive[sign_class] = false_positive[sign_class] + 1
        else:
            false_positive[sign_class] = false_positive[sign_class] + 1
    return true_positive, false_positive


def gt_dict(gt):
    gt_dict = dict()
    for i in range(0, len(gt)):
        file = gt[0][i]
        bbox = [gt[1][i], gt[2][i], gt[3][i], gt[4][i]]
        sign_class = gt[5][i]
        if file not in gt_dict:
            gt_dict[file] = [[bbox, sign_class]]
        else:
            gt_dict[file].append([bbox, sign_class])
    return gt_dict



GT_PATH = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/' \
          'data/TestGTSDB/gt.txt'
RESULT_PATH = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/' \
               'detection_api/data/results/cloud_ext/combined_test/result.csv'

all_thresholds = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  0.95, 0.99, 0.999, 0.9999]
number_of_gt_signs = 361 # look in GT-file

ground_truth = pd.read_csv(GT_PATH, sep = ';', header = None)
all_result = pd.read_csv(RESULT_PATH, sep = ';', header = None)

for threshold in all_thresholds:
    result = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6])
    for i in range(0, len(all_result)):
        if all_result[6][i] > threshold:
            tmp = pd.DataFrame([[all_result[0][i], all_result[1][i], all_result[2][i], all_result[3][i],
                           all_result[4][i], all_result[5][i], all_result[7][i]]],
                               columns=[0, 1, 2, 3, 4, 5, 6])
            result = result.append(tmp, ignore_index=True)

    pred_rcnn = [result[0], result[1], result[2], result[3], result[4], result[5]]
    pred_combined = [result[0], result[1], result[2], result[3], result[4], result[6]]

    rcnn_true, rcnn_false = correct_pred(pred_rcnn, gt_map=gt_dict(ground_truth))
    combined_true, combined_false = correct_pred(pred_combined, gt_map=gt_dict(ground_truth))



    precision_rcnn = sum(rcnn_true)/ (sum(rcnn_true) + sum(rcnn_false))
    recall_rcnn = sum(rcnn_true) / number_of_gt_signs

    precision_comb = sum(combined_true)/(sum(combined_false) + sum(combined_true))
    recall_comb = sum(combined_true) / number_of_gt_signs
    print('For threshold: ' + str(threshold))
    print('precision_rcnn: ' + str(precision_rcnn))
    print('recall_rcnn: ' + str(recall_rcnn))
    print('precision_comb: ' + str(precision_comb))
    print('recall_comb: ' + str(recall_comb))

    print(str(sum(rcnn_true) + sum(rcnn_false)))