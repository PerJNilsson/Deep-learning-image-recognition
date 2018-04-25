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


def removeFromMap(filename, index_to_remove):
    old_sign_list = gt_map[filename]
    if len(old_sign_list) == 1:
        del gt_map[filename]
    else:
        new_sign_list = []
        for i in range(0, len(old_sign_list)):
            if i != index_to_remove:
                new_sign_list.append(old_sign_list[i])
        gt_map[file] = new_sign_list


GT_PATH = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/' \
          'data/TestGTSDB/gt.txt'
RESULT_PATH = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/' \
              'data/results/cloud/combined_test/result.csv'

gt_map = dict()

gt = pd.read_csv(GT_PATH, sep = ';', header = None)
result = pd.read_csv(RESULT_PATH, sep = ';', header = None)
false_positive = np.zeros(43, dtype=int)
positive = np.zeros(43, dtype=int)

for i in range(0, len(gt)):
    file = gt[0][i]
    bbox = [gt[1][i], gt[2][i], gt[3][i], gt[4][i]]
    sign_class = gt[5][i]
    if file not in gt_map:
        gt_map[file] = [[bbox, sign_class]]
    else:
        gt_map[file].append([bbox, sign_class])

for i in range(0, len(result)):
    file = result[0][i]
    bbox = [result[1][i], result[2][i], result[3][i], result[4][i]]
    sign_class = result[5][i]
    if file in gt_map:
        gt_signs = gt_map[file]
        for j in range(0, len(gt_signs)):
            iou = intersection_over_union(bbox, gt_signs[j][0])
            if iou > 0.5 and sign_class == gt_signs[j][1]:
                positive[sign_class] = positive[sign_class] + 1
                removeFromMap(file, j)
                break
            if j == len(gt_signs)-1:
                false_positive[sign_class] = false_positive[sign_class] + 1
    else:
        false_positive[sign_class] = false_positive[sign_class] + 1

all_pred = false_positive + positive
sum_AP = 0
classes_rep = 0
for i in range(0, len(all_pred)):
    if all_pred[i] != 0:
        sum_AP += positive[i]/all_pred[i]
        classes_rep = classes_rep + 1
    else:
        print('Class ' + str(i) + ' not represented.')

mAP = sum_AP/classes_rep
print(mAP)
