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


GT_PATH = 'FOO'
PRED_PATH = 'BAR'

pred = [1, 1, 4, 5]
gt = [2, 2, 5, 6]
print(intersection_over_union(pred, gt))
