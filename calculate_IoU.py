def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

if __name__ == '__main__':
    clist = [[15, 186, 1091, 266]]
    plist = [[1172, 13, 1206, 99], [1145, 34, 1193, 155], [1203, 71, 1259, 211], [82, 74, 148, 238],
             [281, 63, 345, 224]]

    for p in plist:
        iou = bb_intersection_over_union(clist[0], p)
        print(iou)