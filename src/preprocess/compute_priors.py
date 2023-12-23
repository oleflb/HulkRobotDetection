from ..dataloader.lightningdataset import DataModule
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torchvision.ops import box_iou
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

def average_iou(bboxes, centroids):
    # bboxes: [-1, 2]
    # centroids: [n, 2]
    intersection_width = torch.minimum(centroids[:, [0]], bboxes[:, 0]).T
    intersection_height = torch.minimum(centroids[:, [1]], bboxes[:, 1]).T

    if torch.any(intersection_height == 0) or torch.any(intersection_width == 0):
        raise ValueError("Some bboxes have size 0")

    intersection_area = intersection_height * intersection_width
    boxes_area = torch.prod(bboxes, dim=1, keepdim=True)
    anchors_area = torch.prod(bboxes, dim=1, keepdim=True)
    union_area = boxes_area + anchors_area - intersection_area

    avg_iou = torch.mean(torch.max(intersection_area / union_area, dim=1).values)

    return avg_iou

def kmeans_aspect_ratios(bboxes, kmeans_max_iter, num_aspect_ratios):
    assert len(bboxes)
    assert bboxes.shape[1] == 2

    normalized_bboxes = bboxes / torch.sqrt(torch.prod(bboxes, dim=1, keepdim=True))
    kmeans = KMeans(init='random', n_init=10, n_clusters=num_aspect_ratios, random_state=0, max_iter=kmeans_max_iter)
    kmeans.fit(X=normalized_bboxes)

    ar = torch.from_numpy(kmeans.cluster_centers_)
    assert len(ar)

    avg_iou_perc = average_iou(normalized_bboxes, ar)

    aspect_ratios = [w/h for w,h in ar]

    return aspect_ratios, avg_iou_perc


def main(args):
    dataloader = DataModule((480, 640), num_workers=8, batch_size=1)
    dataloader.setup("fit")
    dataset = dataloader.train_dataloader()

    ws = []
    hs = []

    # for _, labels in tqdm(itertools.islice(dataset, 100)):
    for _, labels in tqdm(dataset):
        bboxes = labels[0]["boxes"]
        for bbox in bboxes:
            [min_x, min_y, max_x, max_y] = bbox
            ws.append(max_x - min_x)
            hs.append(max_y - min_y)

    bboxes = torch.tensor([ws, hs])
    aspect_ratios, mean_iou = kmeans_aspect_ratios(bboxes.T, 1000, 4)
    print(f"Final IOU: {100*mean_iou:.2f}%")
    print(aspect_ratios)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_k", help="maximum k to check", default=16)
    main(parser.parse_args())
