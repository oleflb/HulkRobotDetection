from ..dataloader.lightningdataset import DataModule
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torchvision.ops import box_iou
import sklearn.cluster
from tqdm import tqdm
import matplotlib.pyplot as plt

def iou(bboxes, centroid):
    return box_iou(bboxes, centroid)

class IOUKMeans(sklearn.cluster.KMeans):
    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm
        )

    def _transform(self, X):
        print(X)
        # return anchor_iou(X, self.cluster_centers_)

def main(args):
    dataloader = DataModule((480, 640), num_workers=8, batch_size=1)
    dataloader.setup("fit")
    dataset = dataloader.train_dataloader()

    ws = []
    hs = []

    for _, labels in tqdm(dataset):
        bboxes = labels[0]["boxes"]
        for bbox in bboxes:
            [min_x, min_y, max_x, max_y] = bbox
            ws.append(max_x - min_x)
            hs.append(max_y - min_y)

    ws = np.array(ws)
    hs = np.array(hs)
    plt.scatter(ws / hs, np.zeros_like(ws))
    plt.show()
    
    plt.scatter(ws, hs)
    plt.show()

    plt.scatter(np.log(ws), np.log(hs))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_k", help="maximum k to check", default=16)
    main(parser.parse_args())
