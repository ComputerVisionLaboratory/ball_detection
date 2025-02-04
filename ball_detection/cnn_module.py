# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_cnn_module.ipynb (unless otherwise specified).

__all__ = ['bbox_iou', 'SingleBoxDetector', 'COCO_INSTANCE_CATEGORY_NAMES', 'name2idx', 'sports_ball_ID',
           'TorchVisionDetector']

# Cell
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision.models import detection as models

import pytorch_lightning as pl
from .metrics import bbox_iou, hungarian_loss

from torch_optimizer import AdaBelief

# Cell
def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

# Cell
class SingleBoxDetector(pl.LightningModule):
    def __init__(self, model=None, pretrained=False, freeze_extractor=False, log_level=10, num_classes=None, weight_path=None):
        super().__init__()
        self.pretrained = pretrained
        self.freeze_extractor = freeze_extractor
        self.lr = 0.001
        self.batch_size = 2

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        outputs = self.calculate_metrics(y_hat=y_hat, y=y)
        return outputs

    def training_epoch_end(self, outputs):
        avg_metrics = {}
        for metric in outputs[0].keys():
            val = torch.stack([x[metric] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f"{metric}/train", val, self.current_epoch)
            avg_metrics[metric] = val

#         epoch_dictionary = {'loss': avg_metrics['loss']}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        outputs = self.calculate_metrics(y_hat=y_hat, y=y)
        return outputs

    def validation_epoch_end(self, outputs):
        avg_metrics = {}
        for metric in outputs[0].keys():
            val = torch.stack([x[metric] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f"{metric}/validation", val, self.current_epoch)
            avg_metrics[metric] = val

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        outputs = self.calculate_metrics(y_hat=y_hat, y=y)
        return outputs

    def configure_optimizers(self):
        optimizer = AdaBelief(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer)
        return {
       'optimizer': optimizer,
       'lr_scheduler': scheduler,
       'monitor': 'iou_loss'
   }
#     >    return torch.optim.SGF(self.parameters(), lr=self.lr, aldsfk'a)

    def calculate_metrics(self, y, y_hat):
        iou = hungarian_loss(y, y_hat[0], bbox_iou, True)
        return {
            "iou": iou
        }

    def on_sanity_check_start(self):
        self.logger.disable()

    def on_sanity_check_end(self):
        self.logger.enable()

# Cell
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

name2idx = {i:COCO_INSTANCE_CATEGORY_NAMES[i] for i in range(len(COCO_INSTANCE_CATEGORY_NAMES))}
sports_ball_ID = COCO_INSTANCE_CATEGORY_NAMES.index('sports ball')

# Cell
class TorchVisionDetector(SingleBoxDetector):
    def __init__(self, model=None, pretrained=False, freeze_extractor=False, log_level=10, num_classes=None, weight_path=None):
        super().__init__()

#         available_models = ['maskrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn']
#         assert model in available_models, "Model most be from ['maskrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn']"
        self.model = model

    def forward(self, images, targets=None):
        predictions = self.model(images, targets)
        # filter out predictions of sports ball
#         func = lambda x : x[1] == sports_ball_ID
#         ball_predictions = [list(filter(func, zip(*pred.values()))) for pred in predictions]
#         ball_bboxes = [[p[0] for p in batch] for batch in ball_predictions]
        return predictions

    def training_step(self, batch, batch_idx):
#         if epoch == 0:
#             warmup_factor = 1. / 1000
#             warmup_iters = min(1000, len(data_loader) - 1)

#             lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        images, targets = batch
        loss_dict = self(images, targets)
        losses = sum(loss if not loss.isnan() else 0  for loss in loss_dict.values() ) / len(loss_dict.values())
        iou_loss = 0 # hungarian_loss
        total_loss = dict(loss_dict, **{"loss": losses, "iou": iou_loss})
        return total_loss

    def training_epoch_end(self, outputs):
        d = {'epoch':self.current_epoch}
        for metric in outputs[0].keys():
            try:
                val = torch.stack([x[metric] for x in outputs]).mean()
                self.logger.experiment.add_scalar(
                    f"{metric}/train", val, self.current_epoch
                )
                d[f"{metric}/train"] = val
            except:
                pass
#         print(d)
        pass

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        def filter_ball_bboxes(X):
            func = lambda x : x[1] == 1
            ball_predictions = [list(filter(func, zip(*pred.values()))) for pred in X]
            ball_bboxes = [[p[0] for p in batch] for batch in ball_predictions]
            return ball_bboxes

        pred_bboxes = filter_ball_bboxes([{key : d[key].cpu().detach().numpy() for key in ['boxes', 'labels']} for d in preds])
        targ_bboxes = filter_ball_bboxes([{key : d[key].cpu().detach().numpy() for key in ['boxes', 'labels']} for d in targets])
#         print(pred_bboxes)
#         print(targ_bboxes)
        iou_loss = sum([hungarian_loss(A,B, bbox_iou) for A, B in zip(pred_bboxes, targ_bboxes)]) / len(preds)
        self.log('iou_loss', iou_loss)
        return iou_loss

    def validation_epoch_end(self, outputs):
        score = sum(outputs)/len(outputs)
        self.logger.experiment.add_scalar(
                f"iou/validation", score, self.current_epoch
            )
        print('epoch',  self.current_epoch, f"iou/validation", score)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        def filter_ball_bboxes(X):
            func = lambda x : x[1] == 1
            ball_predictions = [list(filter(func, zip(*pred.values()))) for pred in X]
            ball_bboxes = [[p[0] for p in batch] for batch in ball_predictions]
            return ball_bboxes

        pred_bboxes = filter_ball_bboxes([{key : d[key].cpu().detach().numpy() for key in ['boxes', 'labels']} for d in preds])
        targ_bboxes = filter_ball_bboxes([{key : d[key].cpu().detach().numpy() for key in ['boxes', 'labels']} for d in targets])

        iou_loss = sum([hungarian_loss(A,B, bbox_iou) for A, B in zip(pred_bboxes, targ_bboxes)]) / len(preds)
        self.log('iou_loss', iou_loss)
        return iou_loss
