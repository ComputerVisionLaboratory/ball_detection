#hide
get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")
get_ipython().run_line_magic("load_ext", " tensorboard")


from rich import print


import os
os.chdir('../../')
os.getcwd()


from sklearn.preprocessing import LabelEncoder

class_labels = ['soccer_ball', 'background']

le = LabelEncoder()
le.fit(class_labels)

class_map = dict(zip(le.transform(le.classes_), le.classes_))
print(class_map)


import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torchvision.transforms import Compose
from ball_detection.dataset import XMLDetectionDataModule
data_dir = '../data/'

transform = A.Compose([
        A.Flip(p=0.5),
        A.ChannelShuffle(p=0.2),
        A.Blur(p=0.2),
        A.RandomBrightness(p=0.2),
        A.pytorch.transforms.ToTensor(),
    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
)

target_transform = Compose([
    lambda x : dict(x, **{'boxes' : torch.Tensor(x['bboxes']), 'labels': torch.LongTensor(le.transform(x['class_labels']))}),
])

dm = XMLDetectionDataModule(data_dir, transform=transform, target_transform=target_transform, batch_size=4)
dm.setup(mode='use_dir')


import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from ball_detection.cnn_module import TorchVisionDetector

# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# backbone.out_channels = 1280

# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))

# model = FasterRCNN(backbone,
#                    num_classes=2,
#                    rpn_anchor_generator=anchor_generator)

#                    box_roi_pool=roi_pooler)



# num_classes = 2
# in_features = module.model.roi_heads.box_predictor.cls_score.in_features
# module.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

module=TorchVisionDetector(model)


images, targets = next(iter(dm.train_dataloader()))


# Eval mode
module.eval()
eval_preds = module(images)
print(f"Batchsize: {len(eval_preds)}, output dict keys {eval_preds[0].keys()}, shape of bbox outputs: {eval_preds[0]['boxes'].shape}")


import numpy as np
from ball_detection.metrics import hungarian_loss, bbox_iou

iou = []
for pred, target in zip(eval_preds, targets):
    iou.append(hungarian_loss(pred['boxes'].detach().numpy(), target['bboxes'], bbox_iou))
print(np.mean(iou))


# Train mode
module.train()
train_preds = module(images, targets)
print(train_preds)


import pytorch_lightning as pl

logger = pl.loggers.TensorBoardLogger('tb_logs', name='faster-rcnn-resnet')
trainer = pl.Trainer(gpus=1, checkpoint_callback=False, auto_lr_find=True, logger=logger)


# debug_trainer = pl.Trainer(fast_dev_run=2)
# debug_trainer.fit(module, dm)


# # Run learning rate finder
# lr_finder = trainer.tuner.lr_find(module, dm, max_lr=0.01)

# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()

# new_lr = lr_finder.suggestion()
# print("new lr", new_lr)


trainer.max_epochs = 10 # warm y
module.lr = 0.0005
trainer.fit(module, dm)


module.model.


# # import torch
# # import torchvision

# # # Load a pre-trained version of MobileNetV2
# torch_model = module.model

# scripted_model = torch.jit.script(torch_model)

# import coremltools
# mlmodel = coremltools.converters.convert(
#   scripted_model,
#   inputs=[coremltools.TensorType(shape=(1, 3, 64, 64))],
# )


# mlmodel.save("faster_rcnn_resnet.mlmodel")


# import numpy as np
# import torch
# import torchvision.models.detection as models
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor
# from contexttimer import Timer
# import numpy as np
# import albumentations as A
# import cv2
# import warnings
# from detection_nbdev.utils import visualize
# from fastai.vision.data import get_grid
# import pytorch_lightning as pl
# from src.dataset import XMLDetectionDataModule, XMLDetectionDataset
# from src.model import TorchVisionDetector
# from src.metrics import hungarian_loss, bbox_iou
