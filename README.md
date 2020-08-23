# RetinaNet

### Models
| Backbone | Input Shape | COCO val2017 mAP | Link | Training Time on TPU v3-8 |
| --- | --- | --- | --- |  --- |
| **ResNet50** | 640x640 | 37.1 | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-640-3x-64-tpu.json) | ~3h |
| **ResNet50** | 1024x1024 | ... | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-1026-3x-64-tpu.json) | ~7h |
| **ResNet50** | 1280x1280 | ... | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-1280-3x-64-tpu.json) | ~12h |

 - *All models use imagenet pretrained backhone*
 - *All models adopt the 3x training schedule, where 1x schedule is ~12 epochs. The learning rate schedule is adjusted accordingly*

```
@misc{1708.02002,
Author = {Tsung-Yi Lin and Priya Goyal and Ross Girshick and Kaiming He and Piotr Dollár},
Title = {Focal Loss for Dense Object Detection},
Year = {2017},
Eprint = {arXiv:1708.02002},
}
```
