### Models
## ResNet50
| Input Shape | COCO val2017 mAP | Link | Training Time on TPU v3-8 |
| --- | --- | --- |  --- |
| **640x640** | 36.8 | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-640-3x-64-tpu.json) | ~3h |
| **1024x1024** | 40.2 | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-1026-3x-64-tpu.json) | ~6.5h |
| **1280x1280** | **40.5** | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-1280-3x-64-tpu.json) | ~10.5h |

 - *Above models use imagenet pretrained backbone.*
 - *The models are trained with the 3x training schedule, where 1x schedule is ~12 epochs. The learning rate schedule is adjusted accordingly.*
___
## ResNet34
| Input Shape | COCO val2017 mAP | Link | Training Time on TPU v3-8 |
| --- | --- | --- |  --- |
| **640x640** | *under training* | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-34-640-30x-64-tpu.json) | ~28h |
| **1024x1024** | *under training* | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-34-1024-30x-64-tpu.json) | ~68h |

## ResNet101
| Input Shape | COCO val2017 mAP | Link | Training Time on TPU v3-8 |
| --- | --- | --- |  --- |
| **640x640** | 40.2 | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-101-640-30x-64-tpu.json) | ~38h |
| **1024x1024** | *under training* | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-101-1024-30x-64-tpu.json) | ~96h |

 - *Above models use randomly initialized backbones.*
 - *The models are trained with the **30x** training schedule, where 1x schedule is ~12 epochs. The learning rate schedule is adjusted accordingly.*

## ResNet152
| Input Shape | COCO val2017 mAP | Link | Training Time on TPU v3-8 |
| --- | --- | --- |  --- |
| **640x640** | *under training* | [checkpoint](https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x) / [config](configs/retinanet-151-640-50x-64-tpu.json) | ~80h |

 - *Above model uses randomly initialized backbone.*
 - *The models are trained with the **50x** training schedule, where 1x schedule is ~12 epochs. The learning rate schedule is adjusted accordingly.*
___
