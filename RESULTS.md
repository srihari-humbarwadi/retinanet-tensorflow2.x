### Models
 - All models, unless stated are trained with square inputs.
 - All models are trained on coco2017 train split and evaluated on the coco2017 val split.
 - The RetinaNet paper trains the model for ~ 12.7 epochs on the coco2017, this is referred to as 1x schedule, the models listed below are trained for 1x, 3x or 30x schedules.
## ResNet50 640x640
<pre>
 - Backbone            : ResNet50 v1 (ImageNet pretrained weights)
 - Schedule            : 3x
 - Time required       : 50mins
 - System              : v3-32 TPU pod
 - config              : <a href="configs/v3-32/mscoco-retinanet-resnet50-640x640-3x-256.json">mscoco-retinanet-resnet50-640x640-3x-256</a>
 - weights             : <a href="#">coming soon </a>

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.377
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.570
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.710
</pre>

 <pre>
 - Backbone            : ResNet50 v1 (random weight initialization)
 - Schedule            : 30x
 - Time required       : 9h:30min
 - System              : v3-32 TPU pod
 - config              : <a href="configs/v3-32/mscoco-retinanet-resnet50-640x640-30x-256.json">mscoco-retinanet-resnet50-640x640-30x-256</a>
 - weights             : <a href="#">coming soon </a>

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.590
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.425
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.731
</pre>
___
