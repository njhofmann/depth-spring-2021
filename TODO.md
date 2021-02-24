1. set up dataloader pipelines for sun rgb-d data
   1. what are the bounding boxes of?
   1. input size: 425×560×3 (560, 425, 3)?   
   1. set up data augmentation pipes
      1. normalization on per layer...?
      1. get them to work with bounding boxes and segs
      1. random scaling, jittering, and cropping
1. set up semseg pipeline
   1. apply transforms to dataset...?
   1. setup dataloader
   1. define optimizer and loss func
   1. train model
   1. eval model
   1. save model?
1. look at depth operators in detail
    1. easily interchangeable with conv2d?
    1. max extension?
    1. 
1. baseline experiments for semantic segmentation and object recognition
1. get additional datasets set up...?
1. setup convolutional operators
1. convolutional experiments