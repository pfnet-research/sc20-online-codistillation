# online-distillation-imagenet

This is an artifact of a SC20 paper about online codistillation for ImageNet training.

## Dataset preparation

The original archive of Tiny ImageNet dataset can be downloaded from [Tiny ImageNet Visual Recognition Challenge](https://tiny-imagenet.herokuapp.com/).
To make the dataset readable from our script, you can use scripts in the `dataset-prep` directory.

## Training

The following command launches the training script:
```
./mpi.bash <num workers> <options>
```

- Options (except dataset path) for 90 epoch training with synchronized online codistillation:
```
--no_output_model=t --tiny_imagenet=t --lr '{"initial_lr": 0.1, "cosine": true, "warmup_epoch": 5 }' --epoch 90 --seed $RANDOM --batchsize 64 --model resnet18v2 --weight_decay 5e-4 --distillation=t --burnin_epoch 36 --equalize_data=t
```
