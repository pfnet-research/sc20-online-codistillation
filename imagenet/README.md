# online-distillation-imagenet

This is an artifact of a SC20 paper about online codistillation for ImageNet training.

## Dataset preparation

ImageNet dataset in the MXNet RecordIO format is required.
You can find the instruction of making the dataset in [MXNet's site](https://mxnet.apache.org/api/faq/recordio).

## Training

The following command launches the training script:
```
./mpi.bash <num workers> <options>
```

- Options (except dataset path) for 70 epoch training with synchronized online codistillation:
```
--lr '{"initial_lr": 2.82842712474619, "warmup_epoch": 0.078125, "legw_roller_coaster": true }' --fp16=t --loss_scaling=8 --seed $RANDOM --epoch 70 --batchsize 64 --label_smoothing 0.1 --lars_eta 0.001 --distillation=t --distillation_overlap=t --burnin_epoch 28
```
