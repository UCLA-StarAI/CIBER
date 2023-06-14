# Image Classification

## Training

To collect weights from SGD trajectories and compare with baseline SWAG, we need to run the training of the neural network models (code adapted from https://github.com/wjmaddox)

```bash
python3 training.py --data_path=<PATH> --epochs=<EPOCHS> --dataset=<DATASET> --save_freq=<SAVE_FREQ> \
--model=<MODEL> --lr_init=<LR_INIT> --wd=<WD> --swag --swag_start=<SWAG_START> --swag_lr=<SWAG_LR> --cov_mat --use_test \
--dir=<DIR> --seed=<SEED>
```

Our training script for reproducing experiments in the paper is as below.
```bash
# VGG16
python3 swag.py --data_path=data --epochs=300 --dataset=[CIFAR10 or CIFAR100] --save_freq=5 \
--model=VGG16 --lr_init=0.05 --wd=5e-4 --swag --swag_start=161 --swag_lr=0.01 --cov_mat --use_test --dir=<DIR> --seed=<SEED>

# PreResNet164
python3 swag.py --data_path=data --epochs=300 --dataset=CIFAR10 --save_freq=5 \
--model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swag --swag_start=161 --swag_lr=0.01 --cov_mat --use_test --dir=<DIR> --seed=<SEED>

python3 swag.py --data_path=data --epochs=300 --dataset=CIFAR100 --save_freq=5 \
--model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swag --swag_start=161 --swag_lr=0.05 --cov_mat --use_test --dir=<DIR> --seed=<SEED>

# WideResNet28x10
python3 swag.py --data_path=data --epochs=300 --dataset=[CIFAR10 or CIFAR100]  --save_freq=5 \
--model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swag --swag_start=161 --swag_lr=0.05 --cov_mat --use_test --dir=<DIR> --seed=<SEED>

python3 swag.py --data_path=data --epochs=300 --dataset=CIFAR100 --save_freq=5 \
--model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swag --swag_start=161 --swag_lr=0.05 --cov_mat --use_test \
--dir=ckpts/wide_cifar100_run1 --seed=1 --num_workers=0 --gpu=0
```

## Collapsed Inference
With weights collected from the SGD trajectories, we are ready to run collapsed inference by first collecting the collapsed set and the corresponding logits, and then solving the resulting WMI problems to obtain the final predictions. `run.sh` shows an example script for reproducing the results on dataset CIFAR-10 with model VGG-16.
