# UCI Regression Experiments

## Preparation

- Run `mkdir data log`.
- Download `uci.tar.gz` from [Google Drive](https://drive.google.com/file/d/0BxWe_IuTnMFcYXhxdUNwRHBKTlU/view) and put
it in the `data` folder.
- Run `cd data; tar -xzvf uci.tar.gz`.

## Running Experiments

An example script for running the regression problems with collapsed weights at the last layer is as follows
```bash
python3 training.py --dataset <DATASET> [--uci-small] --split <SPLIT> --epochs <EPOCHS> \
--lr_init <LR> --wd <WD> --batch_size <BATCH_SIZE> \
--log_dir <DIR> --weight_init <WEIGHT_INIT> --double-bias-lr \
--factor <FACTOR> --patience <PATIENCE> --seed <SEED>
python3 latte_parser.py --dir log/<DIR>/<DATASET>-<SEED>-<SPLIT>/wmi
julia -t auto solve_latte.jl --dir log/<DIR>/<DATASET>-<SEED>-<SPLIT>/wmi --e "<EPSILON>"
```

An example script for running the regression problems with collapsed weights at the second-to-last hidden layer is as follows
```bash
python3 training.py --dataset <DATASET> [--uci-small] --split <SPLIT> --epochs <EPOCHS> \
--lr_init <LR> --wd <WD> --batch_size <BATCH_SIZE> \
--log_dir <DIR> --weight_init <WEIGHT_INIT> --double-bias-lr \
--factor <FACTOR> --patience <PATIENCE> --hidden --seed <SEED>
python3 latte_parser_relu.py --dir log/<DIR>/<DATASET>-<SEED>-<SPLIT>/wmi
julia -t auto solve_latte.jl --dir log/<DIR>/<DATASET>-<SEED>-<SPLIT>/wmi --e "<EPSILON>" --relu
```

## Results

To view the results, run
```bash
python3 result_summary.py --dataset <DATASET> [--relu]
```