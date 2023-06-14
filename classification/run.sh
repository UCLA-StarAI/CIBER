for RUN in 1 2 3
do
    python3 collect_weights.py --dir=ckpts/vgg16_cifar10_run$RUN --dataset=CIFAR10 --model=VGG16 --seed=$RUN --gpu=1 \
    --collapsed="165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 250 255 260 265 270 275 280 285 290 295 300"

    python3 collect_logits.py --dir=ckpts/vgg16_cifar10_run$RUN --dataset=CIFAR10 --data_path=data  --use_test --model=VGG16 --seed=$RUN --gpu=1 --collapsed="165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 250 255 260 265 270 275 280 285 290 295 300"

    for COLLAPSED in 165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 250 255 260 265 270 275 280 285 290 295 300
    do
        julia -t auto solve_wmi.jl --dir ckpts/vgg16_cifar10_run$RUN --i $COLLAPSED
    done
done

for RUN in 1 2 3
do
    python3 wmi_eval.py --dir=ckpts/vgg16_cifar10_run$RUN --dataset=CIFAR10 --data_path=data --use_test --model=VGG16 --gpu=1 --all_collapsed 165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 250 255 260 265 270 275 280 285 290 295 300
done