num_gpus=$1
config_path=$2

torchrun --nproc_per_node=$num_gpus \
    --nnodes=1 \
    --node_rank=0 \
    main.py \
    --config $config_path \