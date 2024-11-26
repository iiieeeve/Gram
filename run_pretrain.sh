
torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8  --master_port 1543 ./Gram/main_pretrain_base_class_quantization.py \
    -R pretrain_base_class_quantization \
    -CP ./config/pretrain_base_class_quantization.yaml \
    --if_DDP \


torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8  --master_port 1543 ./Gram/main_pretrain.py \
    -R pretrain_base \
    -CP ./config/pretrain_base.yaml \
    --if_DDP \

