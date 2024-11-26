

for s in {0,10,20,30,40}
do
    torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8  --master_port 16274 ./Gram/main_finetune.py \
    -R TUEV_fineune_base \
    -CP ./config/TUEV_finetune.yaml \
    --if_DDP \
    --if_finetune \
    --load_model_path ./result/checkpoints/base.pth \
    --seed ${s} \
    
done

