export CUDA_VISIBLE_DEVICES=4

# python train.py \
#     --single_root /data3/jisu/MFM/datasets/VisA \
#     --single_name visa \
#     --epochs 10 \
#     --batch_size 4 \
#     --grad_accum 4 \
#     --num_vision_tokens 4 \
#     --max_length 128 \
#     --lr 1e-4 \
#     --amp

python train.py \
    --single_root /data3/jisu/MFM/datasets/VisA \
    --single_name visa \
    --epochs 10 \
    --batch_size 4 \
    --grad_accum 4 \
    --num_vision_tokens 4 \
    --max_length 128 \
    --lr 1e-4 \
    --amp \
    --resume_from outputs/epoch_007.pt
# Epoch 하나에 30분 정도 -> 10 epoch 5시간 정도 걸리는 듯?