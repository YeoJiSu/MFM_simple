export CUDA_VISIBLE_DEVICES=4

CKPT_DIR=outputs          # train.py에서 --output_dir 안 줬으니 기본값
CKPT_PATH=${CKPT_DIR}/best.pt

python eval.py \
  --single_root /data3/jisu/MFM/datasets/VisA \
  --single_name visa \
  --ckpt_path "${CKPT_PATH}" \
  --batch_size 4 \
  --num_vision_tokens 4 \
  --max_length 128 \
  --output_dir "${CKPT_DIR}/eval"

# 평가하는 데에 총 3시간 걸렸음.