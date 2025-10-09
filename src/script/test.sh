# AIM-500 AM-2K P3M-500-NP RefMatte_RW_100

SETNAME="AM-2K"
CONFIG_DIR="configs/SDMatte.py"
CKPT_DIR="SDMatte/SDMatte.pth"
INFER_DIR="infer_output/SDMatte/bbox/${SETNAME}"
RESULT_PATH="${INFER_DIR}_result.txt"

# # 使用变量执行命令
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config-dir "$CONFIG_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --inference-dir "$INFER_DIR" \
    --setname "$SETNAME"

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
    --pred-dir "$INFER_DIR" \
    --setname "$SETNAME" \
    --result-path "$RESULT_PATH"
