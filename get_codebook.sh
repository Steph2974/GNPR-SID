#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

# ---------- 默认配置 ----------
DATA_MODE="NYC"
VERSION="v2.0"
LAMBDA="0.50"
KMEANS_INIT="0"
DEVICE="cuda:3"
EPOCHS=300
EVAL_STEP=5
BATCH_SIZE=128
USE_GEO_EMB="0"
USE_CATNAME="1"
USE_REGION="1"
DATA_PATH_CLI=""
# ------------------------------

# ./get_codebook.sh --device cuda:0 --lambda 0.0
# ./get_codebook.sh --device cuda:1 --lambda 0.25
# ./get_codebook.sh --device cuda:2 --lambda 0.75


# ---------- 解析命令行参数（覆盖默认值）----------
usage() {
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --data_mode   <str>    数据集名称        (默认: $DATA_MODE)"
  echo "  --version     <str>    版本号            (默认: $VERSION)"
  echo "  --lambda      <float>  lambda 权重       (默认: $LAMBDA)"
  echo "  --kmeans_init <0|1>    是否用 kmeans 初始化 (默认: $KMEANS_INIT)"
  echo "  --device      <str>    训练设备          (默认: $DEVICE)"
  echo "  --epochs      <int>    训练轮数          (默认: $EPOCHS)"
  echo "  --eval_step   <int>    评估间隔          (默认: $EVAL_STEP)"
  echo "  --batch_size  <int>    batch 大小        (默认: $BATCH_SIZE)"
  echo "  --use_geo_emb <0|1>    是否拼接 geo_emb (默认: $USE_GEO_EMB；需 poi_info 含 geo_emb 列)"
  echo "  --data_path   <path>   poi_info.csv 路径 (默认: datasets/{data_mode}/poi_info.csv)"
  echo "  --use_catname <0|1>    是否使用 Catname one-hot (默认: $USE_CATNAME)"
  echo "  --use_region  <0|1>    是否使用 Region one-hot (默认: $USE_REGION)"
  echo "  -h, --help             显示此帮助"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_mode)   DATA_MODE="$2";   shift 2 ;;
    --version)     VERSION="$2";     shift 2 ;;
    --lambda)      LAMBDA="$2";      shift 2 ;;
    --kmeans_init) KMEANS_INIT="$2"; shift 2 ;;
    --device)      DEVICE="$2";      shift 2 ;;
    --epochs)      EPOCHS="$2";      shift 2 ;;
    --eval_step)   EVAL_STEP="$2";   shift 2 ;;
    --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
    --use_geo_emb) USE_GEO_EMB="$2"; shift 2 ;;
    --data_path)   DATA_PATH_CLI="$2"; shift 2 ;;
    --use_catname) USE_CATNAME="$2"; shift 2 ;;
    --use_region)  USE_REGION="$2";  shift 2 ;;
    -h|--help)     usage ;;
    *) echo "未知参数: $1" >&2; usage ;;
  esac
done
# -----------------------------------------------

if [[ -n "$DATA_PATH_CLI" ]]; then
  DATA_PATH="$DATA_PATH_CLI"
else
  DATA_PATH="datasets/${DATA_MODE}/poi_info.csv"
fi

COMMON_ARGS=(
  --data_mode "$DATA_MODE"
  --data_path "$DATA_PATH"
  --eval_step "$EVAL_STEP"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lamda "$LAMBDA"
  --kmeans_init "$KMEANS_INIT"
  --version "$VERSION"
  --device "$DEVICE"
  --use_geo_emb "$USE_GEO_EMB"
  --use_catname "$USE_CATNAME"
  --use_region "$USE_REGION"
)

echo "项目目录: $ROOT"
echo "配置: data_mode=$DATA_MODE | data_path=$DATA_PATH | version=$VERSION | lambda=$LAMBDA | device=$DEVICE | epochs=$EPOCHS | use_geo_emb=$USE_GEO_EMB | use_catname=$USE_CATNAME | use_region=$USE_REGION"
echo ""

# 0. 激活 conda 环境
# conda init
# conda activate xyl

# 1. 训练 RQ-VAE
echo "开始训练 RQ-VAE ..."
"$PYTHON" code/train_rqvae.py "${COMMON_ARGS[@]}"

# sleep 10 # 训练结束后再导出 codebook，可按需改秒数

# 2. 导出 codebook CSV
echo "导出 codebook CSV ..."
"$PYTHON" code/codebook.py "${COMMON_ARGS[@]}"

# 3. 保存 JSON
echo "开始保存 JSON ..."
"$PYTHON" save_json.py --target_dataset "$DATA_MODE" --version "$VERSION" --div_loss "$LAMBDA" --idorcodebook "codebook"