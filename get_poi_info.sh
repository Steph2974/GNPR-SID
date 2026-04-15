#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-NYC}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./datasets}"

usage() {
  echo "用法: $0 [选项]"
  echo ""
  echo "依次执行: dataprocess_v2_1（生成 poi_info 等）→ get_geo_emb（写入 base_emb/geo_emb）。"
  echo ""
  echo "选项:"
  echo "  --dataset      <str>   数据集名称，如 NYC / TKY / CA (默认: $DATASET)"
  echo "  --output-root  <path>  与 dataprocess 一致，产物在 {output_root}/{dataset}/ (默认: $OUTPUT_ROOT)"
  echo "  -h, --help             显示此帮助"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)     DATASET="$2";     shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "未知参数: $1" >&2; usage ;;
  esac
done

POI_CSV="${OUTPUT_ROOT%/}/${DATASET}/poi_info.csv"

echo "项目目录: $ROOT"
echo "配置: dataset=$DATASET | output_root=$OUTPUT_ROOT"
echo ""

echo "[1/2] 运行 dataprocess_v2_1.py ..."
"$PYTHON" scripts/dataprocess_v2_1.py --dataset "$DATASET" --output-root "$OUTPUT_ROOT"

echo ""
echo "[2/2] 运行 get_geo_emb.py（需本地 SentenceTransformer 模型）..."
"$PYTHON" scripts/get_geo_emb.py --dataset "$DATASET" --output-root "$OUTPUT_ROOT"

echo ""
echo "完成: $POI_CSV（已包含 base_emb / geo_emb 列）"


# python get_poi_info.sh --dataset NYC --output-root ./datasets/