import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]


def build_geo_emb_for_poi_info(
    poi_info: pd.DataFrame,
    text_col: str = "Original_Catname",
    lat_col: str = "Lat",
    lon_col: str = "Lon",
    model_path: str = "./models/all-MiniLM-L6-v2",
    num_anchors: int = 8,
    normalize_embeddings: bool = True,
    random_state: int = 42,
):
    """
    根据 poi_info 中的 Original_Catname + Lat + Lon 生成 base_emb 和 geo_emb。

    参数
    ----
    poi_info : pd.DataFrame
        至少包含 [Pid, Original_Catname, Lat, Lon]
    text_col : str
        类别文本列
    lat_col : str
        纬度列
    lon_col : str
        经度列
    model_path : str
        本地 MiniLM 模型路径
    num_anchors : int
        GeoPE 中参考点数量
    normalize_embeddings : bool
        是否对 text encoder 输出归一化
    random_state : int
        随机种子

    返回
    ----
    poi_info_out : pd.DataFrame
        原表基础上新增:
        - base_emb
        - geo_emb
        - anchor_angles
    anchors : np.ndarray
        shape [num_anchors, 2], 每行为 [lon, lat]
    """

    required_cols = [text_col, lat_col, lon_col]
    for c in required_cols:
        if c not in poi_info.columns:
            raise ValueError(f"poi_info 缺少必要列: {c}")

    poi_info_out = poi_info.copy()

    # 只对有完整信息的行计算
    valid_mask = (
        poi_info_out[text_col].notna()
        & poi_info_out[lat_col].notna()
        & poi_info_out[lon_col].notna()
    )

    valid_df = poi_info_out.loc[valid_mask, [text_col, lat_col, lon_col]].copy()
    valid_df[text_col] = valid_df[text_col].astype(str)

    if len(valid_df) == 0:
        raise ValueError("没有可用于计算 geo_emb 的有效行。")

    # 1) 文本编码：Original_Catname -> base_emb
    model = SentenceTransformer(model_path)
    texts = valid_df[text_col].tolist()

    base_emb = model.encode(
        texts,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)   # [N, D]

    # 2) 以坐标做 KMeans，得到参考点 anchors
    coords = valid_df[[lon_col, lat_col]].to_numpy(dtype=np.float32)  # [lon, lat]
    n = len(coords)

    if n < num_anchors:
        num_anchors = max(1, n)

    km = KMeans(n_clusters=num_anchors, random_state=random_state, n_init=10)
    km.fit(coords)
    anchors = km.cluster_centers_.astype(np.float32)  # [num_anchors, 2] -> [lon, lat]

    # 3) 计算每个 POI 相对每个 anchor 的 bearing angle
    angles = np.zeros((n, num_anchors), dtype=np.float32)

    for i in range(n):
        lon, lat = coords[i]
        for j in range(num_anchors):
            ref_lon, ref_lat = anchors[j]

            delta_lon = lon - ref_lon
            delta_lat = lat - ref_lat

            # 按论文思路，使用方位角来决定旋转
            theta = math.atan2(
                delta_lat,
                delta_lon * math.cos(math.radians(float(ref_lat)))
            )
            if theta < 0:
                theta += 2 * math.pi

            angles[i, j] = theta

    # 4) 对 base_emb 分段，并按各自角度做二维旋转 -> geo_emb
    n, d = base_emb.shape
    splits = np.array_split(np.arange(d), num_anchors)
    geo_emb = np.zeros_like(base_emb, dtype=np.float32)

    for i in range(n):
        for j, dim_idx in enumerate(splits):
            seg = base_emb[i, dim_idx].copy()
            theta = float(angles[i, j])

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            rotated = seg.copy()
            pair_len = (len(seg) // 2) * 2

            # 相邻两维一组旋转
            for k in range(0, pair_len, 2):
                x0, x1 = seg[k], seg[k + 1]
                rotated[k] = cos_t * x0 - sin_t * x1
                rotated[k + 1] = sin_t * x0 + cos_t * x1

            geo_emb[i, dim_idx] = rotated

    # 5) 回填到原 DataFrame
    # 用 object 列 + 与 valid 行索引对齐的 Series，避免 pandas 把等长 list-of-list 当成 2D 赋值而报错
    valid_idx = poi_info_out.index[valid_mask]
    for col in ("base_emb", "geo_emb", "anchor_angles"):
        poi_info_out[col] = pd.Series([None] * len(poi_info_out), dtype=object, index=poi_info_out.index)

    poi_info_out.loc[valid_idx, "base_emb"] = pd.Series(
        [base_emb[i].tolist() for i in range(len(base_emb))], index=valid_idx, dtype=object
    )
    poi_info_out.loc[valid_idx, "geo_emb"] = pd.Series(
        [geo_emb[i].tolist() for i in range(len(geo_emb))], index=valid_idx, dtype=object
    )
    poi_info_out.loc[valid_idx, "anchor_angles"] = pd.Series(
        [angles[i].tolist() for i in range(len(angles))], index=valid_idx, dtype=object
    )

    return poi_info_out, anchors

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="为 poi_info.csv 生成 base_emb / geo_emb 并写回同一路径。")
    p.add_argument("--dataset", type=str, default="NYC", help="数据集名，与 dataprocess_v2_1 的 --dataset 一致")
    p.add_argument(
        "--output-root",
        type=str,
        default="datasets",
        help="与 dataprocess_v2_1 的 --output-root 一致，poi_info 位于 {output_root}/{dataset}/poi_info.csv",
    )
    p.add_argument("--model-path", type=str, default="./models/all-MiniLM-L6-v2")
    p.add_argument("--num-anchors", type=int, default=8)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    out_dir = Path(args.output_root) / args.dataset
    poi_path = out_dir / "poi_info.csv"

    poi_info = pd.read_csv(poi_path)

    poi_info, _anchors = build_geo_emb_for_poi_info(
        poi_info=poi_info,
        text_col="Original_Catname",
        lat_col="Lat",
        lon_col="Lon",
        model_path=args.model_path,
        num_anchors=args.num_anchors,
    )

    poi_info.to_csv(poi_path, index=False)
    print(f"已更新: {poi_path.resolve()}")