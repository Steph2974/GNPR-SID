import os
import ast
import argparse
import pandas as pd
import pygeohash as pgh


def safe_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def codebook_to_tokens(codebook):
    """
    [22, 1, 30, 1] -> <a_22><b_1><c_30><d_1>
    """
    return ''.join([f"<{chr(97 + idx)}_{code}>" for idx, code in enumerate(codebook)])


def geohash_to_tokens(lat, lon, precision=6):
    """
    每个字符独立成一个 token，最大化前缀共享粒度。
    例如 dr5rjm -> <g_d><g_r><g_5><g_r><g_j><g_m>

    空间越近的 POI，共享的前缀 token 越多，LLM 可直接从 token 重叠感知空间关系。
    """
    gh = pgh.encode(float(lat), float(lon), precision=precision)
    if len(gh) != precision:
        gh = gh[:precision].ljust(precision, "0")
    return ''.join([f"<g_{c}>" for c in gh])


def build_pid_geo_token_map(poi_info_df, pid_col="Pid", lat_col="Lat", lon_col="Lon", geohash_precision=6):
    """
    从 poi_info_df 构建:
    Pid -> <g_d><g_r><g_5><g_r><g_j><g_m>
    如果同一个 Pid 有多条记录，则对 Lat/Lon 取均值
    """
    required_cols = [pid_col, lat_col, lon_col]
    for c in required_cols:
        if c not in poi_info_df.columns:
            raise ValueError(f"poi_info_df 缺少必要列: {c}")

    valid_df = poi_info_df.dropna(subset=[pid_col, lat_col, lon_col]).copy()

    # 同一个 Pid 若有多条记录，按均值聚合坐标
    valid_df = (
        valid_df.groupby(pid_col, as_index=False)
        .agg({
            lat_col: "mean",
            lon_col: "mean",
        })
    )

    pid_to_geo_tokens = {}
    for _, row in valid_df.iterrows():
        pid = row[pid_col]
        lat = row[lat_col]
        lon = row[lon_col]
        pid_to_geo_tokens[pid] = geohash_to_tokens(lat, lon, precision=geohash_precision)

    return pid_to_geo_tokens


def save_json(mode, target_dataset, version, div_loss, idorcodebook, geohash_precision=6):
    codebook_path = f"datasets/{target_dataset}/codebooks_{version}_{div_loss}.csv"
    sequence_path = f"datasets/{target_dataset}/data/{mode}.csv"
    poi_info_path = f"datasets/{target_dataset}/poi_info.csv"

    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"找不到 codebook 文件: {codebook_path}")
    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"找不到序列文件: {sequence_path}")
    if not os.path.exists(poi_info_path):
        raise FileNotFoundError(f"找不到 poi_info 文件: {poi_info_path}")

    codebook_df = pd.read_csv(codebook_path)
    poi_sequence_df = pd.read_csv(sequence_path)
    poi_info_df = pd.read_csv(poi_info_path)

    if "Codebook" not in codebook_df.columns or "Pid" not in codebook_df.columns:
        raise ValueError("codebook_df 必须包含列: ['Pid', 'Codebook']")

    codebook_df["Codebook"] = codebook_df["Codebook"].apply(safe_eval)
    poi_to_codebook = dict(zip(codebook_df["Pid"], codebook_df["Codebook"]))

    pid_to_geo_tokens = build_pid_geo_token_map(
        poi_info_df,
        pid_col="Pid",
        lat_col="Lat",
        lon_col="Lon",
        geohash_precision=geohash_precision
    )

    sequences = []
    targets = []

    instruction = (
        "Here is a record of a user's POI accesses. "
        "Your task is to predict the POI that the user is likely to access at the specified time "
        "based on the visit history and geographic information."
    )

    for _, row in poi_sequence_df.iterrows():
        uid = row["Uid"]
        poi_sequence = safe_eval(row["Pids"])
        time_sequence = safe_eval(row["Times"])
        target_time = row["Target_time"]
        target = row["Target"]

        if len(poi_sequence) != len(time_sequence):
            raise ValueError(
                f"Uid={uid} 的 Pids 和 Times 长度不一致: "
                f"{len(poi_sequence)} vs {len(time_sequence)}"
            )

        embedded_sequence = []

        for i, poi in enumerate(poi_sequence):
            geo_tokens = pid_to_geo_tokens.get(poi, "<g_u><g_n><g_k><g_u><g_n><g_k>")

            if idorcodebook == "codebook":
                if poi not in poi_to_codebook:
                    raise KeyError(f"Pid={poi} 不在 codebook_df 中")
                poi_repr = geo_tokens + codebook_to_tokens(poi_to_codebook[poi])
            elif idorcodebook == "id":
                poi_repr = geo_tokens + f"<{poi}>"
            else:
                raise ValueError("Invalid idorcodebook value. Use 'codebook' or 'id'.")

            sep = ", " if i < len(poi_sequence) - 1 else "."
            embedded_sequence.append(f"{poi_repr} at {time_sequence[i]}{sep}")

        target_geo_tokens = pid_to_geo_tokens.get(target, "<g_u><g_n><g_k><g_u><g_n><g_k>")

        if idorcodebook == "codebook":
            if target not in poi_to_codebook:
                raise KeyError(f"Target Pid={target} 不在 codebook_df 中")
            target_embedding = target_geo_tokens + codebook_to_tokens(poi_to_codebook[target])
        elif idorcodebook == "id":
            target_embedding = target_geo_tokens + f"<{target}>"
        else:
            raise ValueError("Invalid idorcodebook value. Use 'codebook' or 'id'.")

        input_text = (
            f"User_{uid} visited: "
            + "".join(embedded_sequence)
            + f" When {target_time} user_{uid} is likely to visit:"
        )

        sequences.append(input_text)
        targets.append(target_embedding)

    semitic_df = pd.DataFrame({
        "instruction": [instruction] * len(sequences),
        "input": sequences,
        "output": targets
    })

    json_data = semitic_df.to_json(orient="records", indent=4, force_ascii=False)

    out_dir = f"datasets/{target_dataset}/data/{version}/{div_loss}"
    os.makedirs(out_dir, exist_ok=True)

    out_path = f"{out_dir}/{mode}_{idorcodebook}_{version}_{div_loss}.json"
    print(f"out_dir: {out_dir}")
    print(f"out_path: {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dataset", type=str, default="NYC", help="target dataset")
    parser.add_argument("--version", type=str, default="v0", help="version")
    parser.add_argument("--div_loss", type=float, default=0.25, help="div loss")
    parser.add_argument("--idorcodebook", type=str, default="codebook", choices=["codebook", "id"], help="idorcodebook")
    parser.add_argument("--geohash_precision", type=int, default=6, help="geohash precision, recommended 6")
    args = parser.parse_args()

    save_json("train", args.target_dataset, args.version, args.div_loss, args.idorcodebook, args.geohash_precision)
    save_json("val", args.target_dataset, args.version, args.div_loss, args.idorcodebook, args.geohash_precision)
    save_json("test", args.target_dataset, args.version, args.div_loss, args.idorcodebook, args.geohash_precision)


if __name__ == "__main__":
    main()
    
# python save_json_geo.py --target_dataset NYC --version v2.4 --div_loss 0.5 --idorcodebook codebook --geohash_precision 6