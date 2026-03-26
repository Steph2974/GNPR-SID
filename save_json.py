import os
import pandas as pd
import numpy as np
import argparse

def save_json(mode, target_dataset, version, div_loss, idorcodebook):

    codebook_df = pd.read_csv(f'datasets/{target_dataset}/codebooks_{version}_{div_loss}.csv')
    poi_sequence_df = pd.read_csv(f'datasets/{target_dataset}/data/{mode}.csv')


    codebook_df['Codebook'] = codebook_df['Codebook'].apply(eval)

    poi_to_codebook = dict(zip(codebook_df['Pid'], codebook_df['Codebook']))

    users = []
    sequences = []
    targets = []

    for _, row in poi_sequence_df.iterrows():
        uid = row['Uid']
        poi_sequence = eval(row['Pids'])
        time_sequence = eval(row['Times'])
        target_time = row['Target_time']
        target = row['Target']

        if idorcodebook == 'codebook':
            embedded_sequence = [
                ''.join([f"<{chr(97 + idx)}_{code}>" for idx, code in enumerate(poi_to_codebook[poi])]) + f' at {time_sequence[i]}, ' 
                if i < len(poi_sequence) - 1 else 
                ''.join([f"<{chr(97 + idx)}_{code}>" for idx, code in enumerate(poi_to_codebook[poi])]) + f' at {time_sequence[i]}.'
                for i, poi in enumerate(poi_sequence)
            ]
            target_embedding = ''.join([f"<{chr(97 + idx)}_{code}>" for idx, code in enumerate(poi_to_codebook[target])])
        
        elif idorcodebook == 'id':
            embedded_sequence = [
                f"<{poi}>" + f' at {time_sequence[i]}, ' if i < len(poi_sequence) - 1 else
                f"<{poi}>" + f' at {time_sequence[i]}.'
                for i, poi in enumerate(poi_sequence)
            ]
            target_embedding = f"<{target}>"
        
        else:
            raise ValueError("Invalid idorcodebook value. Use 'codebook' or 'id'.")

        instruction = f"Here is a record of a user's POI accesses, your task is based on the history to predict the POI that the user is likely to access at the specified time."
        input = f"User_{uid} visited: " + "".join(embedded_sequence) + f" When {target_time} user_{uid} is likely to visit:"
        

        sequences.append(input)
        targets.append(target_embedding)

    semitic_df = pd.DataFrame({
        'instruction': instruction,
        'input': sequences,
        'output': targets
    })

    json_data = semitic_df.to_json(orient="records", indent=4)

    out_dir = f"datasets/{target_dataset}/data/{version}/{div_loss}"
    print(f"out_dir: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{mode}_{idorcodebook}_{version}_{div_loss}.json"
    print(f"out_path: {out_path}")
    with open(out_path, "w") as f:
        f.write(json_data)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dataset", type=str, default="NYC", help="target dataset")
    parser.add_argument("--version", type=str, default="v0", help="version")
    parser.add_argument("--div_loss", type=float, default=0.25, help="div loss")
    parser.add_argument("--idorcodebook", type=str, default="codebook", help="idorcodebook")
    args = parser.parse_args()
    save_json("train", args.target_dataset, args.version, args.div_loss, args.idorcodebook)
    save_json("val", args.target_dataset, args.version, args.div_loss, args.idorcodebook)
    save_json("test", args.target_dataset, args.version, args.div_loss, args.idorcodebook)


if __name__ == "__main__":
    main()