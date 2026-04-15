import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
current_dir = os.getcwd()

# Pid,Uid,Catname,Region,Time,neighbors,forward_neighbors

def _parse_geo_emb_cell(x):
    """Parse geo_emb from CSV cell (list, ndarray, or string from to_csv)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "nan":
            return None
        return np.asarray(ast.literal_eval(s), dtype=np.float32)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


class EmbDataset(Dataset):

    def __init__(
        self,
        datapath,
        use_geo_emb=False,
        geo_emb_col="geo_emb",
        use_catname=True,
        use_region=True,
    ):
        data = pd.read_csv(current_dir + datapath)
        self.ids = data['Pid']
        data['Uid'] = data['Uid'].apply(eval)
        data['Time'] = data['Time'].apply(eval)
        data['neighbors'] = data['neighbors'].apply(eval)
        data['forward_neighbors'] = data['forward_neighbors'].apply(eval)

        path_parts = [p for p in datapath.replace("\\", "/").split("/") if p]
        if len(path_parts) < 2:
            raise ValueError(f"Cannot infer dataset (NYC/TKY/CA) from datapath: {datapath!r}")
        mode = path_parts[-2]
        time_num = 24
        if mode == 'NYC':
            cat_num = 210
            region_num = 92
            neighbor_num = 1084
        elif mode == 'TKY':
            cat_num = 191
            region_num = 60
            neighbor_num = 2294
        elif mode == 'CA':
            cat_num = 304
            region_num = 958
            neighbor_num = 6593
        else:
            raise ValueError("Invalid data mode. Choose from 'NYC', 'TKY', or 'CA'.")


        def to_one_hot_fixed_dim(indices, num_classes, scale_factor=1):
            one_hot = torch.zeros(num_classes, dtype=torch.float32)
            one_hot[indices] = 1
            one_hot *= scale_factor
            return one_hot
        
        self.use_catname = use_catname
        self.use_region = use_region

        if use_catname:
            catgories = []
            for cat in data["Catname"]:
                cat = to_one_hot_fixed_dim(cat, cat_num, scale_factor=1)
                catgories.append(cat)
            self.catgorie = catgories
        else:
            self.catgorie = None

        if use_region:
            regions = []
            for region in data["Region"]:
                region = to_one_hot_fixed_dim(region, region_num, scale_factor=1)
                regions.append(region)
            self.regions = regions
        else:
            self.regions = None

        times =[]
        for time in data[f'Time']:  
            # if len(time) > 10:
            #     time = time[:10]
            time = to_one_hot_fixed_dim(time, time_num, scale_factor=1) 
            times.append(time)
        self.times = times
        
        neighbors = []
        for neighbor in data[f'Uid']:
            # if len(neighbor) > 10:
            #     neighbor = neighbor[:10]
            neighbor = to_one_hot_fixed_dim(neighbor, neighbor_num, scale_factor=1)
            neighbors.append(neighbor)
        self.neighbors = neighbors

        self.use_geo_emb = use_geo_emb
        self.geo_embs = None
        if use_geo_emb:
            if geo_emb_col not in data.columns:
                raise ValueError(
                    f"use_geo_emb=True but column {geo_emb_col!r} not in CSV. "
                    "Run scripts/get_geo_emb.py on poi_info.csv first."
                )
            parsed = [_parse_geo_emb_cell(v) for v in data[geo_emb_col]]
            dims = [p.shape[0] for p in parsed if p is not None]
            if not dims:
                raise ValueError(f"No valid vectors in column {geo_emb_col!r}.")
            geo_dim = dims[0]
            if any(d != geo_dim for d in dims):
                raise ValueError(f"Inconsistent {geo_emb_col} lengths in CSV.")
            self.geo_embs = []
            for p in parsed:
                if p is None:
                    self.geo_embs.append(torch.zeros(geo_dim, dtype=torch.float32))
                else:
                    self.geo_embs.append(torch.from_numpy(p.copy()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        parts = []
        if self.use_catname and self.catgorie is not None:
            parts.append(self.catgorie[idx])
        if self.use_region and self.regions is not None:
            parts.append(self.regions[idx])
        parts.extend([self.times[idx], self.neighbors[idx]])
        if self.use_geo_emb and self.geo_embs is not None:
            parts.append(self.geo_embs[idx])
        return self.ids[idx], torch.cat(parts)

