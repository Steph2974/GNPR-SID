- 论文

  - https://arxiv.org/abs/2506.01375
  - https://arxiv.org/html/2506.01375v2
  - https://mp.weixin.qq.com/s/4hqfyFSYc0e5lF19zaYmcg
  - https://mp.weixin.qq.com/s?__biz=MzUzMjY3NjcxOA==&mid=2247484556&idx=1&sn=e2341342040ec33abc434beb238ead5f&scene=21&poc_token=HOx-LWmje6kCEdHhtjHLrEyAR5QeVJTe43yFj_tf

- 实验

  - We evaluate our approach on three real-world datasets: Foursquare-NYC (Yang et al., [2014](https://arxiv.org/html/2506.01375v2#bib.bib45)), Foursquare-TKY (Yang et al., [2014](https://arxiv.org/html/2506.01375v2#bib.bib45)), and Gowalla-CA (Cho et al., [2011](https://arxiv.org/html/2506.01375v2#bib.bib6)).

    - 搜到其他研究：https://github.com/w11wo/GenUP	

      - We followed the dataset preparation of [LLM4POI](https://github.com/neolifer/LLM4POI) for the FourSquare-NYC, Gowalla-CA, and FourSquare-TKY datasets. We also provide the processed datasets on [🤗 Hugging Face](https://huggingface.co/datasets/w11wo/LLM4POI) Please refer to their repository for more details.

        > ❗️ Moscow and Sao Paulo preprocessing steps will be made available soon.

        | Dataset             | URL                                                          |
        | ------------------- | ------------------------------------------------------------ |
        | FourSquare-NYC      | [🤗](https://huggingface.co/datasets/w11wo/FourSquare-NYC-POI) |
        | FourSquare-TKY      | [🤗](https://huggingface.co/datasets/w11wo/FourSquare-TKY-POI) |
        | Gowalla-CA          | [🤗](https://huggingface.co/datasets/w11wo/Gowalla-CA-POI)    |
        | FourSquare-Moscow   | [🤗](https://huggingface.co/datasets/w11wo/FourSquare-Moscow-POI) |
        | FourSquare-SaoPaulo | [🤗](https://huggingface.co/datasets/w11wo/FourSquare-SaoPaulo-POI) |

  - Following Yan et al. ([2023](https://arxiv.org/html/2506.01375v2#bib.bib44)), 移除交互次数少于 10 次的兴趣点（POI）和签到次数少于 10 次的用户

  - 

- 问题

  - 为什么跑不了 dataprocess.ipynb ？

    - 因为没有原始数据，作者已经跑完并生成了 NYC 数据集所需 JSON 格式数据

  - 原始数据在哪里？

    - https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset

    - 数据流总结

      原始数据 → data.csv → train_data.csv → data/train.csv → train_codebook.json → LLM训练
                          ↓
                    poi_info.csv → RQVAE训练 → codebooks_X.csv

      RQVAE模型：学习将连续POI嵌入转换为离散量化表示，为LLM提供token化的POI编码

      LLM模型：使用量化后的POI编码进行序列预测和推荐

- 步骤：

  1. 准备数据
     1. dataprocess.ipynb 原始数据清洗和格式化
     2. data2json.ipynb 转化为 JSON 格式
  2. 训练语义 id
     1. code/train_rqvae.py
     2. codebook.py
  3. 微调 LLaMA 
  4. 评估

- 细节
  - 