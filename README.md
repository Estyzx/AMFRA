# AMFRA

静态多行程农机调度（异构机群）的参考实现。

## 文件树

```
AMFRA-Paper-Code/
├── README.md                                
├── reinforce_static.py                      训练入口：REINFORCE + 多轨迹 leave-one-out 基线，GPU replay，学习率与熵退火，保存 checkpoint
├── simulate.py                              推理入口：checkpoint greedy 推理 + 最近邻/随机基线，输出调度方案与单趟作业上限核算
├── requirements.txt                         固定依赖（Python 3.8：torch 1.13.0、numpy 1.23.4、gym 0.23.1）
├── .gitignore
├── envs/
│   ├── harvest_static_env.py                numpy 版环境：静态实例、逐机异构属性、动作掩码、三项成本 J = C_move + C_start + C_act
│   └── harvest_static_env_vec.py            torch/GPU 批量版：同一环境，状态常驻显存供训练
└── models/
    ├── attention_model_wrapper.py           Backbone（编码器封装）与 Agent（机器—农田联合动作采样、精确 replay）
    └── nets/
        ├── __init__.py
        └── attention_model/
            ├── __init__.py
            ├── embedding.py                 农田节点嵌入
            ├── encoder.py                   图注意力编码器：归一化、自注意力、多头层
            ├── machine_features.py          13 维机器特征构造
            ├── machine_field_relations.py   12 维机器—农田配对关系特征构造
            ├── machine_encoder.py           把机器特征编码为机器嵌入
            ├── amfra_attention.py           AMFRA 机器—农田关系注意力（可选门控）
            ├── decoder.py                   解码器：联合机器—农田 logits 与 field-pointer 变体
            └── multi_head_attention.py      AMFRA 多头注意力（打分与多头组装）
```
