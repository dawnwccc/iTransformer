import pandas as pd
import numpy as np
import torch
import tqdm

excel_filepath = "赛道二：小、微无人机目标类型识别算法  初赛数据.xlsx"
print("loading data from excel file: ", excel_filepath)
df = pd.read_excel(excel_filepath)
# 根据标签是否为空字符串来分割序列
split_indices = df[df["标签"] != " "].index.tolist()
# 如果最后一个标签不是最后一个数据，添加最后一个数据的索引
if split_indices[-1] != df.index[-1]:
    split_indices.append(df.index[-1] + 1)
print("split data into ", len(split_indices), " segments")
# 分割数据
datas = []
labels = []
start_index = 0
for end_index in tqdm.tqdm(split_indices, total=len(split_indices)):
    if start_index == end_index:
        continue
    segment = df[start_index:end_index].copy()
    # 计算时间变化量
    segment.loc[:, "记录时间(s)"] = segment["记录时间(s)"] - segment["记录时间(s)"].iloc[0]
    segment_label = segment["标签"].iloc[0]
    segment_data = segment[["目标方位角(°)", "目标斜距(m)", "相对高度(m)", "径向速率(m/s)", "记录时间(s)", "RCS"]].values
    datas.append(segment_data)
    labels.append(segment_label)
    start_index = end_index
print("save data to dataset.pt")
datas_array = np.array(datas, dtype=object)
labels_array = np.array(labels)
shuffled_indices = np.random.permutation(len(datas))
shuffled_datas_tensor = [torch.tensor(array, dtype=torch.float32) for array in datas_array[shuffled_indices]]
shuffled_labels_tensor = torch.tensor(labels_array[shuffled_indices], dtype=torch.uint8)

train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
train_size = int(len(shuffled_datas_tensor) * train_ratio)
val_size = int(len(shuffled_datas_tensor) * val_ratio)
test_size = len(shuffled_datas_tensor) - train_size - val_size

# 训练集
train_datas = shuffled_datas_tensor[:train_size]
train_labels = shuffled_labels_tensor[:train_size]
# 验证集
val_datas = shuffled_datas_tensor[train_size:train_size + val_size]
val_labels = shuffled_labels_tensor[train_size:train_size + val_size]
# 测试集
test_datas = shuffled_datas_tensor[train_size + val_size:]
test_labels = shuffled_labels_tensor[train_size + val_size:]

torch.save({
    "datas": train_datas,
    "labels": train_labels
}, "dataset/train.pt")
torch.save({
    "datas": val_datas,
    "labels": val_labels
}, "dataset/val.pt")
torch.save({
    "datas": test_datas,
    "labels": test_labels
}, "dataset/test.pt")

print("train data size: ", len(train_datas))
print("val data size: ", len(val_datas))
print("test data size: ", len(test_datas))