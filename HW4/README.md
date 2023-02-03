# 实验四笔记
## 前序工作
我并没有在Google云盘中成功将数据集导入colab，恰好手头有GPU设备所以选择了在本地实现。  
从Google云盘上下载数据集后直接在目录下unzip解压即可  
环境中有`pytorch`和`tqdm`两个包即可

## simple
在simple难度中，只使用提供的样例代码即可跑到。  
### debug
但需要注意的是，新版本的pytorch在跑样例代码推理阶段时的这句话  
```python
  for feat_paths, mels in tqdm(dataloader):
```
会报错，去掉`tqdm`，改成
```python
  for feat_paths, mels in dataloader:
```
### 改进
将`batch_size`改为512，以更充分的利用GPU
### 结果

|  Private Score  | Public Score  |
|  ----  | ----  |
| 0.91666  | 0.91785 |

对，很神奇，直接连medium都过了……


## Medium
按照提示，我做了以下修改
- `d_model`缩小为70
- 多头注意力处的`nhead`数量减少为一个，也就是只有一个“头”
- 最后预测的MLP改成单层线性，从之前的两个Linear夹一个Relu改成单层Linear
一通操作后的score如下

|  Private Score  | Public Score  |
|  ----  | ----  |
| 0.89888  | 0.90071 |  

不但没达到medium，甚至连之前的都不如，属实难绷…这个助教是不是骗人啊  
进一步修改将TransformerEncoder中的layer改成2，训练时间大幅提高，并且GPU利用率也上去不少。
### 结果
|  Private Score  | Public Score  |
|  ----  | ----  |
| 0.93055  | 0.93166 |
顺利通过medium

## Hard
### pooling
根据助教提示，我首先将pooling换成self-attention为基础的pooling，做了以下操作  
- 将每行数据的第一个替换成固定开始tag（40个-3.6）
- 将mean的pooling换成attention层，并只保留开始tag的输出
#### 结果很差
|  Private Score  | Public Score  |
|  ----  | ----  |
| 0.67611  | 0.68738 |

