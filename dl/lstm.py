#coding=utf-8
import torch
import torch.nn as nn

#一个栗子，假如我们输入有3个句子，每个句子都由5个单词组成，而每个单词用10维的词向量表示，则seq_len=5, batch=3, input_size=10

#词向量维数10,隐藏元维度20,2个LSTM层串联,
bilstm = nn.LSTM(input_size=10, hidden_size=3, num_layers=2, bidirectional=True)
#序列长度seq_len=5,batch_size=3,词向量维数=10
input = torch.randn(5, 3, 10)
#初始化的隐藏元和记忆元,通常它们是维度是一样的
#2个LSTM层，batch_size=3,隐藏元维度20
h0 = torch.randn(4, 3, 3)
c0 = torch.randn(4, 3, 3)

#这里有2层lstm，output是最后一层lstm的每个词向量对应隐藏层的输出,其与层数无关，只与序列长度相关
#hn,cn是所有层最后一个隐藏元和记忆元的输出
output, (hn, cn) = bilstm(input, (h0, c0))
# output=output.permute(1,0,2)
# hn=hn.permute(1,0,2)
# cn=cn.permute(1,0,2)
print('output shape: ', output.shape)
print('hn shape: ', hn.shape)
print('cn shape: ', cn.shape)

print('output:',output)
print('output0:',output[0])
print('hn:',hn)
print('cn:',cn)
# 输出：
# output shape:  torch.Size([5, 3, 40])
# hn shape:  torch.Size([4, 3, 20])
# cn shape:  torch.Size([4, 3, 20])

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# torch.manual_seed(1)
#
# lstm = nn.LSTM(3, 3)  # 输入维度是3, 输出维度也是3
# print(lstm.all_weights)
#
# inputs = [torch.randn(1, 3) for _ in range(5)] # 构造一个长度为5的序列
#
# print('Inputs:',inputs)
#
# # 初始化隐藏状态
# hidden = (torch.randn(1, 1, 3),
#           torch.randn(1, 1, 3))
# print('Hidden:',hidden)
# for i in inputs:
#  # 将序列的元素逐个输入到LSTM，这里的View是把输入放到第三维，看起来有点古怪，回头看看上面的关于LSTM输入的描述，这是固定的格式，以后无论你什么形式的数据，都必须放到这个维度。就是在原Tensor的基础之上增加一个序列维和MiniBatch维，这里可能还会有迷惑，前面的1是什么意思啊，就是一次把这个输入处理完，在输入的过程中不会输出中间结果，这里注意输入的数据的形状一定要和LSTM定义的输入形状一致。
#     # 经过每步操作,hidden 的值包含了隐藏状态的信息
#  out, hidden = lstm(i.view(1, 1, -1), hidden)
# print('out1:',out)
# print('hidden2:',hidden)
# # 另外, 我们还可以一次对整个序列进行训练. LSTM 返回的第一个值表示所有时刻的隐状态值,
# # 第二个值表示最近的隐状态值 (因此下面的 "out"的最后一个值和 "hidden" 的值是一样的).
# # 之所以这样设计, 是为了通过 "out" 的值来获取所有的隐状态值, 而用 "hidden" 的值来
# # 进行序列的反向传播运算, 具体方式就是将它作为参数传入后面的 LSTM 网络.
#
# # 增加额外的第二个维度
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print('out2',out)
# print('hidden3',hidden)