from pycocotools.coco import COCO
import nltk
import pickle
import torch
ann_train_file='F:\PycharmProject\data/annotations/captions_train2014.json'
coco_train = COCO(ann_train_file)
with open(u'F:\PycharmProject\data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

tokens=[]
allcaps=[]
for i in range(5):
    allcap=coco_train.imgToAnns[318556][i]['caption']
    tokens.append(nltk.tokenize.word_tokenize(str(allcap).lower()))
tokens.sort(key=lambda  x: len(x), reverse=True)
for i in range(5):
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens[i]])
    caption.append(vocab('<end>'))
    allcaps.append(caption)
target=torch.Tensor(allcaps[0])
for i in range(target.shape[0]):
    s=target[i].item()
    ss=vocab.idx2word[s]
    print(ss)
s=vocab.idx2word

Batch=len(allcaps)
Num_sentences=len(allcaps[1])

allcap=torch.zeros(Batch, Num_sentences,len(target)).long()


print(len(coco_train.dataset['categories']))

print(len(coco_train.dataset['images']))

print(len(coco_train.dataset['annotations']))
