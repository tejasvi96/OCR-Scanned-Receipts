# For recognition of textdata in bounding boxes
import os
import cv2
import torch
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
logger.add("OCR.log")
import torch.optim as optim
import tqdm
from torch.utils.data import TensorDataset, DataLoader
train_dir=r"/home/tejasvi/0325updated.task1train(626p)"
val_dir=r"/home/tejasvi/text.task1_2-test（361p)"
load_model_file=r'./OCR.pt'
max_length=31
row=1000
column=50
n_epochs=15
lr=0.001
names=os.listdir(train_dir)
names=[i.replace(".jpg",'') for i in names if ".jpg" in i and '(' not in i]
names_val=os.listdir(val_dir)
names_val=[i.replace(".jpg",'') for i in names_val if ".jpg" in i and '(' not in i]
bsize=64
embedding_size=180
encoder_dim=180
hidden_size=180
densenet_depth=32
densenet_growthrate=24
pad_token = 0
SOS_token = 1
EOS_token = 2
unk_token=3
device=torch.device("cuda:2")

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
#     print(h,w)
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def get_data(train_dir,names):
    images=[]
    textdata=[]
    for nm in names:
        image=cv2.imread(train_dir+"//"+nm+".jpg")
        data_file=train_dir+"//"+nm+".txt"
        with open(data_file,'r',encoding='utf-8') as fp:
            try:
                data=fp.readlines()
            except:
                continue
                pass
        try:
            text=[i.split(',')[8].replace("\n","") for i in data]
            textdata.append(text)
        except:
            continue
        bboxes=[]
        for i in data:
            coords=i.split(',')[:8]
            coords=[int(i) for i in coords]
            xs=[coords[i] for i in range(0,len(coords),2)]
            ys=[coords[i] for i in range(1,len(coords),2)]
            xmin=min(xs)
            xmax=max(xs)
            ymin=min(ys)
            ymax=max(ys)
            outs=[xmin,xmax,ymin,ymax]
            bboxes.append(outs)
        h, w= image.shape[:2]
        image_c = image.copy()
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)

        for i in range(len(bboxes)):
            cropped=image[bboxes[i][2]:bboxes[i][3],bboxes[i][0]:bboxes[i][1]]
            cropped=image_resize(cropped/255.0,height=50)
            images.append(cropped)
    return images,textdata
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "pad", 2: "EOS",1:"SOS",3:"unk"}
        self.n_words = 4  # Count pad and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def get_tokenized_data(lang,flat_list):
    targets=[]
    lens=[]
    for i in flat_list:
        encoded=[lang.word2index[j] for j in i ]
        encoded=[SOS_token]+encoded
        encoded.append(EOS_token)
        l=len(encoded)
        lens.append(l)
        if(l<max_length):
            encoded=encoded+([pad_token]*(max_length-l))
        elif (l>=max_length):
            encoded=encoded[:max_length-1]+[EOS_token]
        targets.append(encoded)
    return targets
def get_images_array(images,r,c):
    n=len(images)
    cnt=0
    ims=np.zeros((n,c,r),dtype=np.float32)
    for i in range(len(images)):
        try:
            ims[i,:,:min(images[i].shape[1],r) ]=images[i][:,:min(images[i].shape[1],r) ]
        except:
            cnt+=1
            pass
        if cnt!=0:
            print("Warning some image are skipped")
    return ims
train_images,textdata=get_data(train_dir,names)
eng=Lang("english")
for i in textdata:
    for j in i:
        eng.addSentence(j)
train_textdata=[item for sublist in textdata for item in sublist]
tokenized_train_textdata=get_tokenized_data(eng,train_textdata)

arr_tokenized_train_textdata=np.array(tokenized_train_textdata)
ims=get_images_array(train_images,row,column)
traindataset=TensorDataset(torch.from_numpy(ims),torch.from_numpy(arr_tokenized_train_textdata))
trainloader=DataLoader(traindataset,batch_size=bsize)
val_images,val_textdata=get_data(val_dir,names_val)
val_textdata=[item for sublist in val_textdata for item in sublist]
tokenized_val_textdata=get_tokenized_data(eng,val_textdata)
arr_tokenized_val_textdata=np.array(tokenized_val_textdata)
val_ims=get_images_array(val_images,row,column)
valdataset=TensorDataset(torch.from_numpy(val_ims),torch.from_numpy(arr_tokenized_val_textdata))
valloader=DataLoader(valdataset,batch_size=8)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
#         self.max1=nn.MaxPool2d(kernel_size=2)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=24,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
#         in_planes=48
        kernel_size=3
        stride=1
        self.conv1 = nn.Conv2d(1, in_planes, kernel_size=kernel_size, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.max1=nn.MaxPool2d(kernel_size=2)
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
#         out = self.max1(out)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
#         out = out.view(-1, self.in_planes)
        out=out.permute(0,2,3,1)    
        return out


# In[ ]:


class Decoder(nn.Module):
    def __init__(self, vocab_size,encoder_dim, hidden_size, embedding_dim, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.Linear(encoder_dim + hidden_size, 1)
        self.gru = nn.GRU(hidden_size + embedding_dim, hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim = 1)
        self.relu = nn.ReLU()

    def forward(self, decoder_input, current_hidden_state, encoder_outputs):
        decoder_input = self.embedding(decoder_input)    # (BATCH_SIZE, EMBEDDING_DIM)

        aligned_weights = torch.randn(encoder_outputs.size(1), encoder_outputs.size(0)).to(self.device)

        encoder_outputs=encoder_outputs.permute(1,0,2)

        for i in range(encoder_outputs.size(0)):

            aligned_weights[i] = self.attention(torch.cat((current_hidden_state.squeeze(dim=0), encoder_outputs[i] ), dim = -1)).squeeze()

        aligned_weights = self.softmax(aligned_weights)   # (BATCH_SIZE, HIDDEN_STATE * 2)
        aligned_weights = aligned_weights.view(aligned_weights.size(1), aligned_weights.size(0))

        encoder_outputs=encoder_outputs.permute(1,0,2)
        
        context_vector = torch.bmm(aligned_weights.unsqueeze(1), encoder_outputs)
#     
        x = torch.cat((context_vector.squeeze(1), decoder_input), dim = 1).unsqueeze(0)
        x = self.relu(x)
        x, current_hidden_state = self.gru(x, current_hidden_state)
        x = self.log_softmax(self.dense(x.squeeze(0)))
        return x, current_hidden_state, aligned_weights
class AttenS2S(nn.Module):
    def __init__(self, encoder, decoder, max_sent_len, device):
        super(AttenS2S, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_sent_len = max_sent_len

    def forward(self, source, target, tf_ratio = .5):
        enc_outputs = self.encoder(source)
        enc_outputs=enc_outputs.squeeze(dim=1)

        dec_outputs = torch.zeros(target.size(0), target.size(1), self.decoder.vocab_size).to(self.device)
        dec_input = target[:, 0]
        dec_h0 = torch.zeros(1, dec_input.size(0), encoder_dim).to(device)
        weights = torch.zeros(target.size(1), target.size(0), target.size(1))   # (TARGET_LEN, BATCH_SIZE, SOURCE_LEN)
        for k in range(target.size(1)):
            out, dec_h0, w = self.decoder(dec_input, dec_h0, enc_outputs)
            weights[k, :, :] = w
            dec_outputs[:, k] = out
            if np.random.choice([True, False], p = [tf_ratio, 1-tf_ratio]):
                dec_input = target[:, k]
            else:
                dec_input = out.argmax(1).detach()

        return dec_outputs, weights

def calc_val_loss():
    val_items=0
    val_loss=0
    model.eval()
    valloader=DataLoader(valdataset,batch_size=bsize)
    for i,(data,txt) in tqdm.tqdm(enumerate(valloader)):
        with torch.no_grad():
            val_items+=data.shape[0]
            data=data.to(device)
            txt=txt.to(device)
            data=data.unsqueeze(1)
            outputs,weights=model(data.float(),txt.long())
            y=txt
            loss = criterion(outputs.resize(outputs.size(0) * outputs.size(1), outputs.size(-1)), y.resize(y.size(0) * y.size(1)))
            val_loss+=loss.item()
    return val_loss/val_items

trainloader=DataLoader(traindataset,batch_size=bsize)
vocab_size=eng.n_words
encoder=DenseNet3(depth=densenet_depth,growth_rate=densenet_growthrate,num_classes=None);
decoder=Decoder(vocab_size=vocab_size,encoder_dim=encoder_dim, hidden_size=hidden_size,embedding_dim=embedding_size,device=device)
model=AttenS2S(encoder,decoder,max_length,device)

model.load_state_dict(torch.load(load_model_file,map_location=device))
optimizer=optim.Adam(model.parameters(),lr=lr)
criterion=nn.CrossEntropyLoss(ignore_index=pad_token)
val_loss=10
val_loss_values=[]
train_loss_values=[]
model=model.to(device)
for j in range(n_epochs):
    sum_loss=0
    n_items=1
    model.train()
    for i,(data,txt) in tqdm.tqdm(enumerate(trainloader)):
        n_items+=data.shape[0]
        data=data.to(device)
        txt=txt.to(device)
        optimizer.zero_grad()

        data=data.unsqueeze(1)
        outputs,weights=model(data.float(),txt.long())
        y=txt
        loss = criterion(outputs.resize(outputs.size(0) * outputs.size(1), outputs.size(-1)), y.resize(y.size(0) * y.size(1)))

        sum_loss+=loss.item()
        loss.backward()
        optimizer.step()
    train_loss_values.append(sum_loss/n_items)
    logger.info("Training "+str(sum_loss/n_items))
    val_loss_values.append(calc_val_loss())
    logger.info("Validation "+str(val_loss_values[j]))
    if val_loss_values[j]<val_loss:
        val_loss=val_loss_values[j]
        torch.save(model.state_dict(),"./OCR.pt")

# CAndo
# see what the maximum sized image actually looks like and try to reduce that
# Remove the maxpool completely
# Follwo standard Densenet
# If actually convergeence is there
# Then think of multiscale attention where the last layer just before avg pooling is concatenated
# Dont change attention keep the atteantion as bahdanou only

