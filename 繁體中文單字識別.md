# 繁體中文單字識別

範例程式碼(使用 pytorch 跟 ResNet18)

https://colab.research.google.com/drive/19ZsLN9HIColISI44ocNvKZW42dPfonHb?usp=sharing#scrollTo=Oh3JHaSsxMqs

## STEP1:colab使用

### (一)連接colab跟google drive

#### 方法1:

![](https://i.imgur.com/RXO1tHj.png)


#### 方法2:


執行此程式碼連接你的雲端硬碟
```
from google.colab import drive
drive.mount('/content/drive')
```
### (二)更改執行階段

預設是 None
可以點選左上角工具列中 [執行階段] >> [變更執行階段] >> [GPU or TPU] 來設定

### (三)查看顯示卡資訊

```
!nvidia-smi
```
##### colab 上執行Linux指令須加上!或是%



## STEP2:訓練資料產生

### 中文單字產生器

https://github.com/rachellin0105/Single_char_image_generator

可以用這個小工具產生中文單字
```
%cd /content/drive/MyDrive/
# 把資料生成工具 clone 下來
!git clone https://github.com/rachellin0105/Single_char_image_generator.git
%cd Single_char_image_generator
```
#### 有基礎的指令可以使用
更多功能可以去[Readme](https://github.com/rachellin0105/Single_char_image_generator/blob/master/README.md)查看

--output_dir: 設定產出的資料夾 (預設是./output)
--num_per_word: 要產生幾個字
--output_image_size: 產生的字體的圖片大小

#### 以下這些可以幫助我們做資料增強

##### 推薦在訓練的時候做資料增強

多增加字體、跟背景模糊以符合現實場景

--blur
--prydown
--lr_motion
--ud_motion
--config_file

### 產生指令

我們只產生10種中文字做示範，所以我們要取出chars.txt前10個中文字
```
# Single_char_image_generator/chars.txt 是字典，預設有102字，可以在上面增減字。這邊因為是示範，我們只留前10個字。
!head -n 10 chars.txt > temp.txt
!mv temp.txt chars.txt
```
安裝需要用到的套件

並且產生中文單字(每種字各100個)
##### 可以多產生測試檔案
```
# 安裝它需要的套件
!python -m pip install -r requirements.txt

# 用一行指令執行生成 
!python OCR_image_generator_single_ch.py --num_per_word=100
```
## STEP3:測試資料產生

建立一個test資料夾放測試資料
```
!mkdir test
!python OCR_image_generator_single_ch.py --num_per_word=30 --output_dir='test'
```
## STEP3:ResNet
### (一)import 需要用到的 library 跟 module
```
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
```
### (二)查看有沒有用到GPU
```
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('GPU')
else:
  device = torch.device('cpu')
  print('CPU')
```
### (三)定義如何取得資料、提供答案給模型

```
class ChineseCharDataset(Dataset):
  def __init__(self, data_file, root_dir, dict_file):
    # data_file:  標註檔的路徑 (標註檔內容: ImagePath, GroundTruth)
    # root_dir: ImagePath所在的資料夾路徑
    # dict_file: 字典的路徑

    # 使用 pandas 將生成的單字labels.txt當作csv匯入進來
    self.char_dataframe = pd.read_csv(data_file, index_col=False, encoding='utf-8', header=None)
    self.root_dir = root_dir
    with open(dict_file, 'r', encoding='utf-8') as f:
      
      # 將資料集包含的字集匯入進來
      word_list = [line for line in f.read().split('\n') if line.strip() != '']
      self.dictionary = {word_list[i]: i for i in range(0, len(word_list))}

    print(self.char_dataframe)
    print(self.dictionary)

  def __len__(self):
    return len(self.char_dataframe)

  def __getitem__(self, idx):
    
    # 取得第idx張圖片的path，並將圖片打開
    image_path = os.path.join(self.root_dir, self.char_dataframe.iloc[idx, 0])
    image = Image.open(image_path)

    # 取得 Ground Truth 並轉換成數字
    char = self.char_dataframe.iloc[idx, 1]
    char_num = self.dictionary[char]
    #資料增強可在這邊做
    
    return (transforms.ToTensor()(image), torch.Tensor([char_num]))
```
### (四)定義資料夾位置包到dataloader、模型儲存位置
```
%cd /content/drive/MyDrive/Single_char_image_generator/

# 宣告好所有要傳入 ChineseCharDataset 的引數
data_file_path = '/content/drive/MyDrive/Single_char_image_generator/output/labels.txt'
root_dir = '/content/drive/MyDrive/Single_char_image_generator/'
dict_file_path = '/content/drive/MyDrive/Single_char_image_generator/chars.txt'
#測試資料集位置
test_data_file_path = '/content/drive/MyDrive/Single_char_image_generator/test/labels.txt'


# 模型儲存位置
save_path = '/content/drive/MyDrive/Single_char_image_generator/checkpoint.pt'

# 宣告我們自訂的Dataset，把它包到 Dataloader 中以便我們訓練使用
char_dataset = ChineseCharDataset(data_file_path, root_dir, dict_file_path)
test_dataset = ChineseCharDataset(test_data_file_path, root_dir, dict_file_path)
char_dataloader = DataLoader(char_dataset, batch_size=2, shuffle=True, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2 )
```
### (五)訓練模型

#### 可以自己調整參數
(一)增加epoch數
(二)使用不同batch size
(三)替換不同ResNet網路層數
(四)不同的優化器(SGD, Momentum, AdaGrad等等)

也可把資料集改成灰階，改成[灰階程式碼](https://colab.research.google.com/drive/1oBom6xI4Eb_U2PgmeaNBBrM2eFeKJ-hG#scrollTo=TU6Aq5zzmtF1)
```
# --- Training ---

# 我們使用torchvision提供的 ResNet-18 當作我們的模型。 
net = models.resnet18(num_classes=10) # num_classes 為類別數量(幾種不一樣的字)
net = net.to(device) # 傳入GPU
net.train()

optimizer = optim.Adam(net.parameters(), lr=0.002)

# 訓練總共Epochs數
epochs = 20

each_loss = []
for i in tqdm(range(1, epochs + 1)):
  losses = 0
  for idx, data in enumerate(char_dataloader):
    image, label = data
    image = image.to(device)
    label = label.squeeze() # 將不同batch壓到同一個dimension
    label = label.to(device, dtype=torch.long)
    
    net.zero_grad()
    result = net(image)

    # 計算損失函數
    loss = F.cross_entropy(result, label)
    losses += loss.item()
    #if idx % 10 == 0:  # 每10個batch輸出一次
      #print(f'epoch {i}- loss: {loss.item()}')

    # 計算梯度，更新模型參數
    loss.backward()
    optimizer.step()

  avgloss = losses / len(char_dataloader)
  each_loss.append(avgloss)
  print(f'{i}th epoch end. Avg loss: {avgloss}')

# 儲存模型
torch.save({
  'epoch': epochs,
  'model_state_dict': net.state_dict(),
  'optimizer_state_dict': optimizer.state_dict(),  
}, save_path)

# 畫出訓練過程圖表 (Y_axis - loss / X_axis - epoch)
plt.plot(each_loss, '-b', label='loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
```
train完可以看到loss慢慢下降

![](https://i.imgur.com/HsP06IV.png)

### (六)測試模型
```
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        image, label = data
        image = image.to(device)
        label = label.to(device)
        
        outputs = net(image)

        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted.view(-1) == label.view(-1)).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

```
output結果

![](https://i.imgur.com/jYSIurW.png)


## 其他資訊將會持續更新