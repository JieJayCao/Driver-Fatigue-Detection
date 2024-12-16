import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from tqdm import tqdm
from thop import profile, clever_format

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def set_seed(seed=42):
    np.random.seed(seed)  # Set NumPy random seed
    torch.manual_seed(seed)  # Set PyTorch random seed
    torch.cuda.manual_seed(seed)  # Set random seed for current GPU
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure convolution operations are deterministic
    torch.backends.cudnn.benchmark = False  # Disable non-deterministic optimization

set_seed(42)  # Call function to set fixed random seed


config = {
    'subjects_num': 12,
    'n_epochs': 25, 
    'batch_size': 50,
    'save_name': 'logs/DualFusion-{epoch:02d}-{val_acc:.2f}',
    'log_path1': 'logs/DualFusion_logs',  # Modified
    'num_class': 2 # Modified, binary classification: 0-awake, 1-fatigue
}



def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

## Data Loader

class EEG_IntraSub_Dataset(Dataset):
    def __init__(self, path, mode, test_sub):
        self.mode = mode
        sub_list = [i for i in range(config['subjects_num'])]
        data = []
        label = []

        
        for i in sub_list:
            data_sub = np.load(path + f'sub_{i}_eeg.npy')
            label_sub = np.load(path + f'sub_{i}_labels.npy')
            data.extend(data_sub)
            label.extend(label_sub)
            
        data = np.array(data)
        label = np.array(label).flatten()
        
        # Generate random indices for synchronized shuffling
        shuffle_idx = np.random.permutation(len(data))
        data = data[shuffle_idx]
        label = label[shuffle_idx]
    
        if mode == 'train':
            data = data[:int(len(data)*0.7)]
            label = label[:int(len(label)*0.7)]
       
        elif mode == 'val':
            data = data[int(len(data)*0.7):int(len(data)*0.9)]
            label = label[int(len(label)*0.7):int(len(label)*0.9)]
        
        elif mode == 'test':
            data = data[int(len(data)*0.9):]
            label = label[int(len(label)*0.9):]
        
        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(label)

    def __len__(self):
        return len(self.data)  

    def __getitem__(self, index):
        return self.data[index], self.label[index]
        
class EEG_InterSub_Dataset(Dataset):
    def __init__(self, path, mode, test_sub):
        self.mode = mode
        self.test_sub = test_sub
        
        if mode == 'train' or mode == 'val':
            train_sub = [i for i in range(config['subjects_num'])]
            train_sub.remove(test_sub)
            data = []
            label = []
            for i in train_sub:
                data_sub = np.load(path + f'sub_{i}_eeg.npy')
                label_sub = np.load(path + f'sub_{i}_labels.npy')
                data.extend(data_sub)
                label.extend(label_sub)
                
            data = np.array(data)
            label = np.array(label).flatten()
            # Generate random indices for synchronized shuffling
            shuffle_idx = np.random.permutation(len(data))
            data = data[shuffle_idx]
            label = label[shuffle_idx]
    
            if mode == 'train':
                data = data[:int(len(data)*0.8)]
                label = label[:int(len(label)*0.8)]
                
            elif mode == 'val':
                data = data[int(len(data)*0.8):]
                label = label[int(len(label)*0.8):]
                   
        
        elif mode == 'test':
            
            data = np.load(path + f'sub_{test_sub}_eeg.npy')
            label = np.load(path + f'sub_{test_sub}_labels.npy')

        
        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(label)      
    def __len__(self):
        return len(self.data)  # Return total number of data samples

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def prep_dataloader(path, mode, batch_size, test_sub, isIntraSub = False, njobs=1):
    if isIntraSub:
        print("IntraSub")
        dataset = EEG_IntraSub_Dataset(path, mode, test_sub)
    else:
        print("InterSub")
        dataset = EEG_InterSub_Dataset(path, mode, test_sub)
        
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), drop_last=False, num_workers=njobs,
                            pin_memory=True)
    return dataloader

## Model

class DualFusion(pl.LightningModule):
    def __init__(self, num_channels=17, output_channels=34, hidden_dim=32, 
                 final_output_dim=2, dropout_prob=0.5):
        super(DualFusion, self).__init__()
        
     
        # Frequency Representation Network
        self.embedding = nn.Linear(193, 32)
        self.FFTCNN = nn.Sequential(
            nn.Conv1d(num_channels, 8, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(8, 32, kernel_size=3, stride=1, padding=1,groups=8),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(1)
            )
        
        # Temporal Multi-Scale Network

        self.CNN1 = nn.Sequential(
            nn.Conv1d(num_channels, output_channels, kernel_size= 3, stride=1, padding=1, groups=num_channels),
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )
        
        self.CNN2 = nn.Sequential(
            nn.Conv1d(num_channels, 20, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(20, output_channels, kernel_size=5, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
            )
        # Sepctral Linear
        self.fc1 = nn.Linear(32, 2)
        
        # Multi-Scale Conv
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(output_channels*2, hidden_dim)
        self.activation = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, final_output_dim)
        self.fc4 = nn.Linear(4, 2)

    
        self._init_weights()
        
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        :param x: Input tensor of shape [batch_size, num_channels, input_dim]
        """
        batch_size, num_channels, input_dim = x.shape
       
        # Spectral Modality
        fft_out = torch.abs(torch.fft.fft(x, dim=-1)[:,:,:193])
        fft_out = self.embedding(fft_out)
        fft_out = self.FFTCNN(fft_out)
        fft_out = fft_out.reshape(batch_size, -1)
        out1 = self.fc1(fft_out)
        
        # Temporal Modality
        x1 = self.global_avg_pool(self.CNN1(x))
        x2 = self.global_avg_pool(self.CNN2(x))    
        x_fuse = torch.cat((x1, x2), dim=1)
        x_fuse = x_fuse.reshape(batch_size, -1)
        out2 = self.fc2(x_fuse)
        out2 = self.activation(out2)
        out2 = self.fc3(out2)
        
        concat_out = torch.cat((out1, out2), dim=1)
        final_out = self.fc4(concat_out)
        
        return final_out

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
     
        return optimizer 
      
    def training_step(self, batch):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        self.log('training_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        loss = {'loss': loss}
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def test_step(self, batch):
        x, y = batch
        preds = self(x)
        
        y_pre = torch.argmax(F.log_softmax(preds, dim=1), dim=1)
        acc = accuracy_score(y.cpu(), y_pre.cpu())
        pre = precision_score(y.cpu(), y_pre.cpu(), average='weighted')
        recall = recall_score(y.cpu(), y_pre.cpu(), average='weighted')
        f1 = f1_score(y.cpu(), y_pre.cpu(), average='weighted')

        self.log('test_acc', acc)
        self.log('test_pre', pre)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        
           
        return {'test_acc': acc, 'test_pre': pre, 'test_recall': recall, 'test_f1': f1} 


def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            preds = model(x)
            y_pre = torch.argmax(F.log_softmax(preds, dim=1), dim=1)
            acc = accuracy_score(y.cpu(), y_pre.cpu())
            pre = precision_score(y.cpu(), y_pre.cpu(), average='weighted')
            recall = recall_score(y.cpu(), y_pre.cpu(), average='weighted')
            f1 = f1_score(y.cpu(), y_pre.cpu(), average='weighted')

    return acc, pre, recall, f1
       
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename=config['save_name'],
    save_top_k=1,
    mode='min',
    save_last=True
)

if __name__ == '__main__':
    tr_path = val_path = test_path =  "Dataset/SEED-VIG-Subset/"
    device = get_device()
    isIntraSub = True
    
    model = DualFusion()
    input = torch.randn(1, 17, 384)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Parameters: {params}")
    
    AC,PR,RE,F1 = 0,0,0,0
    for test_sub in range(config['subjects_num']):
        tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], test_sub, isIntraSub, njobs=6)
        val_set = prep_dataloader(val_path, 'val', config['batch_size'], test_sub, isIntraSub, njobs=6)
        test_set = prep_dataloader(test_path, 'test', config['batch_size'], test_sub, isIntraSub, njobs=1)
        model =  DualFusion().to(device)
        logger = TensorBoardLogger(config['log_path1'])#, config['log_path2'])
        trainer = Trainer(val_check_interval=1.0, max_epochs=config['n_epochs'], devices=[0], accelerator='gpu',
                        logger=logger,
                        callbacks=[
                            #EarlyStopping(monitor='val_loss', mode='min', check_on_train_epoch_end=True, patience=10, min_delta=1e-4),
                            checkpoint_callback
                        ]
                        )
        
        trainer.fit(model, train_dataloaders=tr_set, val_dataloaders=val_set)
        # 保存最终模型
        #trainer.save_checkpoint('FastAlertNet_final.ckpt')

        # 测试并保存结果
        test_results = trainer.test(model, dataloaders=test_set)
        
        # 将测试结果写入文件
        #f = open('DualFusion_test_results.txt', 'a')
        #f.write('Subject:'+str(test_sub))
        
        AC += test_results[0]['test_acc']
        PR += test_results[0]['test_pre']
        RE += test_results[0]['test_recall']
        F1 += test_results[0]['test_f1']
        
    AC /= config['subjects_num']
    PR /= config['subjects_num'] 
    RE /= config['subjects_num']
    F1 /= config['subjects_num']
    print(f"&{AC*100:.2f}",f"&{PR*100:.2f}",f"&{RE*100:.2f}",f"&{F1*100:.2f}")
        #f.write('\n')
        #f.close()