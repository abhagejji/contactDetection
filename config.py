from torchvision import datasets, transforms, models

from data_loader import MyDataLoading_color,MyDataLoading_four_channel
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset

class MyTansforms():
    def __init__(self):
        pass

    def shadow_training_transform(self):
        self.traintransforms = transforms.Compose([
            transforms.GaussianBlur(kernel_size = 5),
            transforms.RandomResizedCrop(244),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
        return self.traintransforms
    def shadow_test_transform(self,):
        self.testtransforms = transforms.Compose([
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
        return self.testtransforms
    

   
class MyDataset(pl.LightningModule):

    def __init__(self,train_df,val_df,test_df,color_channel,shadow_channel,theDataLoader):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        #TODO
        self.column_x = 'path_to_img'
        self.column_y =  'label'
        # self.data_manager = MyConfig()
        self.color_channel = color_channel
        self.shadow_channel = shadow_channel

        self.num_workers = 16 #TODO
        self.batch_size = 64

        self.transforms = MyTansforms()
        self.transforms_train = self.transforms.shadow_training_transform()
        self.MyDataLoading = theDataLoader

    def split_xy(self,df,default_string):
        colx = df[self.column_x]
        x = colx.apply(lambda x: default_string + x if isinstance(x, str) else x)
        return x,df[self.column_y]
   
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size= self.batch_size, shuffle=True, num_workers=self.num_workers)#self.data_manager.get_train_dataloader()
        #,last_drop = True, prefetch_factor = 2
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size= self.batch_size, shuffle=False, num_workers=self.num_workers)#self.data_manager.get_train_dataloader() #self.data_manager.get_validation_dataloader()

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size= self.batch_size, shuffle=False, num_workers=self.num_workers)


    def setup(self, stage: str) ->None:
        self.trainX, self.trainY = self.split_xy(self.train_df)
        self.train_data = self.MyDataLoading(self.trainX, self.trainY, color_channel = self.color_channel, shadow_channel =self.shadow_channel,mode = "train" , transforms =  self.transforms_train )

        self.valX, self.valY = self.split_xy(self.val_df)
        self.val_data = self.MyDataLoading(self.valX, self.valY, color_channel = self.color_channel, shadow_channel =self.shadow_channel,mode = "train" , transforms =  self.transforms_train )

        self.testX, self.testY = self.split_xy(self.test_df)
        self.test_data = self.MyDataLoading(self.testX, self.testY, color_channel = self.color_channel, shadow_channel =self.shadow_channel,mode = "train" , transforms =  self.transforms_train )
