from data_preprocessing import datalist, prog_labellist
from torch.utils.data import Dataset, DataLoader

# DATASET CLASS
class UWHVFDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label

TRAIN_RATIO = 0.8    # 80% for training
VAL_RATIO = 0.1      # 10% for validation
TEST_RATIO = 0.1     # 10% for testing

num_train = int(TRAIN_RATIO * len(datalist))
num_val = int(VAL_RATIO * len(datalist))
num_test = int(TEST_RATIO * len(datalist))

# split data and labels
training_data = datalist[:num_train]
training_labels = prog_labellist[:num_train]

validation_data = datalist[num_train:num_train +num_val] 
validation_labels = prog_labellist[num_train:num_train+num_val]

test_data = datalist[num_train+num_val:num_train + num_val+num_test]   
test_labels = prog_labellist[num_train+num_val:num_train+num_val+num_test]

# create datasets
train_dataset = UWHVFDataset(training_data, training_labels, transform=None)
val_dataset = UWHVFDataset(validation_data, validation_labels, transform=None)
test_dataset = UWHVFDataset(test_data, test_labels, transform=None)

# DATALOADER CLASS
dataloaders = {
    'train' : DataLoader(train_dataset, batch_size=5, shuffle=True),
    'validation' : DataLoader(val_dataset, batch_size=5, shuffle=False),
    'test' : DataLoader(test_dataset, batch_size=5, shuffle=False)
}

print("Dataset and Dataloader complete!")