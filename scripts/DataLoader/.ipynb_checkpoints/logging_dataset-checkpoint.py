from torch.utils.data import DataLoader, Dataset
import torch

def log_method_call(func):
    def wrapper(*arg, **kwargs):
        print(f"Calling Method : {func.__name__}")
        result = func(*arg, **kwargs)
        print(f"Finished Calling method : {func.__name__}")
        return result 
    return wrapper


class CustomerDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    @log_method_call
    def __getitem__(self, idx):
        # Return feature and label
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
    
    @log_method_call
    def __len__(self):
        return len(self.features)
    
# Example usage
if __name__ == "__main__":
    # Dummy input
    features = torch.randn(100, 124)
    labels = torch.randn(100)

    # Create instance
    data = CustomerDataset(features, labels)

    # Get len
    print(f"The lenght of the data loaded : {data.__len__()}")

    # Get items 
    try:
        feature, label = data.__getitem__(1)

        print(f"Got the features and lebsl from index 1")

    except:
        print("Couldnt use get features, labels from index 1")




