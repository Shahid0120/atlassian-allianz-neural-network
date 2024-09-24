import torch 
from torch.nn import Module

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int):
    

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(model=model,
                                            train_dataloader=train_dataloader,
                                            loss=loss,
                                            optimizer=optimizer)
        
        test_loss, test_acc = test_step(model=model,
                                        test_dataloader=test_dataloader,
                                        loss=loss)
        
        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        # Append training loss for each epoch
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar('Loss/train', test_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        
    # Append into results
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    return results