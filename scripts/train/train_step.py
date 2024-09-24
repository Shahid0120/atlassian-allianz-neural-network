import torch 
from torch.nn import Module

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
  model.train()  # Set model to train mode

  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(train_dataloader):
      # Forward pass
      y_pred = model(X)

      # Calculate loss
      loss_val = loss(y_pred, y)
      train_loss += loss_val.item()

      # Optimizer zero grad
      optimizer.zero_grad()

      # Backward pass
      loss_val.backward()

      # Update weights
      optimizer.step()

      # Calculate accuracy
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item() / len(y_pred)

  # Average loss and accuracy per batch
  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  return train_loss, train_acc