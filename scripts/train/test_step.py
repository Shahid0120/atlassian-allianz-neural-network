import torch 
from torch.nn import Module

# Test step function
def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              loss: torch.nn.Module):
  model.eval()  # Set model to evaluation mode

  test_loss, test_acc = 0, 0

  with torch.inference_mode():
      for batch, (X, y) in enumerate(test_dataloader):
          # Forward pass
          test_y_logits = model(X)

          # Calculate loss
          loss_val = loss(test_y_logits, y)
          test_loss += loss_val.item()

          # Calculate accuracy
          test_pred_labels = test_y_logits.argmax(dim=1)
          test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

  # Average loss and accuracy per batch
  test_loss /= len(test_dataloader)
  test_acc /= len(test_dataloader)

  return test_loss, test_acc