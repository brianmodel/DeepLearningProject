import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from utils import SAVE_DIR

def checkpoint(save_dir, model, optimizer, scheduler, epoch):
    checkpoint = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler.state_dict(),
    }
    
    save_path = os.path.join(save_dir, f"checkpoint_{epoch}.pt")
    torch.save(checkpoint, save_path)

def load_checkpoint(file, model, scheduler, optimizer):
      checkpoint = torch.load(file)
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      return model, optimizer, checkpoint['epoch']

def save_model(file, model):
    torch.save(model.state_dict(), file)

def load_model(file, model):
    model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))

def train(model, train_dl, val_dl, num_epochs, lr, device, model_name):
    # Writing training data to tensorboard
    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "runs"))

    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='cos')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        model.train()
        print("-----------------------------------")
        print("Epoch %d" % (epoch+1))
        print("-----------------------------------")

        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        
        # Repeat for each batch in the training set
        progress_bar = tqdm(train_dl, ascii = True)
        for i, (inputs, labels) in enumerate(progress_bar):
            # Get the input features and target labels, and put them on the GPU
            # inputs, labels = inputs.to(device), labels.to(device)

            # Since we are processing data as part of pipeline, we only need to move labels to device
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((i + 1), loss.item()))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)

        writer.add_scalar("Training loss", avg_loss, epoch)
        writer.add_scalar("Training accuracy", acc, epoch)
        writer.add_scalar("Validation loss", val_loss, epoch)
        writer.add_scalar("Validation accuracy", val_acc, epoch)
        writer.flush()

        print(f"Training Loss: {avg_loss:.2f} Accuracy: {acc:.2f}")
        print(f"Validation Loss: {val_loss:.2f} Accuracy: {val_acc:.2f}")

        if ((epoch + 1) % 5 == 0):
            checkpoint_dir = os.path.join(SAVE_DIR, 'checkpoints')
            print(f'Checkpointing to {checkpoint_dir}')
            checkpoint(checkpoint_dir, model, optimizer, scheduler, epoch+1)
    
    save_model(os.path.join(SAVE_DIR, f"{model_name}.pt"), model)
    print('Finished Training!')
    writer.close()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_prediction = 0
    total_prediction = 0
    
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm(dataloader, ascii=True)
        for i, (inputs, labels) in enumerate(progress_bar):
            # inputs = inputs.to(device)
            labels = labels.to(device)

            # Normalize the inputs
#             inputs_m, inputs_s = inputs.mean(), inputs.std()
#             inputs = (inputs - inputs_m) / inputs_s
            
            outputs = model(inputs)
            
            _, prediction = torch.max(outputs,1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((i + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    acc = correct_prediction / total_prediction
    return avg_loss, acc
