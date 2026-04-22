# Phase 3: Model Development & Training

### 1. Training Infrastructure

**What:** We define helper classes and functions for the training process, including `Early_Stopping` and `train_model`.

**Why:** A robust training loop needs to handle data transfers to the GPU, monitor validation performance, and stop training early if the model stops improving to prevent overfitting.

#### Variable Documentation:
- `Early_Stopping`: (class) Monitors validation loss and stops training if it hasn't improved for a certain number of epochs (`patience`).
- `train_model`: (function) The main loop that performs forward/backward passes and updates model weights.

```
class Early_Stopping():
    def __init__(self, patience = 5, min_delta = 0.00001):
        self.patience = patience 
        self.min_delta = min_delta
        self.min_val_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss + self.min_delta < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False

def train_model(model, train_loader, val_loader, epochs, loss_fn, optimizer, scheduler, early_stopper, device, save_prefix = 'model'):
    history = {'accuracy': [], 'val_acc': [], 'val_loss': [], 'loss': []}
    best_val_loss = np.inf
    model.to(device)

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        correct = 0
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()

        model.eval()
        v_correct = 0
        v_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                v_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                v_correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = 100 * correct / len(train_loader.dataset)
        v_epoch_loss = v_running_loss / len(val_loader.dataset)
        v_accuracy = 100 * v_correct / len(val_loader.dataset)
        
        scheduler.step(v_epoch_loss)
        
        history['accuracy'].append(accuracy)
        history['loss'].append(epoch_loss)
        history['val_acc'].append(v_accuracy)
        history['val_loss'].append(v_epoch_loss)
        
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1:03d}: Loss: {epoch_loss:.4f}, Acc: {accuracy:.2f}%, Val Loss: {v_epoch_loss:.4f}, Val Acc: {v_accuracy:.2f}%, Time: {elapsed:.2f}s')

        if v_epoch_loss < best_val_loss:
            best_val_loss = v_epoch_loss
            torch.save(model.state_dict(), f'{save_prefix}_best.pt')

        if early_stopper(v_epoch_loss):
            print('Early stopping triggered.')
            break
            
    return history
```

### 2. Baseline Architecture

**What:** We define a simple Convolutional Neural Network (CNN) as our baseline.

**Why:** A baseline model allows us to establish a starting point for performance. Any future improvements (like BatchNorm or Dropout) should be measured against this version.

#### Variable Documentation:
- `BaselineCNN`: (class) A simple architecture with 2 convolutional layers and one fully connected layer.

```
class BaselineCNN(torch.nn.Module):
    def __init__(self, num_classes, image_size):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # Halves the image size
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)  # Halves the image size again
        )
        
        # Calculate the spatial dimension after two MaxPool2d(2) layers
        final_spatial_dim = image_size // 4 
        
        # 16 is the number of output channels from the last Conv2d layer
        self.flattened_size = 16 * final_spatial_dim * final_spatial_dim
        
        self.fc = torch.nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

### 3. Hyperparameter Configuration and Execution

**What:** We initialize the Baseline model, loss function, and optimizer, then start the training process.

**Why:** Proper hyperparameter selection is key to convergence. We use `CrossEntropyLoss` for our multi-class problem and `Adam` for efficient weight updates.

#### Variable Documentation:
- `EPOCHS`: (int) Maximum number of training iterations.
- `LEARNING_RATE`: (float) Size of the steps taken during weight optimization.
- `optimizer`: (torch.optim) The algorithm used to update weights (Adam).
- `scheduler`: (torch.optim.lr_scheduler) Dynamically reduces the learning rate when validation loss plateaus.

```
EPOCHS = 30
LEARNING_RATE = 0.005

train_loader_balanced = torch.utils.data.DataLoader(train_set_balanced, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

model_baseline = BaselineCNN(len(classes), IMAGE_SIZE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_baseline.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
early_stopper = Early_Stopping(patience=5)

history_baseline = train_model(
    model_baseline, 
    train_loader_balanced, 
    val_loader, 
    EPOCHS, 
    loss_fn, 
    optimizer, 
    scheduler, 
    early_stopper, 
    device, 
    save_prefix='baseline'
)
```

### 4. Evaluating Results

**What:** We plot the accuracy and loss histories to visualize the training progress.

**Why:** Plots help us identify if the model is overfitting (divergence between train and val loss) or if it has reached a stable performance level.

```
vcpi_util.show_history(history_baseline) # QUERO ISTO EM PLOT 2x2 btw
"""
def show_history(history):

    for key in history.keys():
    
        # summarize history for accuracy
        plt.plot(history[key])
        plt.title(key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.show()
"""
```

### 5. Test Set Evaluation

**What:** We evaluate the final performance of the baseline model on the unseen Test Set.

**Why:** The test set is the ultimate benchmark. Performance here reflects how the model will behave in the real world. We use the best weights saved during training (`baseline_best.pt`) to ensure we are testing the most optimized version of the model.

#### Variable Documentation:
- `test_loader`: (DataLoader) Iterator for the test set.

```
def evaluate_test_set(model, test_loader, device, weights_path):
    # Load the best weights
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    correct = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    test_accuracy = 100 * correct / len(test_loader.dataset)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    return all_targets, all_preds

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
targets, preds = evaluate_test_set(model_baseline, test_loader, device, 'baseline_best.pt')
```



E AGORA A NOVA REDE