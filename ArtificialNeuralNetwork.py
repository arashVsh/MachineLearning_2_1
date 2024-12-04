import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


class ANN(nn.Module):
    def __init__(self, layers, activation):
        super(ANN, self).__init__()
        self.layers = layers
        self.activation = activation

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        if self.activation == "relu":
            for i in range(len(self.layers) - 1):
                x = torch.relu(self.layers[i](x))
        else:
            for i in range(len(self.layers) - 1):
                x = torch.sigmoid(self.layers[i](x))

        x = self.layers[-1](x)
        return x


def trainANN(layers, activation, epochs, train_loader, val_loader, device):
    if len(layers) == 0:
        print("No layers defined for the ANN!")
        return None
    if activation not in ["relu", "sigmoid"]:
        print("No valid activation function defined!")
        return None
    if epochs < 1:
        print(f'Number of epochs is {epochs} but should be at least 1')
        return None

    model = ANN(layers=layers, activation=activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Track training and validation losses
    train_losses = []
    val_losses = []

    # Train the model
    for epoch in range(epochs):
        # print(f"Epoch {epoch + 1}")
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # Accumulate training loss

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # Store training loss
        # print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()  # Accumulate validation loss

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)  # Store validation loss
        # print(f"Validation Loss: {avg_val_loss:.4f}")

    # Plot training and validation losses
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o"
    )
    plt.plot(
        range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o"
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


def testANN(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct / total * 100:.2f}%")


# class ANN(nn.Module):
#     def __init__(self, layers):
#         super(ANN, self).__init__()
#         self.fc1 = nn.Linear(28*28, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = x.view(-1, 28*28)  # Flatten
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
