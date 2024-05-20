import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from config.config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, criterion, optimizer, writer, epoch):
    model.train()

    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{config.general.num_epochs}, Loss: {running_loss / len(train_loader)}")


def evaluate_model(model, test_loader, epoch, writer):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    report = classification_report(y_true, y_pred, output_dict=True)
    for class_label in report:
        if class_label.isdigit():
            writer.add_scalar(f'Accuracy/class_{class_label}', report[class_label]['recall'], epoch)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def train_triplet_model(model, train_loader, val_loader, triplet_loss, optimizer, epoch, writer, num_epochs=10):
    # Initialize k-NN classifier (consider fitting it with training set embeddings outside this function)
    knn = KNeighborsClassifier(n_neighbors=3)


    model.train()
    running_loss = 0.0
    for anchor, positive, negative, _ in train_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()

        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)

        loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)

    print(f"Epoch {epoch + 1}/{config.general.num_epochs}, Training Loss: {running_loss / len(train_loader)}")

    # Validation phase for loss
    model.eval()
    val_loss = 0.0
    embeddings = []
    val_labels = []
    with torch.no_grad():
        for anchor, positive, negative, labels in val_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_embed = model(anchor)
            val_loss += triplet_loss(anchor_embed, model(positive), model(negative)).item()

            # Collect embeddings and labels for classification
            embeddings.extend(anchor_embed.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}")

    # Fit k-NN on the collected embeddings and evaluate
    embeddings = np.array(embeddings)
    val_labels = np.array(val_labels)

    # Assuming you have pre-fitted the k-NN with training embeddings
    knn.fit(embeddings, val_labels)  # This should ideally be fitted with separate training embeddings
    y_pred = knn.predict(embeddings)
    report = classification_report(val_labels, y_pred, output_dict=True)
    for class_label in report:
        if class_label.isdigit():
            writer.add_scalar(f'Accuracy/class_{class_label}', report[class_label]['recall'], epoch)

    print(confusion_matrix(val_labels, y_pred))
    print(classification_report(val_labels, y_pred))