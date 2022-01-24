import torch
import config
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def train(dataloader, model, loss_fn, optimizer):
    model.train()  # Sets the model for training.

    total = 0
    correct = 0
    running_loss = 0

    for batch, (x, y) in enumerate(dataloader):  # Iterates through the batches.

        output = model(x.to(config.DEVICE))  # model's predictions.
        loss = loss_fn(output, y.to(config.DEVICE))  # loss calculation.

        running_loss += loss.item()

        total += y.size(0)
        predictions = output.argmax(
            dim=1).cpu().detach()  # Index for the highest score for all the samples in the batch.
        correct += (
                    predictions == y.cpu().detach()).sum().item()  # No.of.cases where model's predictions are equal to the label.

        optimizer.zero_grad()  # Gradient values are set to zero.
        loss.backward()  # Calculates the gradients.
        optimizer.step()  # Updates the model weights.

    avg_loss = running_loss / len(dataloader)  # Average loss for a single batch

    print(f'\nTraining Loss = {avg_loss:.6f}', end='\t')
    print(f'Accuracy on Training set = {100 * (correct / total):.6f}% [{correct}/{total}]')  # Prints the Accuracy.

    return avg_loss


def validate(dataloader, model, loss_fn):
    model.eval()  # Sets the model for evaluation.

    total = 0
    correct = 0
    running_loss = 0

    with torch.no_grad():  # No need to calculate the gradients.

        for x, y in dataloader:
            output = model(x.to(config.DEVICE))  # model's output.
            loss = loss_fn(output, y.to(config.DEVICE)).item()  # loss calculation.
            running_loss += loss

            total += y.size(0)
            predictions = output.argmax(dim=1).cpu().detach()
            correct += (predictions == y.cpu().detach()).sum().item()

    avg_loss = running_loss / len(dataloader)  # Average loss per batch.

    print(f'\nValidation Loss = {avg_loss:.6f}', end='\t')
    print(f'Accuracy on Validation set = {100 * (correct / total):.6f}% [{correct}/{total}]')  # Prints the Accuracy.

    return avg_loss


def optimize(train_dataloader, valid_dataloader, model, loss_fn, optimizer, nb_epochs):
    # Lists to store losses for all the epochs.
    train_losses = []
    valid_losses = []

    for epoch in range(nb_epochs):
        print(f'\nEpoch {epoch + 1}/{nb_epochs}')
        print('-------------------------------')
        train_loss = train(train_dataloader, model, loss_fn, optimizer)  # Calls the train function.
        train_losses.append(train_loss)
        valid_loss = validate(valid_dataloader, model, loss_fn)  # Calls the validate function.
        valid_losses.append(valid_loss)

    print('\nTraining has completed!')

    return train_losses, valid_losses


def test(dataloader, model):
    y_true_tensor = torch.tensor([]).cuda()
    y_pred_tensor = torch.tensor([]).cuda()

    model.eval()  # Sets the model for evaluation.

    total = 0
    correct = 0

    with torch.no_grad():  # No need to calculate the gradients.

        for x, y in dataloader:
            output = model(x.to(config.DEVICE))  # model's output.

            total += y.size(0)
            predictions = output.argmax(dim=1).cpu().detach()
            correct += (predictions == y.cpu().detach()).sum().item()

            y_true_tensor = torch.cat((y_true_tensor, y.to(config.DEVICE)))
            y_pred_tensor = torch.cat((y_pred_tensor, predictions.to(config.DEVICE)))

    print(f'Accuracy on Test set = {100 * (correct / total):.6f}% [{correct}/{total}]')  # Prints the Accuracy.

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    y_true = y_true_tensor.tolist()
    y_pred = y_pred_tensor.tolist()
    matrix = confusion_matrix(y_true, y_pred)


    df_cm = pd.DataFrame(matrix, range(matrix.shape[0]), range(matrix.shape[0]))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

    plt.show()

    print(matrix)

    classify_report = classification_report(y_true, y_pred)
    print(classify_report)