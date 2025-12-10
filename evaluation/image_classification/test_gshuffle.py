import logging

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

from datasets_utils import CheXpertDataSet
from utils import GShuffle

trBatchSize = 16
class_names = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

checkpoint_path = (
        "/jet/home/lisun/work/xinliu/hi-ml/hi-ml-multimodal/src/"
        + "new_caches_v7/T0.1_L0.1_shuffle-temp0.07/bt50/cache-2023-11-25-22-45-13-moco/model_last.pth"
)

torch.manual_seed(0)

pathFileTrain = "/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/train_mod1.csv"
pathFileValid = "/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/valid_mod.csv"
pathFileTest = "/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/test_mod.csv"

log_fn = "output/pretrained/bt50/gshuffleT01_L01_shuffle-temp0_07_training_log.txt"
result_fn = "output/pretrained/bt50/gshuffleT01_L01_shuffle-temp0_07_test_results.csv"
ac_fn = "output/pretrained/bt50/gshuffle_T01_L01_shuffle-temp0_07_test_accuracies.txt"

# Set up logging
logging.basicConfig(
    filename=log_fn, level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GShuffle()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.0015, weight_decay=5e-4, momentum=0.9
)
print("checkpoint_path:", checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location=device)
msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
optimizer.load_state_dict(checkpoint["optimizer"])
# After loading the optimizer state dict
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

print(msg)
model = model.to(device)
model = model.image_encoder
model.classifier = model.create_downstream_classifier(
    classifier_hidden_dim=4, num_classes=2, num_tasks=5
).to(device)

criterion = torch.nn.CrossEntropyLoss()


def train_model(model, data_loader, optimizer, criterion, device) -> float:
    """
    Trains the model for one epoch on the given data loader.

    :param model: The model to train.
    :param data_loader: DataLoader with training data.
    :param optimizer: Optimizer to use for training.
    :param criterion: Loss function.
    :param device: Device to run the training on.
    :return: Average loss for this epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (images, labels, _) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images).class_logits.to(device)
        loss = 0.0

        # Calculate loss for each task
        for task in range(model.classifier.num_tasks):
            task_labels = labels[:, task].type(torch.LongTensor).to(device)
            task_outputs = outputs[:, :, task]
            loss += criterion(task_outputs, task_labels).to(device)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        if batch_idx % 300 == 299:  # Print every 100 mini-batches
            print(f"[{batch_idx + 1:5d}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

    return running_loss / len(data_loader)


def validate_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    task_correct = [0] * len(class_names)
    task_total = [0] * len(class_names)

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).class_logits.to(device)
            loss = sum(
                criterion(
                    outputs[:, :, task],
                    labels[:, task].type(torch.LongTensor).to(device),
                )
                for task in range(len(class_names))
            )
            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            for task in range(len(class_names)):
                _, predicted = torch.max(outputs[:, :, task], 1)
                task_labels = labels[:, task].type(torch.LongTensor).to(device)
                task_total[task] += task_labels.size(0)
                task_correct[task] += (predicted == task_labels).sum().item()

    epoch_loss = running_loss / len(data_loader.dataset)
    task_accuracy = [
        100 * task_correct[i] / task_total[i] for i in range(len(class_names))
    ]
    print(f"Validation loss: {epoch_loss:.3f}")
    for task_name, acc in zip(class_names, task_accuracy):
        print(f"{task_name} Validation Accuracy: {acc:.2f}%")
    return epoch_loss, task_accuracy


def evaluate_model(model, data_loader, device):
    test_header = [
        "Path",
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion",
    ]

    with open(result_fn, "w") as f:
        f.write(",".join(test_header) + "\n")
        num_tasks = 5
        model.eval()  # Set the model to evaluation mode
        task_correct = [0] * num_tasks
        task_total = [0] * num_tasks
        task_accuracy = [0] * num_tasks

        with torch.no_grad():  # No need to track gradients for evaluation
            for images, labels, image_names in data_loader:
                images, labels = images.to(device), labels.to(device)
                # labels = torch.max(labels, 1)[1]
                # Forward pass to get output/logits
                outputs = model(images).class_logits
                # For each task, calculate the accuracy
                for task in range(num_tasks):
                    # Get the predictions for the current task
                    _, predicted = torch.max(outputs[:, :, task], 1)

                    # Get the labels for the current task
                    task_labels = labels[:, task].type(torch.LongTensor).to(device)

                    # Update total and correct counts for the current task
                    task_total[task] += task_labels.size(0)
                    task_correct[task] += (predicted == task_labels).sum().item()

                # Calculate accuracy for each task
                for task in range(num_tasks):
                    task_accuracy[task] = 100 * task_correct[task] / task_total[task]

    return task_accuracy


def run():
    transforms = Compose([Resize(256), CenterCrop(224), ToTensor()])
    # Load dataset
    datasetTrain = CheXpertDataSet(pathFileTrain, transforms, policy="ones")
    print("Train data length:", len(datasetTrain))

    datasetValid = CheXpertDataSet(pathFileValid, transforms)
    print("Valid data length:", len(datasetValid))

    datasetTest = CheXpertDataSet(pathFileTest, transforms, policy="ones")
    print("Test data length:", len(datasetTest))

    dataLoaderTrain = DataLoader(
        dataset=datasetTrain,
        batch_size=trBatchSize,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    dataLoaderVal = DataLoader(
        dataset=datasetValid,
        batch_size=trBatchSize,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=2, pin_memory=True)

    # Training and validation loop
    num_epochs = 10  # Set the number of epochs
    for epoch in range(num_epochs):
        train_loss = train_model(model, dataLoaderTrain, optimizer, criterion, device)
        val_loss, val_accuracy = validate_model(model, dataLoaderVal, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info(f"Training Loss: {train_loss:.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f}")
        for task_name, acc in zip(class_names, val_accuracy):
            logging.info(f"{task_name} Validation Accuracy: {acc:.2f}%")

    # Testing loop
    test_accuracy = evaluate_model(model, dataLoaderTest, device)
    for i, acc in zip(class_names, test_accuracy):
        logging.info(f"Task {i} Accuracy: {acc:.2f}%")
        print(f"Task {i} Test Accuracy: {acc:.2f}%")

    # Save the test accuracies to a file
    with open(ac_fn, "w") as f:
        for task_name, acc in zip(class_names, test_accuracy):
            f.write(f"Task {task_name} Accuracy: {acc:.2f}%\n")


run()
