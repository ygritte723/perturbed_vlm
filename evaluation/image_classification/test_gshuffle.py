import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

# Add project root to sys.path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import config
from datasets_utils import CheXpertDataSet
from utils import GShuffle

def train_model(model, data_loader, optimizer, criterion, device) -> float:
    """
    Trains the model for one epoch on the given data loader.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels, _) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images).class_logits.to(device)
        loss = 0.0

        for task in range(model.classifier.num_tasks):
            task_labels = labels[:, task].type(torch.LongTensor).to(device)
            task_outputs = outputs[:, :, task]
            loss += criterion(task_outputs, task_labels).to(device)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print(f"[{batch_idx + 1:5d}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

    return running_loss / len(data_loader)


def validate_model(model, class_names, data_loader, criterion, device):
    model.eval()
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


def evaluate_model(model, result_fn, data_loader, device):
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
        model.eval()
        task_correct = [0] * num_tasks
        task_total = [0] * num_tasks
        task_accuracy = [0] * num_tasks

        with torch.no_grad():
            for images, labels, image_names in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).class_logits
                for task in range(num_tasks):
                    _, predicted = torch.max(outputs[:, :, task], 1)
                    task_labels = labels[:, task].type(torch.LongTensor).to(device)
                    task_total[task] += task_labels.size(0)
                    task_correct[task] += (predicted == task_labels).sum().item()

                for task in range(num_tasks):
                    task_accuracy[task] = 100 * task_correct[task] / task_total[task]

    return task_accuracy


def main():
    trBatchSize = 16
    class_names = [
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion",
    ]
    
    # Check if config paths are set, otherwise fallback or warn
    if not hasattr(config, 'CHEXPERT_TRAIN_CSV'):
        print("Warning: CHEXPERT_TRAIN_CSV not in config. Using hardcoded/placeholder.")
        pathFileTrain = "/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/train_mod1.csv"
        pathFileValid = "/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/valid_mod.csv"
        pathFileTest = "/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/test_mod.csv"
    else:
        pathFileTrain = config.CHEXPERT_TRAIN_CSV
        pathFileValid = config.CHEXPERT_VALID_CSV
        pathFileTest = config.CHEXPERT_TEST_CSV

    log_fn = "output/pretrained/bt50/gshuffleT01_L01_shuffle-temp0_07_training_log.txt"
    result_fn = "output/pretrained/bt50/gshuffleT01_L01_shuffle-temp0_07_test_results.csv"
    ac_fn = "output/pretrained/bt50/gshuffle_T01_L01_shuffle-temp0_07_test_accuracies.txt"

    # Ensure output directories exist
    os.makedirs(os.path.dirname(log_fn), exist_ok=True)

    logging.basicConfig(
        filename=log_fn, level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s"
    )

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GShuffle()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.0015, weight_decay=5e-4, momentum=0.9
    )

    # Use config for checkpoint path if available, or keep hardcoded one with warning/check
    checkpoint_path = config.CHECKPOINT_PATH_OUR # Assuming this matches the intent
    print("checkpoint_path:", checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(msg)
        # Attempt to load optimizer state if it matches
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception as e:
            print(f"Could not load optimizer state: {e}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch or pretrained initialization.")

    model = model.to(device)
    # We are using the image encoder part for classification
    model = model.image_encoder
    model.classifier = model.create_downstream_classifier(
        classifier_hidden_dim=4, num_classes=2, num_tasks=5
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    transforms = Compose([Resize(256), CenterCrop(224), ToTensor()])
    
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

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(model, dataLoaderTrain, optimizer, criterion, device)
        val_loss, val_accuracy = validate_model(model, class_names, dataLoaderVal, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info(f"Training Loss: {train_loss:.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f}")
        for task_name, acc in zip(class_names, val_accuracy):
            logging.info(f"{task_name} Validation Accuracy: {acc:.2f}%")

    test_accuracy = evaluate_model(model, result_fn, dataLoaderTest, device)
    for i, acc in zip(class_names, test_accuracy):
        logging.info(f"Task {i} Accuracy: {acc:.2f}%")
        print(f"Task {i} Test Accuracy: {acc:.2f}%")

    with open(ac_fn, "w") as f:
        for task_name, acc in zip(class_names, test_accuracy):
            f.write(f"Task {task_name} Accuracy: {acc:.2f}%\n")


if __name__ == "__main__":
    main()
