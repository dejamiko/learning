import os
import shutil

import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from config import Config
from playground.extractor import Extractor
from tm_utils import ImagePreprocessing


class ImagePairDataset(Dataset):
    def __init__(self, dataframe, base_dir, transform=None):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.transform = transform
        self.counter = 0

    def __len__(self):
        return len(self.dataframe) * 5

    def __getitem__(self, idx):
        img_name1 = os.path.join(
            self.base_dir, self.dataframe.iloc[idx // 5, 1], f"image_{self.counter}.png"
        )
        img_name2 = os.path.join(
            self.base_dir, self.dataframe.iloc[idx // 5, 2], f"image_{self.counter}.png"
        )
        image1 = Image.open(img_name1).convert("RGB")
        image2 = Image.open(img_name2).convert("RGB")
        similarity = self.dataframe.iloc[idx // 5, 3]

        self.counter += 1
        if self.counter == 5:
            self.counter = 0

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(similarity, dtype=torch.float)


class SiameseNetwork(nn.Module):
    def __init__(self, backbone, frozen):
        super(SiameseNetwork, self).__init__()
        if backbone:
            resnet = models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            )
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(nn.LazyLinear(128))

        if frozen:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        if self.head is not None:
            x = self.head(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return F.cosine_similarity(output1, output2)


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), "checkpoint.pt")
        self.best_loss = val_loss


def generate_training_images(preprocessing_steps, training_data_dir):
    # create a training_data directory and populate it with preprocessed images
    config = Config()
    config.IMAGE_PREPROCESSING = preprocessing_steps
    parent = "_data/training_data"
    os.makedirs(training_data_dir)
    for d in os.listdir(parent):
        if not os.path.isdir(os.path.join(parent, d)):
            continue
        if d == "siamese_similarities":
            continue
        ims = Extractor._load_images(os.path.join(parent, d), config)
        os.makedirs(os.path.join(training_data_dir, d))
        for im, fn in ims:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(training_data_dir, d, fn), im)


def prepare_data(preprocessing_steps, training_data_dir):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    base_dir = os.getcwd()
    df = pd.read_csv(os.path.join(base_dir, "_data/training_data/similarity_df.csv"))
    generate_training_images(preprocessing_steps, training_data_dir)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = ImagePairDataset(
        dataframe=train_df, base_dir=base_dir, transform=transform
    )
    val_dataset = ImagePairDataset(
        dataframe=val_df, base_dir=base_dir, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader


def training_loop(train_loader, val_loader, frozen, backbone):
    num_epochs = 200
    criterion = nn.MSELoss()
    model = SiameseNetwork(backbone=backbone, frozen=frozen).to(device)
    optimizer = optim.AdamW(
        model.head.parameters() if frozen and backbone else model.parameters(), lr=0.01
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopping = EarlyStopping(patience=20, delta=0.0001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                outputs = model(img1, img2).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load("checkpoint.pt"))
    os.remove("checkpoint.pt")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processing_steps_to_try = [
        [],
        [ImagePreprocessing.GREYSCALE],
        [ImagePreprocessing.BACKGROUND_REM],
        [ImagePreprocessing.CROPPING],
        [ImagePreprocessing.SEGMENTATION],
        [ImagePreprocessing.CROPPING, ImagePreprocessing.BACKGROUND_REM],
        [ImagePreprocessing.CROPPING, ImagePreprocessing.GREYSCALE],
        [
            ImagePreprocessing.CROPPING,
            ImagePreprocessing.BACKGROUND_REM,
            ImagePreprocessing.GREYSCALE,
        ],
    ]
    training_data_dir = "_training_data"
    if os.path.exists(training_data_dir):
        shutil.rmtree(training_data_dir)
    for ps in processing_steps_to_try:
        train_loader, val_loader = prepare_data(ps, training_data_dir)

        model_config = {"frozen": False, "backbone": False}
        model = training_loop(train_loader, val_loader, **model_config)
        torch.save(model.state_dict(), f"optim/models/siamese_net_trained_{ps}.pth")

        model_config = {"frozen": True, "backbone": True}
        model = training_loop(train_loader, val_loader, **model_config)
        torch.save(
            model.state_dict(), f"optim/models/siamese_net_linearly_probed_{ps}.pth"
        )

        model_config = {"frozen": False, "backbone": True}
        model = training_loop(train_loader, val_loader, **model_config)
        torch.save(model.state_dict(), f"optim/models/siamese_net_fine_tuned_{ps}.pth")

        shutil.rmtree(training_data_dir)
