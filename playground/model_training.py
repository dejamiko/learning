# WRITE IN THE REPORT THAT I HAVE TOO LITTLE DATA TO TRAIN SOMETHING GOOD
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


class ImagePairDataset(Dataset):
    def __init__(self, dataframe, base_dir, transform=None):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name1 = os.path.join(
            self.base_dir, self.dataframe.iloc[idx, 1], "image_0.png"
        )
        img_name2 = os.path.join(
            self.base_dir, self.dataframe.iloc[idx, 2], "image_0.png"
        )
        image1 = Image.open(img_name1).convert("RGB")
        image2 = Image.open(img_name2).convert("RGB")
        similarity = self.dataframe.iloc[idx, 3]

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
                nn.Linear(512, 16), nn.ReLU(inplace=True), nn.Linear(16, 1)
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_extractor = nn.Sequential(conv, global_avg_pool)
            self.head = None

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


def prepare_data():
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    base_dir = "/"
    df = pd.read_csv(os.path.join(base_dir, "_data/similarity_df.csv"))
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
    num_epochs = 20
    criterion = nn.MSELoss()
    model = SiameseNetwork(backbone=backbone, frozen=frozen).to(device)
    optimizer = optim.AdamW(
        model.head.parameters() if frozen and backbone else model.parameters(), lr=0.001
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopping = EarlyStopping(patience=10, delta=0.01)

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


if __name__ == "__main__":
    config = {"frozen": False, "backbone": False}
    train_loader, val_loader = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loop(train_loader, val_loader, **config)
    print("Finished Training")
    model = SiameseNetwork(**config).to(device)
    model.eval()
    with torch.no_grad():
        img1, img2, labels = next(iter(val_loader))
        outputs = model(img1, img2).squeeze()
        for o, l in zip(outputs, labels):
            print(o, l)

    torch.save(model.state_dict(), "siamese_network.pth")
