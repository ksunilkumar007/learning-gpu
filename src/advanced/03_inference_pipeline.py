import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import time

device = torch.device("cuda")

# --Architecture ----------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# --Training Pipeline -----------
class MNISTTrainingPipeline:
    def __init__(self, num_classes=10, lr=0.001):
        self.model = CNN(num_classes=num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([transforms.ToTensor()])
        print(f"Training pipeline ready | device: {device}")

    def train(self, epochs=5):
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                pred = self.model(images)
                loss = self.loss_fn(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

# --Inference Pipeline------
class MNISTInferencePipeline:
    def __init__(self, model_path):
        self.model = CNN().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        print(f"Inference pipeline ready | device: {device}")

    def predict(self, image):
        tensor = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = probabilities.max(dim=1)
        return {
            "digit": predicted.item(),
            "confidence": confidence.item(),
            "all_probs": probabilities.squeeze().tolist()
        }

    def predict_batch(self, images):
        tensors = torch.stack([self.transform(img) for img in images]).to(device)
        with torch.no_grad():
            outputs = self.model(tensors)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(dim=1)
        return [
            {"digit": d.item(), "confidence": c.item()}
            for d, c in zip(predicted, confidences)
        ]

# --- Main --------
if __name__ == "__main__":
    # 1. Train and save
    print("=== Training Pipeline ===")
    trainer = MNISTTrainingPipeline(lr=0.001)
    trainer.train(epochs=3)
    trainer.save("models/mnist_cnn.pth")

    # 2. Load and predict
    print("\n=== Inference Pipeline ===")
    pipeline = MNISTInferencePipeline("models/mnist_cnn.pth")
    test_data = datasets.MNIST(root="data", train=False, download=True)

    print("\nSingle image predictions:")
    for i in range(5):
        image, true_label = test_data[i]
        result = pipeline.predict(image)
        status = "✓" if result["digit"] == true_label else "✗"
        print(f"  {status} True: {true_label} | Predicted: {result['digit']} | Confidence: {result['confidence']:.2%}")

    print("\nBatch prediction (10 images at once):")
    images = [test_data[i][0] for i in range(10)]
    labels = [test_data[i][1] for i in range(10)]
    start = time.time()
    results = pipeline.predict_batch(images)
    elapsed = (time.time() - start) * 1000
    correct = sum(r["digit"] == l for r, l in zip(results, labels))
    print(f"  Accuracy: {correct}/10")
    print(f"  Batch time: {elapsed:.2f} ms")
    print(f"  Per image: {elapsed/10:.2f} ms")
