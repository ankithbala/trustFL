import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tqdm.notebook import tqdm
import pickle


class FederatedNodeParallel:
    def __init__(
        self,
        node_id,
        device,
        train_dataset,
        test_dataset,
        batch_size=1000,
    ) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size)
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 56), nn.ReLU(), nn.Linear(56, 10)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.node_id = node_id
        self.device = device

    def train_epoch(self):
        losses = []
        for idx, (data_x, data_y) in enumerate(self.train_dataloader):
            output = self.network(data_x.to(self.device))
            self.optimizer.zero_grad()
            loss = nn.functional.mse_loss(output, data_y.to(self.device))
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        return sum(losses) / len(losses)

    def process(self, num_epochs=2):
        for i in range(num_epochs):
            self.train_epoch()
            print("Node: {} Epoch: {} done".format(self.node_id, i))


from PIL import Image


class MNISTFed(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None) -> None:
        data, targets = dataset
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.data[index]
        target = int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target)

    def __len__(self):
        return len(self.data)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
target_transform = lambda x: torch.nn.functional.one_hot(torch.tensor(x), 10).float()


def simulate_node(node_id, num_nodes):
    device = torch.device("cuda")

    with open("datasets/fedpkl/{}_train.pkl".format(node_id), "rb") as fp:
        train_dataset = MNISTFed(
            pickle.load(fp), transform=transform, target_transform=target_transform
        )
    with open("datasets/fedpkl/{}_test.pkl".format(node_id), "rb") as fp:
        test_dataset = MNISTFed(
            pickle.load(fp), transform=transform, target_transform=target_transform
        )
    node = FederatedNodeParallel(node_id, device, train_dataset, test_dataset)
    node.process()
