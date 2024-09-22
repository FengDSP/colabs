import time
import os
import threading
import time

import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from torch import nn
import pandas as pd
# from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as multiprocessing

from mup import MuSGD, get_shapes, set_base_shapes, make_base_shapes, MuReadout


batch_size = 64
epochs = 20
log_interval = 300
data_dir = "/tmp"
base_shapes_path = "./demo_width256.bsh"


def train(
    model,
    *,
    device,
    train_loader,
    optimizer,
    train_name,
    scheduler=None,
):
    model.train()
    train_loss = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))

        loss = F.cross_entropy(output, target)
        loss.backward()
        train_loss += loss.item() * data.shape[0]  # sum up batch loss
        optimizer.step()
        if batch_idx % log_interval == log_interval - 1:
            elapsed = time.time() - start_time
            passed_data = batch_idx * len(data)
            passed_percent = 100.0 * batch_idx / len(train_loader)
            print(
                f"Train {train_name}: [{passed_data}/{len(train_loader.dataset)}"
                + f" ({passed_percent:.0f}%)]\tLoss: {loss.item():.6f} | ms/batch {elapsed * 1000 / log_interval:5.2f}"
            )
            start_time = time.time()
        if scheduler is not None:
            scheduler.step()
    train_loss /= len(train_loader.dataset)
    print(f"Train {train_name}: Average loss: {train_loss:.4f}")
    return train_loss


class MLP(nn.Module):
    def __init__(self, width=128, num_classes=10, output_mult=1.0, input_mult=1.0):
        super(MLP, self).__init__()
        self.nonlin = nn.ReLU()
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode="fan_in")
        self.fc_1.weight.data /= self.input_mult**0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode="fan_in")
        nn.init.zeros_(self.fc_3.weight)

    def forward(self, x):
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out) * self.output_mult


class muMLP(nn.Module):
    def __init__(self, width=128, num_classes=10, output_mult=1.0, input_mult=1.0):
        super(muMLP, self).__init__()
        self.nonlin = nn.ReLU()
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = MuReadout(
            width, num_classes, bias=False, output_mult=self.output_mult
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode="fan_in")
        # self.fc_1.weight.data /= self.input_mult**0.5  # Why this doesn't work in multi processing?
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode="fan_in")
        nn.init.zeros_(self.fc_3.weight)

    def forward(self, x):
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out)


def train_with(width, log2lr, epochs=20, mup=True):
    torch.manual_seed(1)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    # testset = datasets.CIFAR10(
    #     root=data_dir, train=False, download=True, transform=transform
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     testset, batch_size=batch_size, shuffle=False, num_workers=2
    #     )

    # device = torch.device("cuda")  # changing to the explicit device set solves the OOM issue somehow.
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    train_name = f"width={width} log2lr={log2lr}"
    print(f"Training starting {train_name}")
    res = []
    try:
        if mup:
            mlp_model = muMLP(width=width).to(device)
            set_base_shapes(mlp_model, base_shapes_path)
            optimizer_class = MuSGD
        else:
            mlp_model = MLP(width=width).to(device)
            optimizer_class = torch.optim.SGD
        for epoch in range(epochs):
            train_loss = train(
                mlp_model,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer_class(mlp_model.parameters(), lr=2**log2lr),
                train_name=f"{train_name} epoch={epoch}",
            )
            res.append(
                {
                    "width": width,
                    "log2lr": log2lr,
                    "epoch": epoch,
                    "train_loss": train_loss,
                }
            )
    except Exception as e:
        print(f"Exception catched {train_name}: {e}")
        raise

    print(f"Training finished. {train_name}")
    
    df = pd.DataFrame(res)
    output_file = "cifar10_mup.jsonl" if mup else "cifar10_std.jsonl"
    df.to_json(output_file, orient="records", lines=True, mode='a')


def main():
    base_shapes = get_shapes(MLP(width=256))
    # just need to change whatever dimension(s) we are scaling
    delta_shapes = get_shapes(MLP(width=256 + 1))
    make_base_shapes(base_shapes, delta_shapes, savefile=base_shapes_path)

    print("Running trainers ...")
    processes = []
    for width in [8192, 4096]:  # [2048, 1024]:  #[512, 256]:  # [128, 64]:
        for log2lr in np.linspace(-8, -1, 8):
            proc = multiprocessing.Process(target=train_with, args=(width, log2lr), daemon=True)
            proc.start()
            processes.append(proc)
    print("Waiting for results ...")
    for proc in processes:
        proc.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
