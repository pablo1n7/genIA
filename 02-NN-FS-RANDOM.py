import os
import click
import sh
import shap
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


class MLPModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            # weight_norm(nn.Linear(num_features, 1024)),
            nn.Linear(num_features, 1024),
            # nn.ELU(),
            # weight_norm(nn.Linear(1024, 256)),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            # nn.ELU(),
            # weight_norm(nn.Linear(128, 64)),
            nn.Linear(128, 64),
            nn.ELU(),
            # weight_norm(nn.Linear(64, 64)),
            nn.Linear(64, 64),
            nn.ELU(),
            # weight_norm(nn.Linear(64, 64)),
            nn.Linear(64, 64),
            nn.ELU(),
            # weight_norm(nn.Linear(64, 1)),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        return torch.clamp(self.model(input_tensor), 0, 1)


def prepare_data(caso: str):

    geneticos = pd.read_csv(f"{caso}/genotype_sim.raw", sep="\t")
    gen = np.array(geneticos[list(geneticos.columns)[6:]])
    feno = pd.read_csv(f"{caso}/fenodata.txt", sep=" ")["bmi"]

    np.save(f"{caso}/x.npy", gen)
    np.save(f"{caso}/iids_all.npy", np.array(geneticos["IID"]))
    np.save(f"{caso}/feno_all.npy", np.array(feno))
    np.save(f"{caso}/feno_all_c.npy", np.array(feno))
    np.save(f"{caso}/columns.npy", np.array(list(geneticos.columns)[6:]))


@click.command()
@click.option("--caso", default="data", help="Specify data location")
@click.option("--ivalue", default=0, type=int, help="Number of iterations")
def main(caso: str, ivalue: int):

    # Prepara data
    # .raw,.txt -> .npy
    prepare_data(caso)

    X = np.load(f"{caso}/x.npy")
    columns = np.load(f"{caso}/columns.npy", allow_pickle=True)
    X_df = pd.DataFrame(X, columns=columns)

    print("DATA A UTILIZAR", X_df.shape)
    X = np.array(X_df)
    columns = X_df.columns
    y = np.load(f"{caso}/feno_all_c.npy")

    if ivalue != 0:
        print("CASO", caso, "- SHUFFLE DATA - ", ivalue)
        np.random.shuffle(y)  # Esto baraja las filas de y directamente
    else:
        print("CASO", caso, "- REAL DATA - ", ivalue)

    caso_random = caso + f"/rand{ivalue}"
    # caso_random = '/media/disk/pablo/random/BMI' + f'/rand{ivalue}'

    sh.mkdir("-p", caso_random)

    np.save(f"{caso_random}/y_random", y)

    y = y == 1
    y = np.array(y, dtype=int)

    print(y[y == 0].shape, y[y == 1].shape)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    DEVICE = "cuda"
    model = MLPModel(len(columns)).to(DEVICE)

    train_ds = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    test_ds = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loader_train = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)

    loader_test = DataLoader(test_ds, batch_size=16, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001) #virexperiment
    loss_fn = nn.BCELoss()

    for epoch in range(30):
        # for epoch in range(150): #virexperiment
        running_loss = 0
        model.train()
        for i, (_X, labels) in enumerate(loader_train):
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model.forward(torch.Tensor(_X).to(DEVICE))

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels.unsqueeze(-1).to(DEVICE))
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():

            running_loss_val = 0
            for i, (_X, labels) in enumerate(loader_test):
                # Make predictions for this batch
                outputs = model.forward(torch.Tensor(_X).to(DEVICE))

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels.unsqueeze(-1).to(DEVICE))
                running_loss_val += loss.item()

        print(
            "EPOCH {}: train loss {}, val loss {}".format(
                epoch + 1, running_loss / 16, running_loss_val / 16
            )
        )

    del outputs, loss, loader_test, loader_train, optimizer

    DEVICE = "cuda"

    e = shap.DeepExplainer(
        model.to(DEVICE), torch.from_numpy(np.array(X_test)).to(DEVICE).float()
    )

    shap_values = e.shap_values(torch.from_numpy(X_test).to(DEVICE).float())

    print("*******************")

    np.save(f"{caso_random}/nfi_values", shap_values)

    df = pd.DataFrame(
        {
            "mean_abs_shap": np.mean(np.abs(shap_values).reshape(-1), axis=0),
            "stdev_abs_shap": np.std(np.abs(shap_values).reshape(-1), axis=0),
            "mean_shap": np.mean(shap_values.reshape(-1), axis=0),
            "name": columns,
        }
    )

    df.sort_values("mean_abs_shap", ascending=False)[:]

    np.save(
        f"{caso_random}/gene_v2.npy",
        np.array(df.sort_values("mean_abs_shap", ascending=False)["name"]),
    )

    np.save(
        f"{caso_random}/data_gene.npy",
        np.array(df.sort_values("mean_abs_shap", ascending=False)),
    )


if __name__ == "__main__":
    main()
