from os import path
import click
import sh
import shutil
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

import tqdm
import numpy as np


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


_BATCH_SIZE = 16
_LR = 0.00001
_N_EPOCHS = 30
_DEVICE = "cuda"


@click.command()
@click.option("--caso", default="data", help="Specify data location")
@click.option("--niters", default=1, type=int, help="Number of iterations")
@click.option("--device", default=_DEVICE, type=str, help="Device to run training on")
@click.option(
    "--batch-size", default=_BATCH_SIZE, type=int, help="Batch size for model training"
)
@click.option("--lr", default=_LR, type=float, help="Learning rate for model training")
@click.option(
    "--n-epochs",
    default=_N_EPOCHS,
    type=int,
    help="Number for epochs for model training",
)
def main(
    caso: str, niters: int, device: str, batch_size: str, lr: float, n_epochs: int
):

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

    for ivalue in range(niters):
        if ivalue == 0:
            print("CASO", caso, "- REAL DATA - ", ivalue)
        else:
            print("CASO", caso, "- SHUFFLE DATA - ", ivalue)
            np.random.shuffle(y)
        tmp_folder = path.join(caso, ".tmp_case")
        caso_random = path.join(tmp_folder, f"rand{ivalue}")

        sh.mkdir("-p", caso_random)

        np.save(path.join(caso_random, "y_random"), y)

        y = y == 1
        y = np.array(y, dtype=int)

        print(y[y == 0].shape, y[y == 1].shape)

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        model = MLPModel(len(columns)).to(_DEVICE)

        train_ds = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        test_ds = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

        loader_train = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )

        loader_test = DataLoader(
            test_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        epoch_bar = tqdm.tqdm(range(1, n_epochs + 1), desc="Training", unit="epoch")
        for epoch in epoch_bar:
            running_loss = 0
            model.train()
            for i, (_X, labels) in enumerate(loader_train):
                optimizer.zero_grad()

                outputs = model.forward(torch.Tensor(_X).to(device))
                loss = loss_fn(outputs, labels.unsqueeze(-1).to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            with torch.no_grad():
                running_loss_val = 0
                for i, (_X, labels) in enumerate(loader_test):
                    outputs = model.forward(torch.Tensor(_X).to(device))
                    loss = loss_fn(outputs, labels.unsqueeze(-1).to(device))
                    running_loss_val += loss.item()

            epoch_bar.set_description(f"Epoch {epoch}/{n_epochs}")
            epoch_bar.set_postfix(
                train_loss=f"{running_loss / batch_size:.5f}",
                test_loss=f"{running_loss_val / batch_size:.5f}",
            )

        del outputs, loss, loader_test, loader_train, optimizer

        e = shap.DeepExplainer(
            model.to(device), torch.from_numpy(np.array(X_test)).to(device).float()
        )

        shap_values = e.shap_values(torch.from_numpy(X_test).to(device).float())

        print("*******************")

        np.save(path.join(f"{caso_random}", "nfi_values.npy"), shap_values)
        df = pd.DataFrame(
            {
                "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
                "stdev_abs_shap": np.std(np.abs(shap_values), axis=0),
                "mean_shap": np.mean(shap_values.reshape(-1), axis=0),
                "name": columns,
            }
        )

        np.save(
            f"{caso_random}/gene_v2.npy",
            np.array(df.sort_values("mean_abs_shap", ascending=False)["name"]),
        )

        np.save(
            path.join(f"{caso_random}", "data_gene.npy"),
            np.array(df.sort_values("mean_abs_shap", ascending=False)),
        )

    if niters > 1:
        ## NULL DISTRIBUTION
        stack_exp = []
        for i in tqdm.tqdm(range(1, niters)):
            shap_values_rand = np.load(
                path.join(tmp_folder, f"rand{i}", "nfi_values.npy")
            )
            mean_abs_rand = np.abs(shap_values_rand).mean(axis=0)
            stack_exp.append(mean_abs_rand)

        null_distribution = np.vstack(stack_exp)

        np.save(path.join(tmp_folder, "null_distribution.npy"), null_distribution)

        ## Empirical P

        shap_values = np.load(path.join(tmp_folder, "rand0", "nfi_values.npy"))
        null_distribution = np.load(path.join(tmp_folder, "null_distribution.npy"))
        mean_abs_shap_real = np.abs(shap_values).mean(axis=0)

        n_permutaciones = null_distribution.shape[0]
        p_values = (np.sum(null_distribution >= mean_abs_shap_real, axis=0) + 1) / (
            n_permutaciones + 1
        )
        np.save(path.join(tmp_folder, "p_values_emp.npy"), p_values)
    columns = np.load(path.join(caso, "columns.npy"))
    nfi = np.load(path.join(tmp_folder, "rand0", "nfi_values.npy"))

    import ipdb

    # ipdb.set_trace()
    df = pd.DataFrame(
        {
            "mean_abs_shap": np.mean(np.abs(nfi), axis=0),
            "stdev_abs_shap": np.std(np.abs(nfi), axis=0),
            "mean_shap": np.mean(nfi.reshape(-1), axis=0),
            "snp": columns,
        }
    )

    if niters == 1:
        df["p_emp"] = np.nan
        df = df.sort_values(by="mean_abs_shap", ascending=False)
    else:
        df["p_emp"] = np.load(
            path.join(tmp_folder, "p_values_emp.npy"), allow_pickle=True
        )
        df = df.sort_values(by="p_emp", ascending=True)
    # RESULTADO FINAL FINAL
    df[
        [
            "snp",
            "p_emp",
            "mean_abs_shap",
            "stdev_abs_shap",
            "mean_shap",
        ]
    ].to_csv(path.join(caso, f"{caso}_output.csv"), index=False)

    # Limpiar carpeta temporal de laburo
    shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    main()
