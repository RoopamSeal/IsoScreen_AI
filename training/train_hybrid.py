import torch
from torch.utils.data import DataLoader
from models.hybrid_model import HybridADMETModel


def train(model, dataloader):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(20):

        for batch in dataloader:

            gnn, fp, transformer, labels = batch

            preds = model(gnn, fp, transformer)

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print("epoch complete")
