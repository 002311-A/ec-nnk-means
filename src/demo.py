import torch
import numpy as np

from NNKMU import NNKMU

cuda_available = torch.cuda.is_available()
device = torch.device("cuda") if cuda_available else -1

epochs = 10 
atoms = 1500
sparsity = 20
seed = 145
ep = 0.001 # entropy parameter
metric = 'error' # anomaly detection metric

def load_data():
    splits = ['0','1','2','3']
    dataset = {}

    for split in splits:
        dataset[split] = torch.load(f"bin/agnews/agnews_{split}.pt")

    X_train = []

    # Loop through each group
    for category in splits:

        data = dataset[category]
        X_train.extend(data)

        ## Reduce dataset size to 10,000 random samples for quick computation
        if len(X_train) > 10000:
            idxs = torch.randperm(len(X_train))[:10000]
            X_train = [X_train[i] for i in idxs]

    # Convert list to torch tensors
    X_train = torch.squeeze(torch.stack(X_train))

    return X_train.numpy()

data = load_data()

model = NNKMU(num_epochs=epochs, metric=metric, n_components=atoms, ep=ep, weighted=False, num_warmup=1)
model.fit(data)

codes = model.get_codes(data)

print(codes.shape)