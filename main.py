import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm
import pandas as pd


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, i):
        seq = self.data[i : i + self.seq_len]
        label = self.data[i + self.seq_len, -2]
        return {
            "seq": seq.clone().detach().float(),
            "label": label.clone().detach().float().unsqueeze(0),
        }


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=10):
        super(Model, self).__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.4,
        )
        self.fc = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, output_dim)
        self.h0 = None
        self.c0 = None

    def forward(self, x):
        # print(x.shape)
        if (
            self.h0 is None
            or self.c0 is None
            or self.h0.shape[1] != x.shape[0]
            or self.c0.shape[1] != x.shape[0]
        ):
            self.h0 = torch.zeros(
                self.rnn.num_layers, x.size(0), self.rnn.hidden_size
            ).to(x.device)
            self.c0 = torch.zeros(
                self.rnn.num_layers, x.size(0), self.rnn.hidden_size
            ).to(x.device)

        out, (self.h0, self.c0) = self.rnn(x, (self.h0, self.c0))

        # Detach hidden states from the computation graph to prevent backpropagation
        self.h0 = self.h0.detach()
        self.c0 = self.c0.detach()
        out = self.fc(out[:, -1, :])
        out = self.fc2(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(input_dim=6, hidden_dim=32, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-5)

data = {}

for csv_file in os.listdir("./data"):
    if not csv_file.endswith(".csv"):
        continue
    print(f"Processing {csv_file}")
    file = open(f"./data/{csv_file}", "r")

    stock_name = csv_file.split(".")[0]
    df = pd.read_csv(
        file,
        header=1,
        names=["ticker", "date", "open", "high", "low", "close", "volume"],
    )
    df = df.sort_values(by="date", ascending=True)

    data[stock_name] = torch.tensor(df.iloc[:, 1:].values)

    file.close()


dataset = TimeSeriesDataset(data["BID"], seq_len=20)
print(len(dataset))
merged_dataset = dataset
for stock_name in data:
    if stock_name == "BID":
        continue
    merged_dataset = ConcatDataset(
        [merged_dataset, TimeSeriesDataset(data[stock_name], seq_len=20)]
    )
print(len(merged_dataset))
train_dataset, test_dataset = random_split(merged_dataset, [0.8, 0.2])
torch.set_printoptions(sci_mode=False)
print(len(train_dataset), len(test_dataset))
dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=512, shuffle=True)
# assert False
for epoch in range(100):
    running_loss = 0.0
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        seq = batch["seq"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad()
        out = model(seq)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss /= i + 1
    model.eval()
    with torch.inference_mode():

        testing_loss = 0.0
        for i, batch in enumerate(testloader):
            seq = batch["seq"].to(device)
            label = batch["label"].to(device)
            out = model(seq)
            loss = criterion(out, label)
            testing_loss += loss.item()

        print(
            f"Epochs: {epoch+1:3d} train: {running_loss :.3f}| test: {testing_loss / (i+1):.3f}"
        )

    model.train()
