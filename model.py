# Check notebook/clamp_model.ipynb for more details & a more robust implementation.

import torch
import clip
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from cgcnn.data import CIFData, get_train_val_test_loader, collate_pool
from cgcnn.model import CrystalGraphConvNet
import os


class CLAMP(nn.Module):
    def __init__(self, cgcnn_hidden_dim, clip_out_dim, out_dim, device):
        """
        Initialize the CLAMP network.

        Args:
        - cgcnn_hidden_dim (int): Hidden dimension size from the CGCNN model.
        - clip_out_dim (int): Output dimension size from the CLIP text encoder (text embed dimension).
        - out_dim (int): Output dimension size of the final layer.
        """
        super(CLAMP, self).__init__()
        self.cgcnn_hidden_dim = cgcnn_hidden_dim
        self.clip_out_dim = clip_out_dim
        self.out_dim = out_dim
        self.device = device

        # Load the CLIP text encoder model
        self.clip_model, _ = clip.load("RN50", device=device)

        # Initialize the CGCNN model
        # Let's make some assumptions: n_conv=3, n_h=2
        # 'atom_fea_len' and 'h_fea_len' should be set according to your dataset specifics and graph model
        atom_fea_len = 92  # example value, set according to your data
        h_fea_len = 128    # example value, set your own hyperparameter

        self.cgcnn_model = CrystalGraphConvNet(atom_fea_len=atom_fea_len,
                                               h_fea_len=h_fea_len,
                                               n_conv=3,
                                               n_h=2,
                                               classification=False)

        # Combine representations from CLIP and CGCNN using a fully connected layer
        self.fc = nn.Linear(cgcnn_hidden_dim + clip_out_dim, out_dim)

    def forward(self, crystal_inputs, textual_inputs):
        """
        Forward pass through the CLAMP network.

        Args:
        - crystal_inputs (tuple): Inputs for the CGCNN model.
        - textual_inputs (list of str): List of textual descriptions for the CLIP.
        """
        # Get graph-based features from CGCNN
        cgcnn_outputs = self.cgcnn_model(*crystal_inputs)

        # Tokenize the textual inputs and get text-based features from CLIP
        text_tokens = clip.tokenize(textual_inputs).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens).float()

        # Concatenate the CGCNN outputs and CLIP text features
        combined_features = torch.cat((cgcnn_outputs, text_features), dim=1)

        # Pass the combined features through a fully connected layer to obtain final outputs
        outputs = self.fc(combined_features)
        return outputs


# Configuration options
device = "cpu"  # or "cuda" or "mps"
cgcnn_hidden_dim = 128  # Example value; set according to your CGCNN setup
clip_out_dim = 512  # Example value; set according to CLIP's output dimension
out_dim = 1  # or the number of classes for a classification task
epochs = 10
dataset_path = "./cifs/"
batch_size = 1

# Load your dataset
data = CIFData(dataset_path)
train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset=data,
    batch_size=batch_size,
    collate_fn=collate_pool,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    return_test=True
)

# Initialize the CLAMP model
clamp_model = CLAMP(cgcnn_hidden_dim=cgcnn_hidden_dim,
                    clip_out_dim=clip_out_dim, out_dim=out_dim, device=device).to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # or CrossEntropyLoss for classification
optimizer = optim.Adam(clamp_model.parameters(), lr=0.001)

# Define the training loop


def train():
    clamp_model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            # Process inputs
            crystal_inputs = (Variable(inputs[0].to(device)),
                              Variable(inputs[1].to(device)),
                              inputs[2].to(device),
                              [crys_idx.to(device) for crys_idx in inputs[3]])

            # Forward pass through the CLAMP network
            outputs = clamp_model(crystal_inputs, targets)

            # Compute loss
            loss = criterion(outputs, torch.Tensor([[t] for t in targets]).to(
                device))  # Assuming targets are a list of scalar values

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 100 steps
            if batch_idx % 100 == 0:
                print(
                    f"Epoch: {epoch}, Step: {batch_idx}, Loss: {loss.item()}")

        print(f"Epoch {epoch} completed. Loss: {loss.item()}")


# Trigger the training process
if __name__ == '__main__':
    train()

    # Optionally save the trained model
    torch.save(clamp_model.state_dict(), "clamp_model.pth")
