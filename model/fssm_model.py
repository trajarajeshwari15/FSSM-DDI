import torch
import torch.nn as nn


class SelectiveSSM(nn.Module):
    """
    CPU-compatible Selective State Space block.
    Input:
        [batch, sequence_length, d_model]
    Output:
        [batch, sequence_length, d_model]
    """

    def __init__(self, d_model, state_size=16):
        super().__init__()

        self.state_size = state_size

        self.input_projection = nn.Linear(
            d_model,
            state_size
        )

        self.delta_projection = nn.Linear(
            d_model,
            state_size
        )

        self.output_projection = nn.Linear(
            state_size,
            d_model
        )

        self.activation = nn.SiLU()

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        state = torch.zeros(
            batch_size,
            self.state_size,
            device=x.device
        )

        outputs = []

        for t in range(sequence_length):

            current = x[:, t, :]

            update = self.activation(
                self.input_projection(current)
            )

            delta = torch.sigmoid(
                self.delta_projection(current)
            )

            state = (
                (1 - delta) * state
                + delta * update
            )

            output = self.output_projection(state)

            outputs.append(output)

        return torch.stack(
            outputs,
            dim=1
        )


class ISF(nn.Module):
    """
    Interaction-based Selective Filtering.
    """

    def __init__(self, d_model):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):

        weights = self.gate(x)

        return x * weights


class FSSMModel(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=300,
        state_size=16,
        num_classes=5
    ):
        super().__init__()

        # 768-D PubMedBERT → 300-D representation
        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.ssm = SelectiveSSM(
            hidden_dim,
            state_size
        )

        self.isf = ISF(
            hidden_dim
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # DDI classifier
        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):

        # 768 → 300
        x = self.input_projection(x)

        # If input is [batch, 300],
        # create sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Selective State Space
        x = self.ssm(x)

        # ISF
        x = self.isf(x)

        # Feature fusion
        x = self.fusion(x)

        # Global representation
        x = torch.mean(
            x,
            dim=1
        )

        # Classification
        logits = self.classifier(x)

        return logits