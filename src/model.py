# src/model.py
import torch
import torch.nn as nn

class sCIFAR10_GRU(nn.Module):
    """
    input = 1, hidden=181, classes=10
    GRU params: 3*181*(1+181) + 6*181 = 98,826 + 1,086 = 99,912
    Linear params: 181*10 + 10 = 1,820
    Total params: 101,732
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 181, num_classes: int = 10, ablate_biases: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.ablate_biases = ablate_biases

        # just using a single layer GRU
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Come back to this here.
        """
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:
                # prevents grad problems by ensuring eigenvalues are ~1
                # when multiplying between hidden layers
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                # use xaiver (glorot) init
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0.)

                if not self.ablate_biases:
                    # 
                    update_bias_start = self.hidden_size
                    update_bias_end = 2 * self.hidden_size
                    param.data[update_bias_start:update_bias_end].fill_(1.)
        
        if self.ablate_biases:
            self._apply_bias_ablation_hooks()
        
    def _apply_bias_ablation_hooks(self):
        """
        
        """
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                def hook(grad):
                    grad_clone = grad.clone()
                    grad_clone[:2*self.hidden_size] = 0.
                    return grad_clone
                param.register_hook(hook)
    
    def forward(self, x):
        # x shape: (Batch, Seq_len, In_dim) = (B, 1024, 1)
        out, hidden = self.gru(x)
        final_state = hidden.squeeze(0)
        logits = self.fc(final_state)
        return logits