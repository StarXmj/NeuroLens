# engine/network.py
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetworkEngine:
    def __init__(self):
        self.model = None

    def build_model(self, layers_config, input_dim):
        layers_list = []
        current_dim = input_dim

        for idx in range(1, len(layers_config)):
            out_dim = layers_config[idx].nodes
            layers_list.append(nn.Linear(current_dim, out_dim))
            
            act = layers_config[idx].activation
            if act == "ReLU":
                layers_list.append(nn.ReLU())
            elif act == "Sigmoid":
                layers_list.append(nn.Sigmoid())
            elif act == "Tanh":
                layers_list.append(nn.Tanh())
            elif act == "Linear":
                pass
                
            current_dim = out_dim

        self.model = nn.Sequential(*layers_list)

    def train_network(self, layers_config, X, Y, lr, epochs, loss_name):
        return self.train_and_record_history(layers_config, X, Y, lr, epochs, loss_name)

    def train_and_record_history(self, layers_config, X, Y, lr, epochs, loss_name):
        if not self.model:
            return []

        criterion = nn.L1Loss() if loss_name == "MAE" else nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        history = []

        expected_in = self.model[0].in_features
        expected_out = self.model[-1].out_features if isinstance(self.model[-1], nn.Linear) else self.model[-2].out_features

        inputs = X.clone()
        if inputs.shape[1] < expected_in:
            inputs = torch.cat([inputs, torch.zeros(1, expected_in - inputs.shape[1])], dim=1)
        else:
            inputs = inputs[:, :expected_in]

        targets = Y.clone()
        if targets.shape[1] < expected_out:
            targets = torch.cat([targets, torch.zeros(1, expected_out - targets.shape[1])], dim=1)
        else:
            targets = targets[:, :expected_out]

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # On capture l'état juste après la mise à jour des poids pour avoir les gradients
            state = self.capture_current_state(layers_config, inputs)
            history.append(state)

        return history

    def capture_current_state(self, layers_config, inputs):
        self.model.eval()
        with torch.no_grad():
            activations = {0: inputs[0].tolist()}
            sub_x = inputs
            linear_counter = 0

            for idx in range(1, len(layers_config)):
                lin_mod = None
                act_mod = None
                current_lin = 0
                for m_idx, m in enumerate(self.model):
                    if isinstance(m, nn.Linear):
                        if current_lin == linear_counter:
                            lin_mod = m
                            if m_idx + 1 < len(self.model) and not isinstance(self.model[m_idx + 1], nn.Linear):
                                act_mod = self.model[m_idx + 1]
                            break
                        current_lin += 1
                if lin_mod is not None:
                    sub_x = lin_mod(sub_x)
                    if act_mod is not None:
                        sub_x = act_mod(sub_x)
                    activations[idx] = sub_x[0].tolist()
                    linear_counter += 1

            weights_snapshots = []
            biases_snapshots = []
            weights_grads = [] # NOUVEAU : Capture des gradients

            for m in self.model:
                if isinstance(m, nn.Linear):
                    weights_snapshots.append(m.weight.detach().clone())
                    biases_snapshots.append(m.bias.detach().clone())
                    
                    # On sécurise au cas où il n'y a pas encore de gradient
                    if m.weight.grad is not None:
                        weights_grads.append(m.weight.grad.detach().clone())
                    else:
                        weights_grads.append(torch.zeros_like(m.weight))

            return {
                "activations": activations,
                "weights": weights_snapshots,
                "biases": biases_snapshots,
                "grads": weights_grads
            }

    def get_forward_activations(self, layers_config):
        in_vals = [n.value for n in layers_config[0].neurons]
        inputs = torch.tensor([in_vals], dtype=torch.float32)
        expected_in = self.model[0].in_features
        if inputs.shape[1] != expected_in:
            inputs = inputs[:, :expected_in] if inputs.shape[1] > expected_in else torch.cat([inputs, torch.zeros(1, expected_in - inputs.shape[1])], dim=1)
            
        state = self.capture_current_state(layers_config, inputs)
        return state["activations"]