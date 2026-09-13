# engine/network.py
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetworkEngine:
    def __init__(self):
        self.model = None

    def build_model(self, layers_config, input_dim):
        """Construit dynamiquement le réseau PyTorch selon l'architecture visuelle"""
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
        """Alias compatible pour l'entraînement avec enregistrement d'historique"""
        return self.train_and_record_history(layers_config, X, Y, lr, epochs, loss_name)

    def train_and_record_history(self, layers_config, X, Y, lr, epochs, loss_name):
        """Entraîne le réseau et enregistre l'historique complet de chaque epoch pour le replay"""
        if not self.model:
            return []

        if loss_name == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        history = []

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            in_vals = [n.value for n in layers_config[0].neurons]
            inputs = torch.tensor([in_vals], dtype=torch.float32)
            
            expected_in = self.model[0].in_features
            if inputs.shape[1] != expected_in:
                if inputs.shape[1] < expected_in:
                    inputs = torch.cat([inputs, torch.zeros(1, expected_in - inputs.shape[1])], dim=1)
                else:
                    inputs = inputs[:, :expected_in]

            outputs = self.model(inputs)
            
            expected_out = self.model[-1].out_features if isinstance(self.model[-1], nn.Linear) else self.model[-2].out_features
            if Y.shape[1] != expected_out:
                target = torch.zeros(1, expected_out)
                target[0, 0] = Y[0, 0] if Y.numel() > 0 else 0.0
            else:
                target = Y[0:1]

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            state = self.capture_current_state(layers_config)
            history.append(state)

        return history

    def capture_current_state(self, layers_config):
        """Capture les poids et activations du modèle à l'instant T"""
        self.model.eval()
        with torch.no_grad():
            in_vals = [n.value for n in layers_config[0].neurons]
            x = torch.tensor([in_vals], dtype=torch.float32)
            expected_in = self.model[0].in_features
            if x.shape[1] != expected_in:
                x = x[:, :expected_in] if x.shape[1] > expected_in else torch.cat([x, torch.zeros(1, expected_in - x.shape[1])], dim=1)

            activations = {0: in_vals}
            sub_x = x
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
            for m in self.model:
                if isinstance(m, nn.Linear):
                    weights_snapshots.append(m.weight.detach().clone())

            return {
                "activations": activations,
                "weights": weights_snapshots
            }

    def get_forward_activations(self, layers_config):
        state = self.capture_current_state(layers_config)
        return state["activations"]