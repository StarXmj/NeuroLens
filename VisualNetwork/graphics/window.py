# graphics/window.py
import pygame
import sys
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from config.settings import *
from engine.layer import Layer, Link, Neuron
from engine.network import NeuralNetworkEngine
from graphics.ui import DraggableItem, Tab, TextInput, Dropdown, Checkbox

def distance_point_ligne(px, py, x1, y1, x2, y2):
    longeur_ligne = math.hypot(x2 - x1, y2 - y1)
    if longeur_ligne == 0: return math.hypot(px - x1, py - y1)
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (longeur_ligne ** 2)
    if u < 0 or u > 1: return min(math.hypot(px - x1, py - y1), math.hypot(px - x2, py - y2))
    ix, iy = x1 + u * (x2 - x1), y1 + u * (y2 - y1)
    return math.hypot(px - ix, py - iy)

class GameEngine:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("NeuroLens - Mountain & Gradient Descent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 12)
        self.math_font = pygame.font.SysFont("Consolas", 11) 
        self.title_font = pygame.font.SysFont("Arial", 28, bold=True)
        
        self.layers = []
        self.links = [] 
        self.pytorch_engine = NeuralNetworkEngine()
        self.training_history = [] 
        
        self.camera_x = 0
        self.camera_y = 0
        self.camera_zoom = 1.0
        self.is_panning = False
        
        self.active_drag = None
        self.right_panel_open = True
        self.btn_toggle_rect = pygame.Rect(WINDOW_WIDTH - SIDEBAR_RIGHT_WIDTH - 30, 10, 30, 30)
        
        # Les deux boutons d'analyse (Graphique temporel et Vallée/Montagne)
        self.btn_details = pygame.Rect(0, 0, 95, 26) 
        self.btn_mountain = pygame.Rect(0, 0, 95, 26) 
        
        self.inspector_scroll_y = 0 
        
        self.tabs = [
            Tab(10, 10, 95, 30, "Build", is_active=True), 
            Tab(115, 10, 95, 30, "Data", is_active=False),
            Tab(220, 10, 95, 30, "Maths", is_active=False)
        ]
        self.current_view = "Build"
        
        self.ui_items = [
            DraggableItem(20, 60, 200, 65, "Input", 4, []), 
            DraggableItem(20, 140, 200, 65, "Hidden", 5, ["ReLU", "Sigmoid", "Tanh", "Linear"]),
            DraggableItem(20, 220, 200, 65, "Output", 2, ["Softmax", "Sigmoid", "Linear"])
        ]
        
        start_y = 310
        esp = 56 
        
        self.global_hyperparams = [
            TextInput(20, start_y, 200, 22, "Learning Rate (LR)", "0.01", "float"),
            TextInput(20, start_y + esp*1, 200, 22, "Batch Size", "32", "int", disabled=True, help_text="Education mode: not required"),
            TextInput(20, start_y + esp*2, 200, 22, "Number of Epochs", "100", "int"),
            Dropdown(20, start_y + esp*3, 200, 22, "Loss Function", ["MSE", "CrossEntropy", "MAE"]),
            Dropdown(20, start_y + esp*4, 200, 22, "Evaluation Metric", ["Accuracy", "Loss", "Precision"]),
            TextInput(20, start_y + esp*5, 200, 22, "Dropout [e.g.: 0.20]", "0.0", "float"),
            Checkbox(20, start_y + esp*6 + 10, 16, "Batch Normalization", False),
            Dropdown(20, start_y + esp*7 + 10, 200, 22, "Penalty", ["None", "L1", "L2"])
        ]
        
        self.data_grid = []
        self.update_data_grid()
        
        table_start_x = SIDEBAR_LEFT_WIDTH + 60
        self.btn_transfer = pygame.Rect(table_start_x, 500, 200, 40)
        self.dataset_transferred = False

        self._recalculate_ui_positions()
        
        self.sim_mode = "ANIMATION"
        self.is_running = False        
        self.current_epoch = 0
        self.simulation_speed = 1.0    
        self.sim_phase = "FORWARD"     
        self.animation_progress = 0.0  
        
        self.btn_mode_anim = pygame.Rect(0, 0, 90, 26)
        self.btn_mode_exp = pygame.Rect(0, 0, 95, 26)
        self.btn_play = pygame.Rect(0, 0, 75, 30)
        self.btn_reset = pygame.Rect(0, 0, 70, 30)
        self.btn_slow = pygame.Rect(0, 0, 40, 30)
        self.btn_fast = pygame.Rect(0, 0, 40, 30)

        self.last_click_time = 0
        self.hovered_item = None
        self.selected_item = None 
        self.focus_target = None
        self.inspector_value_input = TextInput(0, 0, 150, 25, "Value", "0.0", "float") 
        
        self.btn_retour = pygame.Rect(20, 20, 100, 40)
        self.inspector_buttons = []

    def _recalculate_ui_positions(self):
        y_offset = 60
        for item in self.ui_items:
            item.start_x, item.start_y = 20, y_offset
            item.update_position(20, y_offset)
            y_offset += item.rect.height + 15
            if item.is_expanded: y_offset += len(item.activations) * 28 + 5

    def update_data_grid(self):
        input_layer = next((l for l in self.layers if l.name == "Input"), None)
        num_rows = input_layer.nodes if input_layer else 3 
        table_start_x = SIDEBAR_LEFT_WIDTH + 60
        table_start_y = 100
        
        new_grid = []
        for r in range(num_rows):
            if r < len(self.data_grid):
                new_grid.append(self.data_grid[r])
            else:
                row = [
                    TextInput(table_start_x, table_start_y + r * 30, 80, 24, "", "0.0", "float"),
                    TextInput(table_start_x + 90, table_start_y + r * 30, 80, 24, "", "0.0", "float")
                ]
                new_grid.append(row)
        self.data_grid = new_grid
        self.dataset_transferred = False

    def rebuild_links(self):
        self.training_history = []
        self.current_epoch = 0
        self.is_running = False
        
        self.links = []
        Link._id_counter = 1
        Neuron._id_counter = 1
        for couche in self.layers:
            for n in couche.neurons:
                n.id = f"N{Neuron._id_counter}"
                Neuron._id_counter += 1
                
        for idx in range(len(self.layers) - 1):
            c1, c2 = self.layers[idx], self.layers[idx + 1]
            for n1 in c1.neurons:
                for n2 in c2.neurons:
                    self.links.append(Link(n1, n2))
        self.update_data_grid()

    def get_target_epochs(self):
        try:
            return int(self.global_hyperparams[2].text)
        except ValueError:
            return 100

    def is_dataset_valid(self):
        input_layer = next((l for l in self.layers if l.name == "Input"), None)
        if not input_layer or len(self.data_grid) != input_layer.nodes: return False
        for row in self.data_grid:
            try:
                float(row[0].text.strip())
                float(row[1].text.strip())
            except ValueError:
                return False
        return True

    def prepare_pytorch_experience(self):
        if not self.dataset_transferred or len(self.layers) < 2:
            print("⚠️ [EXPERIENCE] Please transfer valid data first.")
            self.is_running = False
            return

        print("🚀 [EXPERIENCE] Running PyTorch and saving history...")
        X_data = [float(row[0].text.strip()) for row in self.data_grid]
        Y_data = [float(row[1].text.strip()) for row in self.data_grid]
        
        X = torch.tensor([X_data], dtype=torch.float32)
        Y = torch.tensor([Y_data], dtype=torch.float32)

        in_features = self.layers[0].nodes
        self.pytorch_engine.build_model(self.layers, in_features)

        try:
            lr = float(self.global_hyperparams[0].text)
        except ValueError:
            lr = 0.01

        epochs = self.get_target_epochs()
        loss_name = self.global_hyperparams[3].selected

        self.training_history = self.pytorch_engine.train_and_record_history(self.layers, X, Y, lr, epochs, loss_name)
        self.current_epoch = 0
        if self.training_history:
            self.apply_history_state(0)
        print("✅ [EXPERIENCE] History generated! You can now play, pause, and navigate.")

    def apply_history_state(self, epoch_idx):
        if not self.training_history or epoch_idx >= len(self.training_history):
            return
        
        state = self.training_history[epoch_idx]
        
        weights_snapshots = state["weights"]
        linear_layers_idx = 0
        for idx in range(1, len(self.layers)):
            prev_layer = self.layers[idx - 1]
            curr_layer = self.layers[idx]
            if linear_layers_idx < len(weights_snapshots):
                weight_tensor = weights_snapshots[linear_layers_idx]
                max_j = min(len(curr_layer.neurons), weight_tensor.shape[0])
                max_i = min(len(prev_layer.neurons), weight_tensor.shape[1])
                
                for j in range(max_j):
                    n_curr = curr_layer.neurons[j]
                    for i in range(max_i):
                        n_prev = prev_layer.neurons[i]
                        link = next((l for l in self.links if l.n1 == n_prev and l.n2 == n_curr), None)
                        if link:
                            link.value = round(float(weight_tensor[j, i].item()), 2)
                linear_layers_idx += 1

        activations_dict = state["activations"]
        for idx, layer in enumerate(self.layers):
            if idx in activations_dict:
                vals = activations_dict[idx]
                max_n = min(len(layer.neurons), len(vals))
                for n_idx in range(max_n):
                    layer.neurons[n_idx].value = round(float(vals[n_idx]), 2)

    def _get_selected_history(self):
        if not self.training_history or self.sim_mode != "EXPERIENCE": return []
        data = []
        try:
            lr = float(self.global_hyperparams[0].text)
            if self.selected_item['type'] == 'neurone':
                n = self.selected_item['obj']
                l_idx = next(i for i, c in enumerate(self.layers) if n in c.neurons)
                n_idx = self.layers[l_idx].neurons.index(n)
                
                for ep, state in enumerate(self.training_history):
                    if ep > self.current_epoch: break 
                    
                    if l_idx in state['activations'] and n_idx < len(state['activations'][l_idx]):
                        val = float(state['activations'][l_idx][n_idx])
                        if l_idx > 0:
                            inputs = state['activations'][l_idx - 1]
                            weights = state['weights'][l_idx - 1][n_idx]
                            bias = float(state['biases'][l_idx - 1][n_idx])
                            
                            sum_str = " + ".join([f"({float(w):.2f} * {float(i):.2f})" for w, i in zip(weights, inputs)])
                            sum_val = sum([float(w)*float(i) for w, i in zip(weights, inputs)])
                            act_name = self.layers[l_idx].activation
                            
                            lines = [
                                f"[Forward] Z = Σ(w*x) + bias",
                                f"Z = {sum_str}",
                                f"Z = {sum_val:.4f} + {bias:.4f} = {sum_val + bias:.4f}",
                                f"A = {act_name}(Z) = {val:.4f}"
                            ]
                        else:
                            lines = [f"[Input Layer] X = {val:.4f}"]
                        
                        data.append((ep, val, lines, 0.0)) 

            elif self.selected_item['type'] == 'lien':
                l = self.selected_item['obj']
                prev_layer = next(ly for ly in self.layers if l.n1 in ly.neurons)
                curr_layer = next(ly for ly in self.layers if l.n2 in ly.neurons)
                prev_idx = self.layers.index(prev_layer)
                curr_idx = self.layers.index(curr_layer)
                linear_idx = curr_idx - 1 
                n_prev_idx = prev_layer.neurons.index(l.n1)
                n_curr_idx = curr_layer.neurons.index(l.n2)
                
                prev_w = None
                for ep, state in enumerate(self.training_history):
                    if ep > self.current_epoch: break 
                    
                    if linear_idx < len(state['weights']):
                        w_tensor = state['weights'][linear_idx]
                        grad_tensor = state['grads'][linear_idx]
                        if n_curr_idx < w_tensor.shape[0] and n_prev_idx < w_tensor.shape[1]:
                            val = float(w_tensor[n_curr_idx, n_prev_idx].item())
                            grad = float(grad_tensor[n_curr_idx, n_prev_idx].item())
                            
                            if prev_w is None:
                                lines = [f"[Init] w = {val:.4f}"]
                            else:
                                lines = [
                                    f"[Backward] w_new = w_old - (LR * Grad)",
                                    f"Grad(∂L/∂w) = {grad:.6f}",
                                    f"w_new = {prev_w:.4f} - ({lr} * {grad:.6f})",
                                    f"w_new = {val:.4f}"
                                ]
                            prev_w = val
                            data.append((ep, val, lines, grad))
        except Exception:
            pass
        return data

    def open_matplotlib_details(self):
        """Ouvre le graphique temporel 2D"""
        history_data = self._get_selected_history()
        if not history_data or len(history_data) < 2:
            print("⚠️ Not enough data to plot.")
            return

        epochs = [d[0] for d in history_data]
        vals = [d[1] for d in history_data]

        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')

        if self.selected_item['type'] == 'neurone':
            plt.plot(epochs, vals, marker='o', color='cyan', linewidth=2, label='Activation (A)')
            plt.title(f"Activation Progress - Neuron {self.selected_item['obj'].id}", fontsize=14)
            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel("Activation Value", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            grads = [d[3] for d in history_data]
            fig, ax1 = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#1e1e1e')
            ax1.set_facecolor('#1e1e1e')

            color_w = 'lightgreen'
            ax1.set_xlabel('Epochs', color='white', fontsize=12)
            ax1.set_ylabel('Weight Value (W)', color=color_w, fontsize=12)
            ax1.plot(epochs, vals, color=color_w, marker='o', linewidth=2, label='Weight (W)')
            ax1.tick_params(axis='y', labelcolor=color_w)
            ax1.grid(True, alpha=0.2)

            ax2 = ax1.twinx()
            color_g = 'salmon'
            ax2.set_ylabel('Gradient (∂L/∂w)', color=color_g, fontsize=12)
            ax2.plot(epochs, grads, color=color_g, linestyle='--', marker='x', label='Gradient')
            ax2.tick_params(axis='y', labelcolor=color_g)

            plt.title(f"Weight & Gradient Evolution - Link {self.selected_item['obj'].id}", color='white', fontsize=14)
            fig.tight_layout()

        plt.show() 

    def open_matplotlib_mountain(self):
        """La fameuse option 'Montagne' (Loss Landscape) avec la balle qui descend !"""
        history_data = self._get_selected_history()
        if not history_data or len(history_data) < 2: return
        
        if self.selected_item['type'] != 'lien':
            print("⚠️ The Mountain view is only available for Links (Weights).")
            return

        epochs = [d[0] for d in history_data]
        weights = [d[1] for d in history_data]
        grads = [d[3] for d in history_data]

        w_final = weights[-1]
        w_start = weights[0]
        
        diff = w_start - w_final
        a = abs(grads[0] / diff) if abs(diff) > 1e-6 else 1.0

        margin = abs(w_start - w_final) * 0.5
        if margin == 0: margin = 0.5
        
        # Génération des points de la parabole mathématique en Python pur (sans numpy)
        w_min = min(weights) - margin
        w_max = max(weights) + margin
        w_vals = [w_min + (w_max - w_min) * i / 100.0 for i in range(101)]
        loss_landscape = [0.5 * a * (w - w_final)**2 for w in w_vals]
        
        pseudo_loss = [0.5 * a * (w - w_final)**2 for w in weights]

        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')
        
        # Dessine la vallée / montagne
        plt.plot(w_vals, loss_landscape, color='#4CAF50', linewidth=3, label='Loss Landscape (Mountain)')
        plt.fill_between(w_vals, loss_landscape, max(loss_landscape), color='#4CAF50', alpha=0.15)
        
        # Dessine la trajectoire de la balle (le poids)
        plt.plot(weights, pseudo_loss, color='white', linestyle='--', alpha=0.5)
        plt.scatter(weights, pseudo_loss, color='red', zorder=5, s=60, label='Weight rolling down')
        
        # Flèches de descente
        step = max(1, len(weights) // 15)
        for i in range(0, len(weights)-1, step):
            if pseudo_loss[i] > pseudo_loss[i+1]: 
                plt.annotate('', xy=(weights[i+1], pseudo_loss[i+1]), 
                             xytext=(weights[i], pseudo_loss[i]),
                             arrowprops=dict(arrowstyle="->", color='yellow', lw=2))
                         
        plt.title(f"Gradient Descent 'Mountain' - Link {self.selected_item['obj'].id}", fontsize=15, color='white')
        plt.xlabel("Weight Value (W)", fontsize=12)
        plt.ylabel("Loss (Error)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if self.right_panel_open and mx > WINDOW_WIDTH - SIDEBAR_RIGHT_WIDTH:
                    self.inspector_scroll_y -= event.y * 30
                    self.inspector_scroll_y = max(0, self.inspector_scroll_y) 
                elif mx > SIDEBAR_LEFT_WIDTH:
                    if event.y > 0:
                        self.camera_zoom = min(4.0, self.camera_zoom + 0.1)
                    else:
                        self.camera_zoom = max(0.3, self.camera_zoom - 0.1)

            if self.current_view == "Build":
                for hp in self.global_hyperparams: hp.handle_event(event)
                if self.selected_item: self.inspector_value_input.handle_event(event)
            elif self.current_view == "Data":
                for row in self.data_grid:
                    for cell in row:
                        old = cell.text
                        cell.handle_event(event)
                        if cell.text != old: self.dataset_transferred = False

            if event.type == pygame.KEYDOWN and self.selected_item and self.inspector_value_input.active:
                try:
                    val = float(self.inspector_value_input.text)
                    if self.selected_item['type'] == 'neurone': self.selected_item['obj'].value = val
                    elif self.selected_item['type'] == 'lien': self.selected_item['obj'].value = val
                except ValueError: pass

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in [2, 3]:
                    self.is_panning = True
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    return

                if event.button == 1:
                    # CLICS SUR LES NOUVEAUX BOUTONS D'ANALYSE
                    if self.right_panel_open and self.selected_item:
                        if self.btn_details.collidepoint(event.pos):
                            self.open_matplotlib_details()
                            return
                        if self.btn_mountain.collidepoint(event.pos):
                            self.open_matplotlib_mountain()
                            return

                    if self.btn_mode_anim.collidepoint(event.pos): 
                        self.sim_mode = "ANIMATION"
                        self.is_running = False
                        return
                    if self.btn_mode_exp.collidepoint(event.pos): 
                        self.sim_mode = "EXPERIENCE"
                        self.is_running = False
                        self.training_history = []
                        self.current_epoch = 0
                        return

                    if self.current_view == "Data" and self.btn_transfer.collidepoint(event.pos):
                        if self.is_dataset_valid():
                            input_layer = next((l for l in self.layers if l.name == "Input"), None)
                            if input_layer:
                                for i, row in enumerate(self.data_grid):
                                    input_layer.neurons[i].value = float(row[0].text.strip())
                            self.dataset_transferred = True
                        return

                    if self.btn_play.collidepoint(event.pos):
                        if self.sim_mode == "ANIMATION": 
                            self.is_running = not self.is_running
                        else:
                            if not self.training_history:
                                self.prepare_pytorch_experience()
                            self.is_running = not self.is_running
                        return

                    if self.btn_reset.collidepoint(event.pos):
                        self.is_running = False
                        self.current_epoch = 0
                        self.animation_progress = 0.0
                        self.camera_x, self.camera_y, self.camera_zoom = 0, 0, 1.0
                        if self.sim_mode == "EXPERIENCE" and self.training_history:
                            self.apply_history_state(0)
                        return

                    if self.btn_slow.collidepoint(event.pos):
                        self.simulation_speed = max(0.25, self.simulation_speed - 0.25)
                        return
                    if self.btn_fast.collidepoint(event.pos):
                        self.simulation_speed = min(3.0, self.simulation_speed + 0.25)
                        return

                    if self.current_view == "Focus" and self.btn_retour.collidepoint(event.pos):
                        self.current_view = "Build"; return

                    if self.right_panel_open and WINDOW_WIDTH - SIDEBAR_RIGHT_WIDTH <= event.pos[0] <= WINDOW_WIDTH:
                        if self.selected_item:
                            self.inspector_value_input.handle_event(event)
                        else:
                            for btn in self.inspector_buttons:
                                if btn['rect'].collidepoint(event.pos):
                                    idx = btn['layer_idx']
                                    if btn['action'] == 'delete':
                                        self.layers.pop(idx)
                                        self.rebuild_links()
                                        self.selected_item = None
                                    elif btn['action'] == 'change_activation':
                                        acts = ["ReLU", "Sigmoid", "Tanh", "Linear", "Softmax"]
                                        couche = self.layers[idx]
                                        curr = acts.index(couche.activation) if couche.activation in acts else 0
                                        couche.activation = acts[(curr + 1) % len(acts)]
                                    return
                        return

                    if self.current_view == "Build" and event.pos[0] > SIDEBAR_LEFT_WIDTH and event.pos[0] < WINDOW_WIDTH - (SIDEBAR_RIGHT_WIDTH if self.right_panel_open else 0):
                        current_time = pygame.time.get_ticks()
                        if self.hovered_item:
                            self.selected_item = self.hovered_item
                            self.inspector_scroll_y = 0 
                            self.inspector_value_input.text = str(self.hovered_item['obj'].value)
                            if current_time - self.last_click_time < 300:
                                self.focus_target = self.hovered_item; self.current_view = "Focus"; return
                        else: self.selected_item = None
                        self.last_click_time = current_time

                    if self.btn_toggle_rect.collidepoint(event.pos):
                        self.right_panel_open = not self.right_panel_open; return

                    for tab in self.tabs:
                        if tab.rect.collidepoint(event.pos):
                            self.current_view = tab.name
                            for t in self.tabs: t.is_active = (t == tab)
                            return

                    if self.current_view == "Build":
                        for item in self.ui_items:
                            if item.btn_minus.collidepoint(event.pos): item.default_nodes = max(1, item.default_nodes - 1); self.update_data_grid(); return
                            elif item.btn_plus.collidepoint(event.pos): item.default_nodes = min(15, item.default_nodes + 1); self.update_data_grid(); return
                            elif item.rect.collidepoint(event.pos):
                                self.active_drag = item; item.is_dragging = True
                                item.offset_x, item.offset_y = item.rect.x - event.pos[0], item.rect.y - event.pos[1]
                                break

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in [2, 3] or (event.button == 1 and self.is_panning):
                    self.is_panning = False
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                    return

                if event.button == 1 and self.active_drag:
                    if self.active_drag.rect.x > SIDEBAR_LEFT_WIDTH: self._add_layer_from_drag()
                    self.active_drag.update_position(self.active_drag.start_x, self.active_drag.start_y)
                    self.active_drag.is_dragging = False; self.active_drag = None

            elif event.type == pygame.MOUSEMOTION:
                if self.is_panning:
                    self.camera_x += event.rel[0] / self.camera_zoom
                    self.camera_y += event.rel[1] / self.camera_zoom
                    return

                if self.active_drag:
                    self.active_drag.update_position(event.pos[0] + self.active_drag.offset_x, event.pos[1] + self.active_drag.offset_y)

    def _add_layer_from_drag(self):
        nom, nodes, act = self.active_drag.name, self.active_drag.default_nodes, self.active_drag.selected_act
        if (nom == "Input" and any(l.name == "Input" for l in self.layers)) or (nom == "Output" and any(l.name == "Output" for l in self.layers)): return
        nouvelle_couche = Layer(nom, nodes=nodes, activation=act)
        if nom == "Input": self.layers.insert(0, nouvelle_couche)
        elif nom == "Output": self.layers.append(nouvelle_couche)
        else:
            if any(l.name == "Output" for l in self.layers): self.layers.insert(-1, nouvelle_couche)
            else: self.layers.append(nouvelle_couche)
        self.rebuild_links()

    def update_simulation(self):
        if not self.is_running or not self.layers: return
        max_epochs = self.get_target_epochs()
        
        if self.current_epoch >= max_epochs:
            self.current_epoch = max_epochs
            self.is_running = False
            return

        if self.sim_mode == "ANIMATION":
            self.animation_progress += 0.03 * self.simulation_speed
            if self.animation_progress >= 1.0:
                self.animation_progress = 0.0
                if self.sim_phase == "FORWARD": 
                    self.sim_phase = "BACKWARD"
                else:
                    self.sim_phase = "FORWARD"
                    self.current_epoch += 1
        elif self.sim_mode == "EXPERIENCE":
            self.animation_progress += 0.15 * self.simulation_speed
            if self.animation_progress >= 1.0:
                self.animation_progress = 0.0
                self.current_epoch += 1
                if self.training_history:
                    idx_history = min(self.current_epoch, len(self.training_history) - 1)
                    self.apply_history_state(idx_history)

    def get_neuron_layer_and_color(self, neuron):
        for couche in self.layers:
            if neuron in couche.neurons: return couche, DICT_COULEURS_ACTIVATION.get(couche.activation, (150, 150, 150))
        return None, (150, 150, 150)

    def draw_network(self):
        self.update_simulation()
        espace_gauche = SIDEBAR_LEFT_WIDTH
        espace_droit = SIDEBAR_RIGHT_WIDTH if self.right_panel_open else 0
        net_width = WINDOW_WIDTH - espace_gauche - espace_droit
        
        center_x = espace_gauche + net_width // 2
        center_y = WINDOW_HEIGHT // 2

        self.hovered_item = None
        if not self.layers: return
        espacement_x = net_width // (len(self.layers) + 1)
        mx, my = pygame.mouse.get_pos()

        for idx, couche in enumerate(self.layers):
            base_x = espace_gauche + (idx + 1) * espacement_x
            ey = WINDOW_HEIGHT // (couche.nodes + 1)
            for i, n in enumerate(couche.neurons):
                base_y = (i + 1) * ey
                px = base_x + self.camera_x
                py = base_y + self.camera_y
                n.x = int(center_x + (px - center_x) * self.camera_zoom)
                n.y = int(center_y + (py - center_y) * self.camera_zoom)

        scaled_thick = max(1, int(1 * self.camera_zoom))
        scaled_thick_hover = max(2, int(4 * self.camera_zoom))
        
        for l in self.links:
            x1, y1, x2, y2 = l.n1.x, l.n1.y, l.n2.x, l.n2.y
            dist = distance_point_ligne(mx, my, x1, y1, x2, y2)
            couleur_trait = (200, 50, 50) if l.value < 0 else ((50, 150, 255) if l.value > 0 else (100, 100, 100))
            is_hovered = dist < (6 * self.camera_zoom)
            
            if is_hovered or (self.selected_item and self.selected_item['obj'] == l):
                pygame.draw.line(self.screen, (255, 255, 150), (x1, y1), (x2, y2), scaled_thick_hover)
            else: 
                pygame.draw.line(self.screen, couleur_trait, (x1, y1), (x2, y2), scaled_thick)
            
            if is_hovered: self.hovered_item = {'type': 'lien', 'obj': l}

            if self.is_running:
                if self.sim_mode == "ANIMATION":
                    prog = self.animation_progress
                    if self.sim_phase == "BACKWARD": prog = 1.0 - prog
                    color = (50, 255, 50) if self.sim_phase == "FORWARD" else (255, 165, 0)
                else:
                    prog = (pygame.time.get_ticks() / 1000.0 * self.simulation_speed) % 1.0
                    color = (50, 200, 255)
                    
                px = x1 + (x2 - x1) * prog
                py = y1 + (y2 - y1) * prog
                pygame.draw.circle(self.screen, color, (int(px), int(py)), max(2, int(4 * self.camera_zoom)))

            if self.camera_zoom > 0.6:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                txt_val = self.small_font.render(f"{l.value}", True, (200,200,200))
                self.screen.blit(txt_val, (cx - txt_val.get_width()//2, cy - int(15 * self.camera_zoom)))

        scaled_rad = max(4, int(20 * self.camera_zoom))
        scaled_rad_hover = max(5, int(24 * self.camera_zoom))
        
        for couche in self.layers:
            col = DICT_COULEURS_ACTIVATION.get(couche.activation, (150, 150, 150))
            for n in couche.neurons:
                is_hovered = math.hypot(mx - n.x, my - n.y) < scaled_rad_hover
                if is_hovered: self.hovered_item = {'type': 'neurone', 'obj': n, 'couche': couche, 'couleur': col}
                
                if is_hovered or (self.selected_item and self.selected_item['obj'] == n):
                    pygame.draw.circle(self.screen, (255, 255, 255), (n.x, n.y), scaled_rad_hover)

                pygame.draw.circle(self.screen, col, (n.x, n.y), scaled_rad)
                pygame.draw.circle(self.screen, (0, 0, 0), (n.x, n.y), scaled_rad, max(1, int(2 * self.camera_zoom)))
                
                if self.camera_zoom > 0.5:
                    txt_val = self.small_font.render(f"{n.value}", True, (0,0,0))
                    self.screen.blit(txt_val, (n.x - txt_val.get_width()//2, n.y - txt_val.get_height()//2))

        self.draw_simulation_hud(espace_gauche, net_width)

    def draw_simulation_hud(self, espace_gauche, net_width):
        hud_y = 10
        center_x = espace_gauche + net_width // 2
        max_epochs = self.get_target_epochs()

        self.btn_mode_anim.topleft = (center_x - 100, hud_y)
        self.btn_mode_exp.topleft = (center_x + 5, hud_y)

        pygame.draw.rect(self.screen, (50, 130, 200) if self.sim_mode == "ANIMATION" else (60, 60, 60), self.btn_mode_anim, border_radius=4)
        self.screen.blit(self.small_font.render("Animation", True, (255,255,255)), (self.btn_mode_anim.x + 18, self.btn_mode_anim.y + 5))

        pygame.draw.rect(self.screen, (200, 100, 50) if self.sim_mode == "EXPERIENCE" else (60, 60, 60), self.btn_mode_exp, border_radius=4)
        self.screen.blit(self.small_font.render("Experience", True, (255,255,255)), (self.btn_mode_exp.x + 15, self.btn_mode_exp.y + 5))

        status_y = hud_y + 32
        statut_txt = f"MODE: {self.sim_mode} (RUNNING...)" if self.is_running else f"MODE: {self.sim_mode} (READY)"
        couleur_statut = (50, 250, 50) if self.is_running else (150, 150, 150)
        
        txt_surface = self.small_font.render(statut_txt, True, couleur_statut)
        self.screen.blit(txt_surface, (center_x - txt_surface.get_width() // 2, status_y))

        epoch_surface = self.small_font.render(f"Epoch : {self.current_epoch} / {max_epochs}", True, (255, 255, 255))
        self.screen.blit(epoch_surface, (center_x - epoch_surface.get_width() // 2, status_y + 18))

        btn_y = status_y + 40
        self.btn_play.topleft = (center_x - 145, btn_y)
        self.btn_reset.topleft = (center_x - 65, btn_y)
        self.btn_slow.topleft = (center_x + 15, btn_y)
        self.btn_fast.topleft = (center_x + 60, btn_y)

        play_label = "Pause" if self.is_running else "Start"
        pygame.draw.rect(self.screen, (50, 180, 80) if not self.is_running else (200, 80, 50), self.btn_play, border_radius=4)
        self.screen.blit(self.small_font.render(play_label, True, (255,255,255)), (self.btn_play.x + 22, self.btn_play.y + 6))

        pygame.draw.rect(self.screen, (100, 100, 100), self.btn_reset, border_radius=4)
        self.screen.blit(self.small_font.render("Reset", True, (255,255,255)), (self.btn_reset.x + 18, self.btn_reset.y + 6))

        pygame.draw.rect(self.screen, (70, 70, 90), self.btn_slow, border_radius=4)
        self.screen.blit(self.small_font.render("-", True, (255,255,255)), (self.btn_slow.x + 15, self.btn_slow.y + 5))

        spd_surface = self.small_font.render(f"{self.simulation_speed}x", True, (200, 200, 200))
        self.screen.blit(spd_surface, (center_x + 105, btn_y + 6))

        pygame.draw.rect(self.screen, (70, 70, 90), self.btn_fast, border_radius=4)
        self.screen.blit(self.small_font.render("+", True, (255,255,255)), (self.btn_fast.x + 14, self.btn_fast.y + 5))

    def draw_focus_view(self):
        self.update_simulation()
        pygame.draw.rect(self.screen, COLOR_DRAG_ITEM, self.btn_retour, border_radius=5)
        self.screen.blit(self.font.render("< Back", True, COLOR_TEXT), (self.btn_retour.x + 25, self.btn_retour.y + 10))
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2

        if self.focus_target['type'] == 'neurone':
            n, couche, couleur = self.focus_target['obj'], self.focus_target['couche'], self.focus_target['couleur']
            pygame.draw.circle(self.screen, couleur, (cx, cy), 150)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), 150, 8)
            txt_id = self.title_font.render(f"ZOOM - {n.id} (Layer: {couche.name})", True, COLOR_TEXT)
            self.screen.blit(txt_id, (cx - txt_id.get_width()//2, cy - 250))
            txt_val = self.title_font.render(f"Value: {n.value}", True, (0,0,0))
            self.screen.blit(txt_val, (cx - txt_val.get_width()//2, cy - 20))

        elif self.focus_target['type'] == 'lien':
            l = self.focus_target['obj']
            _, col1 = self.get_neuron_layer_and_color(l.n1)
            _, col2 = self.get_neuron_layer_and_color(l.n2)
            pygame.draw.line(self.screen, (200, 200, 200), (cx - 200, cy), (cx + 200, cy), 15)

            if self.is_running:
                for offset in [0.0]:
                    prog = (self.animation_progress + offset) % 1.0
                    if self.sim_phase == "BACKWARD": prog = 1.0 - prog
                    px = (cx - 200) + 400 * prog
                    couleur_particule = (50, 255, 50) if self.sim_phase == "FORWARD" else (255, 165, 0)
                    pygame.draw.circle(self.screen, couleur_particule, (int(px), cy), 8)

            pygame.draw.circle(self.screen, col1, (cx - 200, cy), 80)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx - 200, cy), 80, 5)
            self.screen.blit(self.font.render(l.n1.id, True, (0,0,0)), (cx - 210, cy - 10))
            
            pygame.draw.circle(self.screen, col2, (cx + 200, cy), 80)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx + 200, cy), 80, 5)
            self.screen.blit(self.font.render(l.n2.id, True, (0,0,0)), (cx + 190, cy - 10))
            
            txt = self.title_font.render(f"Link {l.id} Weight: {l.value}", True, COLOR_TEXT)
            self.screen.blit(txt, (cx - txt.get_width()//2, cy - 150))

    def draw_data_view(self):
        start_x, start_y = SIDEBAR_LEFT_WIDTH + 60, 60
        self.screen.blit(self.title_font.render("Dataset (Simple Regression)", True, COLOR_TEXT), (start_x, start_y))
        
        self.screen.blit(self.font.render("Input (X)", True, (150, 150, 255)), (start_x, start_y + 45))
        self.screen.blit(self.font.render("Target (Y)", True, (255, 150, 150)), (start_x + 90, start_y + 45))

        input_layer = next((l for l in self.layers if l.name == "Input"), None)
        if not input_layer:
            err_txt = self.font.render("⚠️ ERROR: No input layer created!", True, (255, 60, 60))
            self.screen.blit(err_txt, (start_x, start_y + 80))
        else:
            for r, row in enumerate(self.data_grid):
                for cell in row: 
                    cell.rect.y = start_y + 80 + (r * 30)
                    cell.draw(self.screen, self.font, self.small_font)
            
            self.btn_transfer.y = start_y + len(self.data_grid) * 30 + 130
            pygame.draw.rect(self.screen, (50, 180, 80) if self.dataset_transferred else (70, 130, 180), self.btn_transfer, border_radius=6)
            self.screen.blit(self.font.render("✓ Transferred!" if self.dataset_transferred else "Transfer Data", True, COLOR_TEXT), (self.btn_transfer.x + 35, self.btn_transfer.y + 10))

    def draw_math_view(self):
        start_x, start_y = SIDEBAR_LEFT_WIDTH + 60, 60
        self.screen.blit(self.title_font.render("Mathematical Glossary", True, COLOR_TEXT), (start_x, start_y))
        
        terms = [
            ("X (Input Data)", "Les caractéristiques d'entrée données au réseau."),
            ("Y (Target)", "La réponse correcte attendue (vérité terrain)."),
            ("W (Weight)", "La force d'une connexion (le Poids). Apprend pendant l'entraînement."),
            ("B (Bias)", "Le Biais. Une constante ajoutée pour décaler la fonction d'activation."),
            ("Z (Weighted Sum)", "La somme pondérée : Z = (W1*X1 + W2*X2 + ...) + Biais"),
            ("A (Activation)", "La sortie finale du neurone : A = f(Z) où f est ReLU, Sigmoid, etc."),
            ("L (Loss)", "L'erreur mathématique entre la prédiction A et la cible Y."),
            ("Forward Pass", "Les données avancent de X vers Y pour faire une prédiction."),
            ("Backward Pass", "L'erreur recule de Y vers X pour corriger les Poids via les Gradients."),
            ("Grad (Gradient)", "La pente de l'erreur (∂L/∂w). Indique dans quel sens corriger le Poids."),
            ("LR (Learning Rate)", "La taille du pas d'apprentissage : W_new = W_old - (LR * Grad).")
        ]
        
        y_offset = start_y + 50
        for title, desc in terms:
            self.screen.blit(self.font.render(title, True, (50, 200, 255)), (start_x, y_offset))
            self.screen.blit(self.small_font.render(desc, True, (200, 200, 200)), (start_x + 180, y_offset + 2))
            y_offset += 35

    def draw_mini_graph(self, surface, x, y, w, h, data):
        pygame.draw.rect(surface, (40, 40, 40), (x, y, w, h), border_radius=4)
        pygame.draw.rect(surface, (100, 100, 100), (x, y, w, h), 1, border_radius=4)
        if not data: return
        
        max_val = max(v for e, v, l, g in data)
        min_val = min(v for e, v, l, g in data)
        val_range = max_val - min_val if max_val != min_val else 1.0
        
        if max_val > 0 and min_val < 0:
            zero_y = y + h - ((0 - min_val) / val_range) * h
            pygame.draw.line(surface, (80, 80, 80), (x, zero_y), (x+w, zero_y), 1)

        points = []
        for i, (ep, val, lines, grad) in enumerate(data):
            px = x + (i / max(1, len(data)-1)) * w
            py = y + h - ((val - min_val) / val_range) * h
            points.append((px, py))
            
        if len(points) > 1:
            pygame.draw.lines(surface, (50, 200, 255), False, points, 2)

    def draw_right_panel(self):
        self.inspector_buttons = []
        if self.right_panel_open:
            panel_x = WINDOW_WIDTH - SIDEBAR_RIGHT_WIDTH
            self.btn_toggle_rect.x = panel_x - 30
            pygame.draw.rect(self.screen, COLOR_SIDEBAR, (panel_x, 0, SIDEBAR_RIGHT_WIDTH, WINDOW_HEIGHT))
            
            if self.selected_item:
                obj = self.selected_item['obj']
                header_color = (150, 150, 150)
                if self.selected_item['type'] == 'neurone':
                    _, header_color = self.get_neuron_layer_and_color(obj)
                elif self.selected_item['type'] == 'lien':
                    _, header_color = self.get_neuron_layer_and_color(obj.n2)

                id_badge_rect = pygame.Rect(panel_x + 20, 20, 80, 30)
                pygame.draw.rect(self.screen, header_color, id_badge_rect, border_radius=5)
                pygame.draw.rect(self.screen, (0,0,0), id_badge_rect, 2, border_radius=5)
                
                txt_id = self.font.render(obj.id, True, (0, 0, 0) if sum(header_color)>400 else (255, 255, 255))
                self.screen.blit(txt_id, (id_badge_rect.x + (80 - txt_id.get_width())//2, id_badge_rect.y + 5))
                
                type_str = "Neuron" if self.selected_item['type'] == 'neurone' else "Link / Weight"
                txt_type = self.small_font.render(type_str, True, (180, 180, 180))
                self.screen.blit(txt_type, (panel_x + 110, 26))

                pygame.draw.line(self.screen, (100, 100, 100), (panel_x + 20, 60), (panel_x + SIDEBAR_RIGHT_WIDTH - 20, 60), 2)
                
                self.inspector_value_input.rect.x = panel_x + 20
                self.inspector_value_input.rect.y = 80
                self.inspector_value_input.draw(self.screen, self.font, self.small_font)
                
                hist_y_start = 140
                pygame.draw.line(self.screen, (100, 100, 100), (panel_x + 20, hist_y_start - 10), (panel_x + SIDEBAR_RIGHT_WIDTH - 20, hist_y_start - 10), 2)
                self.screen.blit(self.font.render("Math Breakdown (History)", True, COLOR_TEXT), (panel_x + 20, hist_y_start))
                
                # --- NOUVEAUX BOUTONS MATPLOTLIB ---
                self.btn_details.x = panel_x + 20
                self.btn_details.y = hist_y_start + 25
                pygame.draw.rect(self.screen, (150, 50, 200), self.btn_details, border_radius=4)
                self.screen.blit(self.small_font.render("2D Graph", True, (255, 255, 255)), (self.btn_details.x + 20, self.btn_details.y + 6))
                
                self.btn_mountain.x = panel_x + 125
                self.btn_mountain.y = hist_y_start + 25
                pygame.draw.rect(self.screen, (70, 150, 80), self.btn_mountain, border_radius=4)
                self.screen.blit(self.small_font.render("Mountain", True, (255, 255, 255)), (self.btn_mountain.x + 20, self.btn_mountain.y + 6))
                
                history_data = self._get_selected_history()
                if history_data:
                    graph_rect_y = hist_y_start + 60
                    self.draw_mini_graph(self.screen, panel_x + 20, graph_rect_y, SIDEBAR_RIGHT_WIDTH - 40, 60, history_data)
                    
                    text_start_y = graph_rect_y + 70
                    clip_rect = pygame.Rect(panel_x, text_start_y, SIDEBAR_RIGHT_WIDTH, WINDOW_HEIGHT - text_start_y - 10)
                    self.screen.set_clip(clip_rect)
                    
                    item_height = 80 if self.selected_item['type'] == 'lien' else 75
                    content_height = len(history_data) * item_height
                    max_scroll = max(0, content_height - clip_rect.height)
                    
                    if self.is_running and self.sim_mode == "EXPERIENCE":
                        self.inspector_scroll_y = max_scroll
                    else:
                        self.inspector_scroll_y = min(self.inspector_scroll_y, max_scroll)
                    
                    y_offset = text_start_y + 5 - self.inspector_scroll_y
                    
                    for ep, val, lines, grad in history_data:
                        if clip_rect.top - item_height <= y_offset <= clip_rect.bottom:
                            ep_txt = self.small_font.render(f"Epoch {ep}:", True, (180, 220, 255))
                            self.screen.blit(ep_txt, (panel_x + 20, y_offset))
                            
                            l_off = 15
                            for line in lines:
                                color = (150, 255, 150) if "]" in line else (230, 230, 230)
                                val_txt = self.math_font.render(line, True, color)
                                self.screen.blit(val_txt, (panel_x + 25, y_offset + l_off))
                                l_off += 15
                            
                        y_offset += item_height
                        
                    self.screen.set_clip(None)
                    
                    if content_height > clip_rect.height:
                        sb_h = max(20, (clip_rect.height / content_height) * clip_rect.height)
                        sb_y = clip_rect.y + (self.inspector_scroll_y / max_scroll) * (clip_rect.height - sb_h)
                        pygame.draw.rect(self.screen, (100, 100, 100), (panel_x + SIDEBAR_RIGHT_WIDTH - 10, sb_y, 6, sb_h), border_radius=3)
                else:
                    self.screen.blit(self.small_font.render("Run Experience to see the math.", True, (150, 150, 150)), (panel_x + 20, hist_y_start + 60))

            else:
                self.screen.blit(self.font.render("GLOBAL INSPECTOR", True, COLOR_TEXT), (panel_x + 20, 20))
                pygame.draw.line(self.screen, (100, 100, 100), (panel_x + 20, 45), (panel_x + SIDEBAR_RIGHT_WIDTH - 20, 45), 2)
                
                y_pos = 60
                for idx, couche in enumerate(self.layers):
                    pygame.draw.rect(self.screen, (60, 60, 60), (panel_x + 10, y_pos, SIDEBAR_RIGHT_WIDTH - 20, 50), border_radius=5)
                    self.screen.blit(self.font.render(f"{couche.name} ({couche.nodes} N)", True, COLOR_TEXT), (panel_x + 20, y_pos + 15))
                    if couche.name != "Input":
                        act_rect = pygame.Rect(panel_x + 120, y_pos + 12, 90, 26)
                        pygame.draw.rect(self.screen, DICT_COULEURS_ACTIVATION.get(couche.activation, (100, 100, 100)), act_rect, border_radius=4)
                        self.screen.blit(self.font.render(couche.activation, True, COLOR_TEXT), (act_rect.x + 10, act_rect.y + 4))
                        self.inspector_buttons.append({'rect': act_rect, 'action': 'change_activation', 'layer_idx': idx})
                    
                    del_rect = pygame.Rect(panel_x + 220, y_pos + 12, 26, 26)
                    pygame.draw.rect(self.screen, COLOR_BTN_MINUS, del_rect, border_radius=4)
                    self.screen.blit(self.font.render("X", True, COLOR_TEXT), (del_rect.x + 8, del_rect.y + 3))
                    self.inspector_buttons.append({'rect': del_rect, 'action': 'delete', 'layer_idx': idx})
                    y_pos += 60
        else:
            self.btn_toggle_rect.x = WINDOW_WIDTH - 30

        pygame.draw.rect(self.screen, COLOR_BTN, self.btn_toggle_rect, border_bottom_left_radius=8, border_top_left_radius=8)
        self.screen.blit(self.font.render(">" if self.right_panel_open else "<", True, COLOR_TEXT), (self.btn_toggle_rect.x + 8, self.btn_toggle_rect.y + 5))

    def run(self):
        while True:
            self.handle_events()
            self.screen.fill(COLOR_BG)
            if self.current_view == "Build":
                self.draw_network()
                pygame.draw.rect(self.screen, COLOR_SIDEBAR, (0, 0, SIDEBAR_LEFT_WIDTH, WINDOW_HEIGHT))
                for tab in self.tabs: tab.draw(self.screen, self.font)
                for item in self.ui_items:
                    if item != self.active_drag: item.draw(self.screen, self.font)
                for hp in reversed(self.global_hyperparams): hp.draw(self.screen, self.font, self.small_font)
                if self.active_drag: self.active_drag.draw(self.screen, self.font)
                self.draw_right_panel()
            elif self.current_view == "Data":
                pygame.draw.rect(self.screen, COLOR_SIDEBAR, (0, 0, SIDEBAR_LEFT_WIDTH, WINDOW_HEIGHT))
                for tab in self.tabs: tab.draw(self.screen, self.font)
                self.draw_data_view()
                self.draw_right_panel()
            elif self.current_view == "Maths":
                pygame.draw.rect(self.screen, COLOR_SIDEBAR, (0, 0, SIDEBAR_LEFT_WIDTH, WINDOW_HEIGHT))
                for tab in self.tabs: tab.draw(self.screen, self.font)
                self.draw_math_view()
                self.draw_right_panel()
            elif self.current_view == "Focus":
                self.draw_focus_view()
            pygame.display.flip()
            self.clock.tick(FPS)