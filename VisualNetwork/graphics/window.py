# graphics/window.py
import pygame
import sys
import math
import torch
import torch.nn as nn
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
        pygame.display.set_caption("Visualisateur - Panneau de droite restauré")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 12)
        self.title_font = pygame.font.SysFont("Arial", 28, bold=True)
        
        self.layers = []
        self.links = [] 
        self.pytorch_engine = NeuralNetworkEngine()
        
        self.active_drag = None
        self.right_panel_open = True
        self.btn_toggle_rect = pygame.Rect(WINDOW_WIDTH - SIDEBAR_RIGHT_WIDTH - 30, 10, 30, 30)
        
        self.tabs = [
            Tab(10, 10, 95, 30, "Création", is_active=True), 
            Tab(115, 10, 95, 30, "Données", is_active=False)
        ]
        self.current_view = "Création"
        
        self.ui_items = [
            DraggableItem(20, 60, 180, 65, "Entrée", 4, []), 
            DraggableItem(20, 140, 180, 65, "Cachée", 5, ["ReLU", "Sigmoid", "Tanh", "Linear"]),
            DraggableItem(20, 220, 180, 65, "Sortie", 2, ["Softmax", "Sigmoid", "Linear"])
        ]
        
        start_y = 310
        esp = 48 
        self.global_hyperparams = [
            TextInput(20, start_y, 180, 22, "Taux apprentissage (LR)", "0.01", "float"),
            TextInput(20, start_y + esp*1, 180, 22, "Taille du lot (Batch)", "32", "int", disabled=True, help_text="Mode éducation : non requis"),
            TextInput(20, start_y + esp*2, 180, 22, "Nombre d'Epochs", "100", "int"),
            Dropdown(20, start_y + esp*3, 180, 22, "Fonction de perte", ["MSE", "CrossEntropy", "MAE"]),
            Dropdown(20, start_y + esp*4, 180, 22, "Métrique d'évaluation", ["Accuracy", "Loss", "Precision"]),
            TextInput(20, start_y + esp*5, 180, 22, "Dropout [ex: 0.20]", "0.0", "float"),
            Checkbox(20, start_y + esp*6 + 10, 16, "Batch Normalisation", False),
            Dropdown(20, start_y + esp*7 + 10, 180, 22, "Pénalité", ["Aucune", "L1", "L2"])
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
        self.btn_mode_exp = pygame.Rect(0, 0, 90, 26)
        self.btn_play = pygame.Rect(0, 0, 70, 30)
        self.btn_reset = pygame.Rect(0, 0, 70, 30)
        self.btn_slow = pygame.Rect(0, 0, 40, 30)
        self.btn_fast = pygame.Rect(0, 0, 40, 30)

        self.last_click_time = 0
        self.hovered_item = None
        self.selected_item = None 
        self.focus_target = None
        self.inspector_value_input = TextInput(0, 0, 150, 25, "Valeur", "0.0", "float") 
        
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
        input_layer = next((l for l in self.layers if l.name == "Entrée"), None)
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
        input_layer = next((l for l in self.layers if l.name == "Entrée"), None)
        if not input_layer or len(self.data_grid) != input_layer.nodes: return False
        for row in self.data_grid:
            try:
                float(row[0].text.strip())
                float(row[1].text.strip())
            except ValueError:
                return False
        return True

    def run_pytorch_experience(self):
        if not self.dataset_transferred or len(self.layers) < 2:
            print("⚠️ [EXPÉRIENCE] Transférez d'abord des données valides.")
            self.is_running = False
            return

        print("🚀 [EXPÉRIENCE] Entraînement PyTorch en cours...")
        X_data = [float(row[0].text.strip()) for row in self.data_grid]
        Y_data = [float(row[1].text.strip()) for row in self.data_grid]
        X = torch.tensor(X_data, dtype=torch.float32).unsqueeze(1)
        Y = torch.tensor(Y_data, dtype=torch.float32).unsqueeze(1)

        in_features = self.layers[0].nodes
        self.pytorch_engine.build_model(self.layers, in_features)

        try:
            lr = float(self.global_hyperparams[0].text)
        except ValueError:
            lr = 0.01

        epochs = self.get_target_epochs()
        loss_name = self.global_hyperparams[3].selected

        self.pytorch_engine.train_network(self.layers, X, Y, lr, epochs, loss_name)
        self.current_epoch = epochs

        linear_layers_idx = 0
        for idx in range(1, len(self.layers)):
            prev_layer = self.layers[idx - 1]
            curr_layer = self.layers[idx]
            while linear_layers_idx < len(self.pytorch_engine.model) and not isinstance(self.pytorch_engine.model[linear_layers_idx], nn.Linear):
                linear_layers_idx += 1
            if linear_layers_idx < len(self.pytorch_engine.model):
                torch_layer = self.pytorch_engine.model[linear_layers_idx]
                weight_tensor = torch_layer.weight.detach()
                
                for j, n_curr in enumerate(curr_layer.neurons):
                    for i, n_prev in enumerate(prev_layer.neurons):
                        link = next((l for l in self.links if l.n1 == n_prev and l.n2 == n_curr), None)
                        if link: 
                            link.value = round(float(weight_tensor[j, i].item()), 2)
                linear_layers_idx += 1

        activations_dict = self.pytorch_engine.get_forward_activations(self.layers)
        for idx, layer in enumerate(self.layers):
            if idx in activations_dict:
                vals = activations_dict[idx]
                for n_idx, neuron in enumerate(layer.neurons):
                    if n_idx < len(vals): neuron.value = round(float(vals[n_idx]), 2)

        print("✅ [EXPÉRIENCE] Entraînement terminé avec succès !")
        self.is_running = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if self.current_view == "Création":
                for hp in self.global_hyperparams: hp.handle_event(event)
                if self.selected_item: self.inspector_value_input.handle_event(event)
            elif self.current_view == "Données":
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

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_mode_anim.collidepoint(event.pos): self.sim_mode = "ANIMATION"; return
                if self.btn_mode_exp.collidepoint(event.pos): self.sim_mode = "EXPERIENCE"; return

                if self.current_view == "Données" and self.btn_transfer.collidepoint(event.pos):
                    if self.is_dataset_valid():
                        input_layer = next((l for l in self.layers if l.name == "Entrée"), None)
                        if input_layer:
                            for i, row in enumerate(self.data_grid):
                                input_layer.neurons[i].value = float(row[0].text.strip())
                        self.dataset_transferred = True
                    return

                if self.btn_play.collidepoint(event.pos):
                    if self.sim_mode == "ANIMATION": 
                        self.is_running = not self.is_running
                    else: 
                        self.is_running = True
                        self.run_pytorch_experience()
                    return

                if self.btn_reset.collidepoint(event.pos):
                    self.is_running = False; self.current_epoch = 0; self.animation_progress = 0.0; return

                if self.current_view == "Focus" and self.btn_retour.collidepoint(event.pos):
                    self.current_view = "Création"; return

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

                if self.current_view == "Création" and event.pos[0] > SIDEBAR_LEFT_WIDTH and event.pos[0] < WINDOW_WIDTH - (SIDEBAR_RIGHT_WIDTH if self.right_panel_open else 0):
                    current_time = pygame.time.get_ticks()
                    if self.hovered_item:
                        self.selected_item = self.hovered_item
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

                if self.current_view == "Création":
                    for item in self.ui_items:
                        if item.btn_minus.collidepoint(event.pos): item.default_nodes = max(1, item.default_nodes - 1); self.update_data_grid(); return
                        elif item.btn_plus.collidepoint(event.pos): item.default_nodes = min(15, item.default_nodes + 1); self.update_data_grid(); return
                        elif item.rect.collidepoint(event.pos):
                            self.active_drag = item; item.is_dragging = True
                            item.offset_x, item.offset_y = item.rect.x - event.pos[0], item.rect.y - event.pos[1]
                            break

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.active_drag:
                if self.active_drag.rect.x > SIDEBAR_LEFT_WIDTH: self._add_layer_from_drag()
                self.active_drag.update_position(self.active_drag.start_x, self.active_drag.start_y)
                self.active_drag.is_dragging = False; self.active_drag = None

            elif event.type == pygame.MOUSEMOTION and self.active_drag:
                self.active_drag.update_position(event.pos[0] + self.active_drag.offset_x, event.pos[1] + self.active_drag.offset_y)

    def _add_layer_from_drag(self):
        nom, nodes, act = self.active_drag.name, self.active_drag.default_nodes, self.active_drag.selected_act
        if (nom == "Entrée" and any(l.name == "Entrée" for l in self.layers)) or (nom == "Sortie" and any(l.name == "Sortie" for l in self.layers)): return
        nouvelle_couche = Layer(nom, nodes=nodes, activation=act)
        if nom == "Entrée": self.layers.insert(0, nouvelle_couche)
        elif nom == "Sortie": self.layers.append(nouvelle_couche)
        else:
            if any(l.name == "Sortie" for l in self.layers): self.layers.insert(-1, nouvelle_couche)
            else: self.layers.append(nouvelle_couche)
        self.rebuild_links()

    def update_simulation(self):
        if not self.is_running or not self.layers or self.sim_mode == "EXPERIENCE": return
        if self.current_epoch >= self.get_target_epochs(): self.is_running = False; return
        self.animation_progress += 0.02 * self.simulation_speed
        if self.animation_progress >= 1.0:
            self.animation_progress = 0.0
            if self.sim_phase == "FORWARD": self.sim_phase = "BACKWARD"
            else:
                self.sim_phase = "FORWARD"; self.current_epoch += 1

    def get_neuron_layer_and_color(self, neuron):
        for couche in self.layers:
            if neuron in couche.neurons: return couche, DICT_COULEURS_ACTIVATION.get(couche.activation, (150, 150, 150))
        return None, (150, 150, 150)

    def draw_network(self):
        self.update_simulation()
        espace_gauche, espace_droit = SIDEBAR_LEFT_WIDTH, SIDEBAR_RIGHT_WIDTH if self.right_panel_open else 0
        net_width = WINDOW_WIDTH - espace_gauche - espace_droit
        self.hovered_item = None
        if not self.layers: return
        espacement_x = net_width // (len(self.layers) + 1)
        mx, my = pygame.mouse.get_pos()

        for idx, couche in enumerate(self.layers):
            x = espace_gauche + (idx + 1) * espacement_x
            ey = WINDOW_HEIGHT // (couche.nodes + 1)
            for i, n in enumerate(couche.neurons): n.x, n.y = x, (i + 1) * ey

        for l in self.links:
            x1, y1, x2, y2 = l.n1.x, l.n1.y, l.n2.x, l.n2.y
            dist = distance_point_ligne(mx, my, x1, y1, x2, y2)
            couleur_trait = (200, 50, 50) if l.value < 0 else ((50, 150, 255) if l.value > 0 else (100, 100, 100))
            is_hovered = dist < 6
            if is_hovered or (self.selected_item and self.selected_item['obj'] == l):
                pygame.draw.line(self.screen, (255, 255, 150), (x1, y1), (x2, y2), 4)
            else: pygame.draw.line(self.screen, couleur_trait, (x1, y1), (x2, y2), 1)
            
            if is_hovered: self.hovered_item = {'type': 'lien', 'obj': l}

            if self.is_running and self.sim_mode == "ANIMATION":
                for offset in [0.0, 0.33, 0.66]:
                    prog = (self.animation_progress + offset) % 1.0
                    if self.sim_phase == "BACKWARD": prog = 1.0 - prog
                    px = x1 + (x2 - x1) * prog
                    py = y1 + (y2 - y1) * prog
                    couleur_particule = (50, 255, 50) if self.sim_phase == "FORWARD" else (255, 165, 0)
                    pygame.draw.circle(self.screen, couleur_particule, (int(px), int(py)), 4)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            txt_val = self.small_font.render(f"{l.value}", True, (200,200,200))
            self.screen.blit(txt_val, (cx - txt_val.get_width()//2, cy - 15))

        for couche in self.layers:
            col = DICT_COULEURS_ACTIVATION.get(couche.activation, (150, 150, 150))
            for n in couche.neurons:
                is_hovered = math.hypot(mx - n.x, my - n.y) < 22
                if is_hovered: self.hovered_item = {'type': 'neurone', 'obj': n, 'couche': couche, 'couleur': col}
                
                if is_hovered or (self.selected_item and self.selected_item['obj'] == n):
                    pygame.draw.circle(self.screen, (255, 255, 255), (n.x, n.y), 24)

                pygame.draw.circle(self.screen, col, (n.x, n.y), 20)
                pygame.draw.circle(self.screen, (0, 0, 0), (n.x, n.y), 20, 2)
                txt_val = self.small_font.render(f"{n.value}", True, (0,0,0))
                self.screen.blit(txt_val, (n.x - txt_val.get_width()//2, n.y - txt_val.get_height()//2))

        self.draw_simulation_hud(espace_gauche, net_width)

    def draw_simulation_hud(self, espace_gauche, net_width):
        hud_y = 10
        center_x = espace_gauche + net_width // 2
        max_epochs = self.get_target_epochs()

        self.btn_mode_anim.topleft = (center_x - 95, hud_y)
        self.btn_mode_exp.topleft = (center_x + 5, hud_y)

        pygame.draw.rect(self.screen, (50, 130, 200) if self.sim_mode == "ANIMATION" else (60, 60, 60), self.btn_mode_anim, border_radius=4)
        self.screen.blit(self.small_font.render("Animation", True, (255,255,255)), (self.btn_mode_anim.x + 15, self.btn_mode_anim.y + 5))

        pygame.draw.rect(self.screen, (200, 100, 50) if self.sim_mode == "EXPERIENCE" else (60, 60, 60), self.btn_mode_exp, border_radius=4)
        self.screen.blit(self.small_font.render("Expérience", True, (255,255,255)), (self.btn_mode_exp.x + 12, self.btn_mode_exp.y + 5))

        status_y = hud_y + 32
        statut_txt = f"MODE : {self.sim_mode} (EN COURS...)" if self.is_running else f"MODE : {self.sim_mode} (PRÊT)"
        couleur_statut = (50, 250, 50) if self.is_running else (150, 150, 150)
        
        txt_surface = self.small_font.render(statut_txt, True, couleur_statut)
        self.screen.blit(txt_surface, (center_x - txt_surface.get_width() // 2, status_y))

        epoch_surface = self.small_font.render(f"Epoch : {self.current_epoch} / {max_epochs}", True, (255, 255, 255))
        self.screen.blit(epoch_surface, (center_x - epoch_surface.get_width() // 2, status_y + 18))

        btn_y = status_y + 40
        self.btn_play.topleft = (center_x - 145, btn_y)
        self.btn_reset.topleft = (center_x - 65, btn_y)
        
        play_label = "Calculer" if self.sim_mode == "EXPERIENCE" else ("Pause" if self.is_running else "Départ")
        pygame.draw.rect(self.screen, (50, 180, 80) if not self.is_running else (200, 80, 50), self.btn_play, border_radius=4)
        self.screen.blit(self.small_font.render(play_label, True, (255,255,255)), (self.btn_play.x + 10, self.btn_play.y + 6))

        pygame.draw.rect(self.screen, (100, 100, 100), self.btn_reset, border_radius=4)
        self.screen.blit(self.small_font.render("Reset", True, (255,255,255)), (self.btn_reset.x + 18, self.btn_reset.y + 6))

        if self.sim_mode == "ANIMATION":
            self.btn_slow.topleft = (center_x + 15, btn_y)
            self.btn_fast.topleft = (center_x + 60, btn_y)
            
            pygame.draw.rect(self.screen, (70, 70, 90), self.btn_slow, border_radius=4)
            self.screen.blit(self.small_font.render("-", True, (255,255,255)), (self.btn_slow.x + 15, self.btn_slow.y + 5))

            spd_surface = self.small_font.render(f"{self.simulation_speed}x", True, (200, 200, 200))
            self.screen.blit(spd_surface, (center_x + 105, btn_y + 6))

            pygame.draw.rect(self.screen, (70, 70, 90), self.btn_fast, border_radius=4)
            self.screen.blit(self.small_font.render("+", True, (255,255,255)), (self.btn_fast.x + 14, self.btn_fast.y + 5))

    def draw_focus_view(self):
        self.update_simulation()
        pygame.draw.rect(self.screen, COLOR_DRAG_ITEM, self.btn_retour, border_radius=5)
        self.screen.blit(self.font.render("< Retour", True, COLOR_TEXT), (self.btn_retour.x + 15, self.btn_retour.y + 10))
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2

        if self.focus_target['type'] == 'neurone':
            n, couche, couleur = self.focus_target['obj'], self.focus_target['couche'], self.focus_target['couleur']
            pygame.draw.circle(self.screen, couleur, (cx, cy), 150)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), 150, 8)
            txt_id = self.title_font.render(f"ZOOM - {n.id} (Couche: {couche.name})", True, COLOR_TEXT)
            self.screen.blit(txt_id, (cx - txt_id.get_width()//2, cy - 250))
            txt_val = self.title_font.render(f"Valeur : {n.value}", True, (0,0,0))
            self.screen.blit(txt_val, (cx - txt_val.get_width()//2, cy - 20))

        elif self.focus_target['type'] == 'lien':
            l = self.focus_target['obj']
            _, col1 = self.get_neuron_layer_and_color(l.n1)
            _, col2 = self.get_neuron_layer_and_color(l.n2)
            pygame.draw.line(self.screen, (200, 200, 200), (cx - 200, cy), (cx + 200, cy), 15)

            if self.is_running and self.sim_mode == "ANIMATION":
                for offset in [0.0, 0.33, 0.66]:
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
            
            txt = self.title_font.render(f"Poids du lien {l.id} : {l.value}", True, COLOR_TEXT)
            self.screen.blit(txt, (cx - txt.get_width()//2, cy - 150))

    def draw_data_view(self):
        start_x, start_y = SIDEBAR_LEFT_WIDTH + 60, 60
        self.screen.blit(self.title_font.render("Jeu de Données (Régression Simple)", True, COLOR_TEXT), (start_x, start_y))
        
        self.screen.blit(self.font.render("Entrée (X)", True, (150, 150, 255)), (start_x, start_y + 45))
        self.screen.blit(self.font.render("Cible (Y)", True, (255, 150, 150)), (start_x + 90, start_y + 45))

        input_layer = next((l for l in self.layers if l.name == "Entrée"), None)
        if not input_layer:
            err_txt = self.font.render("⚠️ IMPOSSIBLE : Aucune couche d'entrée créée !", True, (255, 60, 60))
            self.screen.blit(err_txt, (start_x, start_y + 80))
        else:
            for r, row in enumerate(self.data_grid):
                for cell in row: 
                    cell.rect.y = start_y + 80 + (r * 30)
                    cell.draw(self.screen, self.font, self.small_font)
            
            self.btn_transfer.y = start_y + len(self.data_grid) * 30 + 130
            pygame.draw.rect(self.screen, (50, 180, 80) if self.dataset_transferred else (70, 130, 180), self.btn_transfer, border_radius=6)
            self.screen.blit(self.font.render("✓ Transféré !" if self.dataset_transferred else "Transférer au réseau", True, COLOR_TEXT), (self.btn_transfer.x + 35, self.btn_transfer.y + 10))

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
                
                type_str = "Neurone" if self.selected_item['type'] == 'neurone' else "Lien / Poids"
                txt_type = self.small_font.render(type_str, True, (180, 180, 180))
                self.screen.blit(txt_type, (panel_x + 110, 26))

                pygame.draw.line(self.screen, (100, 100, 100), (panel_x + 20, 60), (panel_x + SIDEBAR_RIGHT_WIDTH - 20, 60), 2)
                
                self.inspector_value_input.rect.x = panel_x + 20
                self.inspector_value_input.rect.y = 95
                self.inspector_value_input.draw(self.screen, self.font, self.small_font)
                
            else:
                self.screen.blit(self.font.render("INSPECTEUR GLOBAL", True, COLOR_TEXT), (panel_x + 20, 20))
                pygame.draw.line(self.screen, (100, 100, 100), (panel_x + 20, 45), (panel_x + SIDEBAR_RIGHT_WIDTH - 20, 45), 2)
                
                y_pos = 60
                for idx, couche in enumerate(self.layers):
                    pygame.draw.rect(self.screen, (60, 60, 60), (panel_x + 10, y_pos, SIDEBAR_RIGHT_WIDTH - 20, 50), border_radius=5)
                    self.screen.blit(self.font.render(f"{couche.name} ({couche.nodes} N)", True, COLOR_TEXT), (panel_x + 20, y_pos + 15))
                    if couche.name != "Entrée":
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
            if self.current_view == "Création":
                self.draw_network()
                pygame.draw.rect(self.screen, COLOR_SIDEBAR, (0, 0, SIDEBAR_LEFT_WIDTH, WINDOW_HEIGHT))
                for tab in self.tabs: tab.draw(self.screen, self.font)
                for item in self.ui_items:
                    if item != self.active_drag: item.draw(self.screen, self.font)
                for hp in reversed(self.global_hyperparams): hp.draw(self.screen, self.font, self.small_font)
                if self.active_drag: self.active_drag.draw(self.screen, self.font)
                self.draw_right_panel()
            elif self.current_view == "Données":
                pygame.draw.rect(self.screen, COLOR_SIDEBAR, (0, 0, SIDEBAR_LEFT_WIDTH, WINDOW_HEIGHT))
                for tab in self.tabs: tab.draw(self.screen, self.font)
                self.draw_data_view()
                self.draw_right_panel()
            elif self.current_view == "Focus":
                self.draw_focus_view()
            pygame.display.flip()
            self.clock.tick(FPS)