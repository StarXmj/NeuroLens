# graphics/ui.py
import pygame
from config.settings import *

class Tab:
    def __init__(self, x, y, width, height, name, is_active=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.name = name
        self.is_active = is_active

    def draw(self, screen, font):
        color = COLOR_TAB_ACTIVE if self.is_active else COLOR_TAB_INACTIVE
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        text = font.render(self.name, True, COLOR_TEXT)
        screen.blit(text, (self.rect.x + 10, self.rect.y + 5))

class DraggableItem:
    def __init__(self, x, y, width, height, name, default_nodes, activations):
        self.rect = pygame.Rect(x, y, width, height)
        self.name = name
        self.default_nodes = default_nodes
        
        # --- GESTION DU MENU RETRACTABLE ---
        self.activations = activations
        self.selected_act = activations[0] if activations else "Entrée"
        self.is_expanded = False
        self.arrow_rect = pygame.Rect(x + 10, y + 10, 20, 20) # Zone cliquable de la flèche à gauche
        self.option_rects = []
        
        self.is_dragging = False
        self.offset_x, self.offset_y = 0, 0
        self.start_x, self.start_y = x, y
        
        self.btn_minus = pygame.Rect(x + 50, y + 35, 20, 20)
        self.btn_plus = pygame.Rect(x + width - 30, y + 35, 20, 20)

    def update_position(self, x, y):
        self.rect.x, self.rect.y = x, y
        self.arrow_rect.x, self.arrow_rect.y = x + 10, y + 10
        self.btn_minus.x, self.btn_minus.y = x + 50, y + 35
        self.btn_plus.x, self.btn_plus.y = x + self.rect.width - 30, y + 35
        
        # Recalcule la position des options du menu si ouvert
        self.option_rects = []
        if self.is_expanded:
            for i in range(len(self.activations)):
                self.option_rects.append(
                    pygame.Rect(x + 10, y + self.rect.height + 5 + (i * 28), self.rect.width - 20, 25)
                )

    def draw(self, screen, font):
        # 1. Le bloc principal
        pygame.draw.rect(screen, COLOR_DRAG_ITEM, self.rect, border_radius=8)
        
        # 2. La petite flèche géométrique
        if self.activations:
            cx, cy = self.arrow_rect.x + 5, self.arrow_rect.y + 5
            if self.is_expanded: # Flèche vers le bas
                pygame.draw.polygon(screen, COLOR_TEXT, [(cx, cy+2), (cx+12, cy+2), (cx+6, cy+10)])
            else: # Flèche vers la droite
                pygame.draw.polygon(screen, COLOR_TEXT, [(cx+2, cy-2), (cx+10, cy+4), (cx+2, cy+10)])
        
        # 3. Textes
        title = font.render(f"{self.name} : {self.selected_act}", True, COLOR_TEXT)
        screen.blit(title, (self.rect.x + 35, self.rect.y + 7))
        
        nodes_text = font.render(f"Neurones: {self.default_nodes}", True, COLOR_TEXT)
        screen.blit(nodes_text, (self.rect.x + 80, self.rect.y + 35))
        
        # 4. Boutons + / -
        pygame.draw.rect(screen, COLOR_BTN_MINUS, self.btn_minus, border_radius=4)
        screen.blit(font.render("-", True, COLOR_TEXT), (self.btn_minus.x + 6, self.btn_minus.y - 2))
        pygame.draw.rect(screen, COLOR_BTN_PLUS, self.btn_plus, border_radius=4)
        screen.blit(font.render("+", True, COLOR_TEXT), (self.btn_plus.x + 4, self.btn_plus.y - 2))
        
        # 5. Dessin du sous-menu déroulant (seulement s'il n'est pas en train d'être glissé)
        if self.is_expanded and not self.is_dragging:
            for i, rect in enumerate(self.option_rects):
                act_name = self.activations[i]
                # Utilise la couleur du dictionnaire pour le fond du bouton !
                couleur_fond = DICT_COULEURS_ACTIVATION.get(act_name, COLOR_DROPDOWN)
                pygame.draw.rect(screen, couleur_fond, rect, border_radius=4)
                
                act_text = font.render(act_name, True, COLOR_TEXT)
                screen.blit(act_text, (rect.x + 10, rect.y + 2))

class TextInput:
    def __init__(self, x, y, width, height, name, text_default="", input_type="float", disabled=False, help_text=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.name = name
        self.text = str(text_default)
        self.active = False
        self.color_active = (100, 150, 200)
        self.color_inactive = (60, 60, 60)
        self.color_disabled = (40, 40, 40)
        self.input_type = input_type
        self.disabled = disabled
        self.help_text = help_text # Le texte pédagogique en dessous

    def handle_event(self, event):
        if self.disabled: return # Si l'option est grisée, on ne fait rien
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
                
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                char = event.unicode
                if self.input_type == "int" and char.isdigit():
                    self.text += char
                elif self.input_type == "float" and (char.isdigit() or (char == '.' and '.' not in self.text)):
                    self.text += char

    def draw(self, screen, font, small_font):
        # Titre au-dessus
        text_color = (120, 120, 120) if self.disabled else (200, 200, 200)
        screen.blit(small_font.render(self.name, True, text_color), (self.rect.x, self.rect.y - 18))
        
        # Boîte de saisie
        if self.disabled:
            color = self.color_disabled
        else:
            color = self.color_active if self.active else self.color_inactive
            
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        
        # Texte à l'intérieur
        txt_color = (100, 100, 100) if self.disabled else (255, 255, 255)
        txt_surface = small_font.render(self.text, True, txt_color)
        screen.blit(txt_surface, (self.rect.x + 5, self.rect.y + 4))
        
        # Affichage du texte pédagogique en dessous si présent
        if self.help_text:
            help_surface = small_font.render(self.help_text, True, (140, 140, 140))
            screen.blit(help_surface, (self.rect.x, self.rect.y + self.rect.height + 2))
class Dropdown:
    def __init__(self, x, y, width, height, name, options):
        self.rect = pygame.Rect(x, y, width, height)
        self.name = name
        self.options = options
        self.selected = options[0]
        self.is_expanded = False
        self.option_rects = []
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_expanded:
                for i, r in enumerate(self.option_rects):
                    if r.collidepoint(event.pos):
                        self.selected = self.options[i]
                        self.is_expanded = False
                        return
                self.is_expanded = False # Clic ailleurs = on ferme
            elif self.rect.collidepoint(event.pos):
                self.is_expanded = True

    def draw(self, screen, font, small_font):
        screen.blit(small_font.render(self.name, True, (200, 200, 200)), (self.rect.x, self.rect.y - 18))
        pygame.draw.rect(screen, (60, 90, 130), self.rect, border_radius=4)
        screen.blit(small_font.render(self.selected + (" ▲" if self.is_expanded else " ▼"), True, (255,255,255)), (self.rect.x + 5, self.rect.y + 4))
        
        # Dessin des options par-dessus le reste
        self.option_rects = []
        if self.is_expanded:
            for i, opt in enumerate(self.options):
                r = pygame.Rect(self.rect.x, self.rect.y + self.rect.height + i * 25, self.rect.width, 25)
                self.option_rects.append(r)
                pygame.draw.rect(screen, (80, 110, 150), r)
                pygame.draw.rect(screen, (40, 60, 90), r, 1) # Bordure
                screen.blit(small_font.render(opt, True, (255,255,255)), (r.x + 5, r.y + 4))

class Checkbox:
    def __init__(self, x, y, size, name, checked=False):
        self.rect = pygame.Rect(x, y, size, size)
        self.name = name
        self.checked = checked
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.checked = not self.checked
            
    def draw(self, screen, font, small_font):
        pygame.draw.rect(screen, (200, 200, 200), self.rect, border_radius=3)
        if self.checked:
            pygame.draw.rect(screen, (50, 200, 50), (self.rect.x+3, self.rect.y+3, self.rect.width-6, self.rect.height-6), border_radius=2)
        screen.blit(small_font.render(self.name, True, (200, 200, 200)), (self.rect.x + self.rect.width + 10, self.rect.y))