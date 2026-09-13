# graphics/ui.py
import pygame

class Tab:
    def __init__(self, x, y, w, h, name, is_active=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.name = name
        self.is_active = is_active

    def draw(self, surface, font):
        color = (100, 100, 100) if self.is_active else (60, 60, 60)
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        text = font.render(self.name, True, (255, 255, 255))
        surface.blit(text, (self.rect.x + (self.rect.width - text.get_width())//2, self.rect.y + 5))

class DraggableItem:
    def __init__(self, x, y, w, h, name, default_nodes, activations):
        self.rect = pygame.Rect(x, y, w, h)
        self.start_x, self.start_y = x, y
        self.name = name
        self.default_nodes = default_nodes
        self.activations = activations
        self.selected_act = activations[0] if activations else ""
        self.is_expanded = False
        self.is_dragging = False
        self.offset_x, self.offset_y = 0, 0
        
        self.btn_minus = pygame.Rect(self.rect.x + 10, self.rect.y + 35, 24, 24)
        self.btn_plus = pygame.Rect(self.rect.x + self.rect.width - 34, self.rect.y + 35, 24, 24)
        self.arrow_rect = pygame.Rect(self.rect.right - 25, self.rect.y + 5, 20, 20)
        self.option_rects = []

    def update_position(self, x, y):
        self.rect.x, self.rect.y = x, y
        self.btn_minus.x, self.btn_minus.y = x + 10, y + 35
        self.btn_plus.x, self.btn_plus.y = x + self.rect.width - 34, y + 35
        self.arrow_rect.x, self.arrow_rect.y = self.rect.right - 25, y + 5

    def draw(self, surface, font):
        pygame.draw.rect(surface, (70, 130, 180), self.rect, border_radius=8)
        title = f"{self.name} : {self.selected_act}" if self.selected_act else self.name
        surface.blit(font.render(title, True, (255, 255, 255)), (self.rect.x + 10, self.rect.y + 5))
        
        if self.activations:
            arrow = "v" if self.is_expanded else ">"
            surface.blit(font.render(arrow, True, (255, 255, 255)), (self.arrow_rect.x + 5, self.arrow_rect.y))

        pygame.draw.rect(surface, (200, 50, 50), self.btn_minus, border_radius=4)
        surface.blit(font.render("-", True, (255, 255, 255)), (self.btn_minus.x + 8, self.btn_minus.y + 2))
        
        nodes_txt = font.render(f"Neurons: {self.default_nodes}", True, (255, 255, 255))
        text_x = self.btn_minus.right + ((self.btn_plus.left - self.btn_minus.right) - nodes_txt.get_width()) // 2
        surface.blit(nodes_txt, (text_x, self.rect.y + 37))
        
        pygame.draw.rect(surface, (50, 200, 50), self.btn_plus, border_radius=4)
        surface.blit(font.render("+", True, (255, 255, 255)), (self.btn_plus.x + 6, self.btn_plus.y + 2))

        if self.is_expanded:
            self.option_rects = []
            oy = self.rect.bottom + 2
            for act in self.activations:
                r = pygame.Rect(self.rect.x, oy, self.rect.width, 25)
                pygame.draw.rect(surface, (100, 100, 120), r, border_radius=4)
                surface.blit(font.render(act, True, (255,255,255)), (r.x + 10, r.y + 2))
                self.option_rects.append(r)
                oy += 27

class TextInput:
    def __init__(self, x, y, w, h, label, default_text, val_type="float", disabled=False, help_text=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.text = default_text
        self.val_type = val_type
        self.active = False
        self.disabled = disabled
        self.help_text = help_text
        self.first_edit = False

    def handle_event(self, event):
        if self.disabled: return
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            if self.active and not was_active:
                self.first_edit = True # Active la purge au premier caractère tapé
                
        if event.type == pygame.KEYDOWN and self.active:
            if self.first_edit and event.key not in (pygame.K_BACKSPACE, pygame.K_RETURN, pygame.K_ESCAPE):
                self.text = "" # Efface automatiquement l'ancien texte
            self.first_edit = False
            
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def draw(self, surface, font, small_font):
        if self.label:
            surface.blit(small_font.render(self.label, True, (200, 200, 200)), (self.rect.x, self.rect.y - 15))
        color = (100, 100, 150) if self.active else (60, 60, 60)
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        txt_surf = font.render(self.text, True, (255, 255, 255) if not self.disabled else (150,150,150))
        surface.blit(txt_surf, (self.rect.x + 5, self.rect.y + 2))
        if self.disabled and self.help_text:
            surface.blit(small_font.render(self.help_text, True, (150, 150, 150)), (self.rect.x, self.rect.bottom + 2))

class Dropdown:
    def __init__(self, x, y, w, h, label, options):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.options = options
        self.selected = options[0]
        self.is_open = False
        self.option_rects = []

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_open:
                for i, r in enumerate(self.option_rects):
                    if r.collidepoint(event.pos):
                        self.selected = self.options[i]
                        self.is_open = False
                        return
                self.is_open = False
            elif self.rect.collidepoint(event.pos):
                self.is_open = True

    def draw(self, surface, font, small_font):
        if self.label:
            surface.blit(small_font.render(self.label, True, (200, 200, 200)), (self.rect.x, self.rect.y - 15))
        pygame.draw.rect(surface, (70, 100, 150), self.rect, border_radius=4)
        surface.blit(font.render(f"{self.selected} v", True, (255, 255, 255)), (self.rect.x + 5, self.rect.y + 2))
        
        if self.is_open:
            self.option_rects = []
            oy = self.rect.bottom
            for opt in self.options:
                r = pygame.Rect(self.rect.x, oy, self.rect.width, self.rect.height)
                pygame.draw.rect(surface, (80, 110, 160), r)
                pygame.draw.rect(surface, (50, 50, 50), r, 1)
                surface.blit(font.render(opt, True, (255, 255, 255)), (r.x + 5, r.y + 2))
                self.option_rects.append(r)
                oy += self.rect.height

class Checkbox:
    def __init__(self, x, y, size, label, checked=False):
        self.rect = pygame.Rect(x, y, size, size)
        self.label = label
        self.checked = checked

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.checked = not self.checked

    def draw(self, surface, font, small_font):
        pygame.draw.rect(surface, (200, 200, 200), self.rect, border_radius=3)
        if self.checked:
            pygame.draw.rect(surface, (50, 200, 50), self.rect.inflate(-4, -4), border_radius=2)
        surface.blit(small_font.render(self.label, True, (200, 200, 200)), (self.rect.right + 10, self.rect.y))