# config/settings.py

DEFAULT_NODES = 3

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

# Dimensions des menus
SIDEBAR_LEFT_WIDTH = 220
SIDEBAR_RIGHT_WIDTH = 280

FPS = 60

# --- COULEURS ---
COLOR_BG = (30, 30, 30)
COLOR_SIDEBAR = (45, 45, 45)
COLOR_DRAG_ITEM = (70, 130, 180)
COLOR_TEXT = (255, 255, 255)
COLOR_NODE = (200, 200, 200)

COLOR_TAB_ACTIVE = (100, 100, 100)
COLOR_TAB_INACTIVE = (60, 60, 60)
COLOR_BTN = (100, 100, 100) # Couleur pour le bouton rétractable

# Les couleurs manquantes pour le menu déroulant et les boutons !
COLOR_BTN_MINUS = (200, 50, 50)
COLOR_BTN_PLUS = (50, 200, 50)
COLOR_DROPDOWN = (60, 90, 130)

DICT_COULEURS_ACTIVATION = {
    "Entrée": (80, 200, 120),
    "ReLU": (70, 150, 220),
    "Sigmoid": (160, 100, 220),
    "Tanh": (220, 150, 70),
    "Softmax": (220, 80, 80),
    "Linear": (180, 180, 180)
}