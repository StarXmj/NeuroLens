### 2. La version Française 🇫🇷
**Nom du fichier : `README-fr.md`**

```markdown
[🇬🇧 English](README.md) | [🇫🇷 Français](README-fr.md) | [🇪🇸 Español](README-es.md)
> **⚠️ Avertissement de version Alpha :** NeuroLens est actuellement en phase Alpha. Le logiciel est toujours en cours de développement et n'est pas terminé. Vous pouvez rencontrer des bugs, des fonctionnalités incomplètes ou quelques incohérences mathématiques.
# 🧠 NeuroLens : Visualisateur Interactif de Réseaux de Neurones

Bienvenue dans **NeuroLens**, un outil pédagogique et interactif conçu pour démystifier le fonctionnement des réseaux de neurones. Construit avec Python, Pygame-CE et PyTorch, NeuroLens fait le pont entre la théorie mathématique abstraite et la pratique visuelle.

Que vous soyez un étudiant découvrant l'IA, un professionnel curieux, ou un chercheur souhaitant illustrer ses concepts, NeuroLens vous permet de "voir" l'apprentissage profond en action.

## ✨ Fonctionnalités Clés

*   **Architecture Modulaire Dynamique :** Construisez votre réseau par simple glisser-déposer (Couches d'Entrée, Cachées, Sortie).
*   **Ajustement en Temps Réel :** Modifiez le nombre de neurones, les fonctions d'activation (ReLU, Sigmoid, Tanh) et les hyperparamètres (Taux d'apprentissage, Epochs, Loss).
*   **Deux Modes de Simulation :**
    *   *Mode Animation :* Une visualisation pédagogique des flux de données (Forward/Backward pass) pour comprendre la mécanique visuellement.
    *   *Mode Expérience PyTorch :* Un véritable entraînement en arrière-plan propulsé par `torch`. L'historique complet est enregistré, vous permettant de naviguer (Play, Pause, Avance Rapide) à travers les époques réelles !
*   **Jeu de Données Intégré :** Saisissez vos propres données de régression ($X$ et $Y$) directement dans l'interface ; le tableau s'adapte automatiquement à votre couche d'entrée.
*   **Inspection Profonde :** Cliquez sur n'importe quel neurone ou lien (poids) pour obtenir un "Focus" détaillé sur sa valeur exacte à une époque donnée.

## 🚀 Installation Rapide

Ce projet utilise `uv` pour une gestion ultrarapide des dépendances Python.

1. Clonez le dépôt :
   ```bash
   git clone [https://github.com/StarXmj/NeuroLens.git](https://github.com/StarXmj/NeuroLens.git)
   cd NeuroLens
2. Installez les dépendances instantanément avec `uv` :
   ```bash
   uv pip install torch pygame-ce matplotlib --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)