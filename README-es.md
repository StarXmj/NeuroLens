### 3. La version Espagnole 🇪🇸
**Nom du fichier : `README-es.md`**

```markdown
[🇬🇧 English](README.md) | [🇫🇷 Français](README-fr.md) | [🇪🇸 Español](README-es.md)

# 🧠 NeuroLens: Visualizador Interactivo de Redes Neuronales

Bienvenido a **NeuroLens**, una herramienta educativa e interactiva diseñada para desmitificar el funcionamiento de las redes neuronales. Construido con Python, Pygame-CE y PyTorch, NeuroLens cierra la brecha entre la teoría matemática abstracta y la práctica visual.

Ya seas un estudiante descubriendo la IA, un profesional curioso o un investigador que desea ilustrar conceptos, NeuroLens te permite "ver" el aprendizaje profundo en acción.

## ✨ Características Principales

*   **Arquitectura Modular Dinámica:** Construye tu red mediante una sencilla interfaz de arrastrar y soltar (Capas de Entrada, Ocultas y de Salida).
*   **Ajuste en Tiempo Real:** Modifica el número de neuronas, funciones de activación (ReLU, Sigmoid, Tanh) e hiperparámetros (Tasa de aprendizaje, Épocas, Pérdida) sobre la marcha.
*   **Dos Modos de Simulación:**
    *   *Modo Animación:* Una visualización pedagógica de los flujos de datos (Forward/Backward pass) para entender la mecánica visualmente.
    *   *Modo Experiencia PyTorch:* Un entrenamiento real en segundo plano impulsado por `torch`. ¡El historial completo queda registrado, permitiéndote navegar (Reproducir, Pausar, Avance Rápido) a través de las épocas reales!
*   **Conjunto de Datos Integrado:** Introduce tus propios datos de regresión ($X$ e $Y$) directamente en la interfaz; la cuadrícula se adapta automáticamente a tu capa de entrada.
*   **Inspección Profunda:** Haz clic en cualquier neurona o enlace (peso) para obtener una vista detallada ("Focus") de su valor exacto en una época determinada.

## 🚀 Inicio Rápido

Este proyecto utiliza `uv` para una gestión ultrarrápida de las dependencias de Python.

1. Clona el repositorio:
   ```bash
   git clone [https://github.com/StarXmj/NeuroLens.git](https://github.com/StarXmj/NeuroLens.git)
   cd NeuroLens