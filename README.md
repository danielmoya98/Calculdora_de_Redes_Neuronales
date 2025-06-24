# Calculadora de Redes Neuronales Avanzada 🧠

## Descripción

Esta aplicación es una herramienta educativa interactiva para aprender y experimentar con redes neuronales. Permite a los usuarios diseñar, visualizar y entrenar redes neuronales artificiales, comprendiendo cada paso del proceso desde la propagación hacia adelante hasta la retropropagación.

## Características Principales

### 1. Arquitectura de Red Neural
- Creación personalizada de redes con múltiples capas
- Configuración flexible de neuronas por capa
- Soporte para diferentes funciones de activación (ReLU, Sigmoide, Tanh, Softmax)

### 2. Visualización Interactiva
- Representación gráfica de la estructura de la red
- Visualización de pesos y conexiones
- Coloración dinámica de nodos por capa

### 3. Cálculos Detallados
- Implementación paso a paso de propagación hacia adelante
- Visualización de cálculos intermedios (Z, A)
- Retropropagación completa con cálculo de gradientes

### 4. Food-101 Dataset 🍎
- Integración con el conjunto de datos Food-101 para reconocimiento de imágenes
- Preprocesamiento y aumento de datos
- Entrenamiento de modelos CNN para clasificación de alimentos
- Predicción en imágenes nuevas

### 5. Asistente IA 🤖
- Integración con modelos de lenguaje basados en Transformers
- Funcionalidad de resumen de textos
- Sistema de chat para preguntas y respuestas

## Tecnologías Utilizadas

- **Python 3.11** - Lenguaje principal de programación
- **CustomTkinter** - Interfaz gráfica moderna
- **NumPy** - Cálculos numéricos eficientes para redes neuronales
- **Matplotlib** - Visualización de redes y resultados
- **TensorFlow** - Framework para modelos de deep learning
- **Transformers** - Modelos avanzados de procesamiento de lenguaje natural

## Estructura del Proyecto

```
├── app/
│   ├── __init__.py
│   └── app_gui.py       # Interfaz gráfica principal
├── neural_network/
│   ├── __init__.py
│   └── nn_calculator.py # Implementación de red neuronal
├── main.py              # Punto de entrada de la aplicación
└── requirements_ia.txt  # Dependencias para modelos de IA
```

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/calculadora-redes-neuronales.git
   cd calculadora-redes-neuronales
   ```

2. Crear y activar un entorno conda:
   ```bash
   conda create -n redes_neuronales python=3.11
   conda activate redes_neuronales
   ```

3. Instalar dependencias requeridas:
   ```bash
   pip install jinja2 keras matplotlib networkx numpy pandas pillow protobuf pyparsing pytz pyyaml requests scipy six sympy tensorflow tornado werkzeug wheel wrapt customtkinter
   ```

4. Para usar el módulo de asistente IA, instalar dependencias adicionales:
   ```bash
   pip install -r requirements_ia.txt
   ```

## Uso

1. Ejecutar la aplicación:
   ```bash
   python main.py
   ```

2. La aplicación se abrirá con varias pestañas para diferentes funcionalidades:
   - **Arquitectura**: Definir capas y neuronas
   - **Pesos y Biases**: Configurar parámetros
   - **Prop. Adelante**: Probar propagación hacia adelante
   - **Retropropagación**: Experimentar con el entrenamiento
   - **Visualización**: Ver representación gráfica de la red
   - **Food-101**: Trabajar con reconocimiento de imágenes
   - **IA Assistant**: Usar modelos de lenguaje para resúmenes y chat

## Características de Aprendizaje

- **Modo Paso a Paso**: Visualiza cada etapa del proceso de red neuronal
- **Experimentación**: Prueba diferentes arquitecturas y parámetros
- **Resultados Intermedios**: Examina valores Z y A en cada capa
- **Gradientes Visuales**: Comprende el flujo de gradientes en retropropagación

## Requisitos del Sistema

- Python 3.11 o superior
- 8GB RAM mínimo (recomendado 16GB para modelos grandes)
- GPU opcional para entrenamiento acelerado
- Sistema operativo: Windows, macOS o Linux

## Contribuir

Las contribuciones son bienvenidas! Si deseas mejorar esta herramienta educativa:

1. Haz fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`)
3. Haz commit de tus cambios (`git commit -am 'Añadir nueva característica'`)
4. Sube los cambios (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Contacto

Si tienes preguntas o sugerencias, no dudes en abrir un issue en el repositorio o contactar al equipo de desarrollo.
