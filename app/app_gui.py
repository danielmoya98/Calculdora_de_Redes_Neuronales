import sys
import threading

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import subprocess


from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import threading
import queue

# Importar la calculadora de red neuronal
from neural_network.nn_calculator import NeuralNetworkCalculator

# Importaciones para visualización (si las usas)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from matplotlib.figure import Figure
from PIL import Image, ImageTk, ImageDraw
import tensorflow as tf  # AGREGAR ESTA LÍNEA


# Configuración de CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Calculadora de Red Neuronal Avanzada")
        self.geometry("1200x800")

        self.nn_calculator = NeuralNetworkCalculator()
        self.current_grads_W: Optional[List[np.ndarray]] = None
        self.current_grads_b: Optional[List[np.ndarray]] = None

        # Inicializar modelos de IA
        self.summarizer = None
        self.qa_pipeline = None
        self.conversation_history = []
        self.is_loading_models = False

        # ✅ VARIABLES PARA FOOD-101 - AGREGAR ESTAS LÍNEAS
        # Variables principales del dataset
        self.food101_X_train = None
        self.food101_X_test = None
        self.food101_y_train = None
        self.food101_y_test = None
        self.food101_class_names = None

        # ✅ VARIABLES PARA IMÁGENES ORIGINALES (VISUALIZACIÓN)
        self.food101_X_train_original = None
        self.food101_X_test_original = None

        # Variables para modelo y augmentation
        self.food101_model = None
        self.food101_datagen = None
        self.food101_test_image = None

        # --- Contenedor principal para organización ---
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(pady=10, padx=10, fill="both", expand=True)

        # --- Botones de acción globales ---
        self.global_actions_frame = ctk.CTkFrame(self.main_container)
        self.global_actions_frame.pack(pady=5, padx=5, fill="x")

        self.load_model_button = ctk.CTkButton(self.global_actions_frame, text="Cargar Modelo", command=self.load_model)
        self.load_model_button.pack(side="left", padx=5)
        self.save_model_button = ctk.CTkButton(self.global_actions_frame, text="Guardar Modelo",
                                               command=self.save_model)
        self.save_model_button.pack(side="left", padx=5)
        self.reset_button = ctk.CTkButton(self.global_actions_frame, text="Resetear Todo", command=self.reset_all)
        self.reset_button.pack(side="left", padx=5)

        # TabView principal
        self.tab_view = ctk.CTkTabview(self.main_container)
        self.tab_view.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab_architecture = self.tab_view.add("1. Arquitectura")
        self.tab_parameters = self.tab_view.add("2. Pesos y Biases")
        self.tab_forward_prop = self.tab_view.add("3. Prop. Adelante")
        self.tab_back_prop = self.tab_view.add("4. Retropropagación")
        self.tab_visualization = self.tab_view.add("5. Visualización")
        self.tab_food101 = self.tab_view.add("6. 🍎 Food-101")  # NUEVA PESTAÑA AQUÍ
        self.tab_ia_assistant = self.tab_view.add("IA Assistant")  # Nuevo tab

        # --- Inicializar contenido de las pestañas ---
        self.layer_frames: List[ctk.CTkFrame] = []  # Para entradas de neuronas y activaciones
        self.weight_textboxes: List[ctk.CTkTextbox] = []
        self.bias_textboxes: List[ctk.CTkTextbox] = []

        self.setup_architecture_tab(self.tab_architecture)
        self.setup_parameters_tab(self.tab_parameters)  # Renombrado de setup_weights_biases_tab
        self.setup_forward_prop_tab(self.tab_forward_prop)
        self.setup_back_prop_tab(self.tab_back_prop)
        self.setup_visualization_tab(self.tab_visualization)
        self.setup_food101_tab()  # NUEVO
        self.setup_ia_assistant_tab()  # Configurar el nuevo tab

        # Inicializar con una capa de entrada y una de salida por defecto
        self.update_layer_frames(initial_layers=2)
        self.update_weights_biases_frames_visibility()

    def setup_architecture_tab(self, tab):
        label = ctk.CTkLabel(tab, text="Definir Arquitectura de la Red", font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(pady=10)

        self.arch_buttons_frame = ctk.CTkFrame(tab)
        self.arch_buttons_frame.pack(pady=5)
        self.add_layer_button = ctk.CTkButton(self.arch_buttons_frame, text="Añadir Capa Oculta",
                                              command=self.add_layer)
        self.add_layer_button.pack(side="left", padx=5)
        self.remove_layer_button = ctk.CTkButton(self.arch_buttons_frame, text="Quitar Última Capa Oculta",
                                                 command=self.remove_layer)
        self.remove_layer_button.pack(side="left", padx=5)

        self.layers_config_scrollable_frame = ctk.CTkScrollableFrame(tab, label_text="Configuración de Capas")
        self.layers_config_scrollable_frame.pack(pady=10, padx=10, fill="both", expand=True)
        # Los frames de capas individuales (self.layer_frames) se añadirán aquí por update_layer_frames

        self.save_arch_button = ctk.CTkButton(tab, text="Guardar y Aplicar Arquitectura",
                                              command=self.save_architecture)
        self.save_arch_button.pack(pady=10)

    def setup_parameters_tab(self, tab):
        label = ctk.CTkLabel(tab, text="Establecer/Ver Pesos y Biases", font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(pady=10)

        self.params_buttons_frame = ctk.CTkFrame(tab)
        self.params_buttons_frame.pack(pady=5)
        self.init_random_params_button = ctk.CTkButton(self.params_buttons_frame, text="Inicializar Pesos Aleatorios",
                                                       command=self.initialize_random_parameters)
        self.init_random_params_button.pack(side="left", padx=5)
        self.save_params_button = ctk.CTkButton(self.params_buttons_frame, text="Guardar Parámetros Ingresados",
                                                command=self.save_parameters)
        self.save_params_button.pack(side="left", padx=5)

        self.weights_biases_scrollable_frame = ctk.CTkScrollableFrame(tab, label_text="Parámetros por Capa")
        self.weights_biases_scrollable_frame.pack(pady=10, padx=10, fill="both", expand=True)
        # Los textboxes de W y b se añadirán/actualizarán aquí por update_weights_biases_frames_visibility

    def setup_forward_prop_tab(self, tab):
        label = ctk.CTkLabel(tab, text="Propagación Hacia Adelante", font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(pady=10)

        ctk.CTkLabel(tab, text="Entrada X (vector columna, ej: 1;2;3 o matriz [[1,2],[3,4]];[[5,6],[7,8]]):").pack(
            pady=(5, 0))
        self.fp_input_X_textbox = ctk.CTkTextbox(tab, height=80)
        self.fp_input_X_textbox.pack(pady=5, padx=10, fill="x")
        self.fp_input_X_textbox.insert("1.0", "1;2;3")  # Ejemplo

        self.fp_run_button = ctk.CTkButton(tab, text="Ejecutar Propagación Adelante",
                                           command=self.run_forward_propagation_from_tab)
        self.fp_run_button.pack(pady=10)

        self.show_intermediate_var_fp = ctk.CTkCheckBox(tab, text="Mostrar Resultados Intermedios (Z, A)")
        self.show_intermediate_var_fp.pack(pady=5)
        self.show_intermediate_var_fp.select()  # Seleccionado por defecto

        ctk.CTkLabel(tab, text="Resultados:").pack(pady=(5, 0))
        self.fp_results_textbox = ctk.CTkTextbox(tab, height=250)
        self.fp_results_textbox.pack(pady=5, padx=10, fill="both", expand=True)

    def setup_back_prop_tab(self, tab):
        label = ctk.CTkLabel(tab, text="Retropropagación (Entrenamiento)", font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(pady=10)

        ctk.CTkLabel(tab, text="Entrada X (para forward pass):").pack(pady=(5, 0))
        self.bp_input_X_textbox = ctk.CTkTextbox(tab, height=60)
        self.bp_input_X_textbox.pack(pady=5, padx=10, fill="x")
        self.bp_input_X_textbox.insert("1.0", "1;2;3")  # Ejemplo

        ctk.CTkLabel(tab, text="Salida Esperada Y_true (mismas dimensiones que salida de red):").pack(pady=(5, 0))
        self.bp_input_Y_textbox = ctk.CTkTextbox(tab, height=60)
        self.bp_input_Y_textbox.pack(pady=5, padx=10, fill="x")
        self.bp_input_Y_textbox.insert("1.0", "0;1")  # Ejemplo

        ctk.CTkLabel(tab, text="Tasa de Aprendizaje (Learning Rate):").pack(pady=(5, 0))
        self.bp_lr_entry = ctk.CTkEntry(tab, placeholder_text="0.01")
        self.bp_lr_entry.pack(pady=5, padx=10, fill="x")
        self.bp_lr_entry.insert(0, "0.01")

        self.bp_run_button = ctk.CTkButton(tab, text="Calcular Gradientes (Backprop)",
                                           command=self.run_backward_propagation_from_tab)
        self.bp_run_button.pack(pady=10)

        self.bp_update_params_button = ctk.CTkButton(tab, text="Actualizar Pesos y Biases",
                                                     command=self.update_nn_parameters_from_tab, state="disabled")
        self.bp_update_params_button.pack(pady=5)

        self.bp_cost_label = ctk.CTkLabel(tab, text="Costo: N/A")
        self.bp_cost_label.pack(pady=5)

        ctk.CTkLabel(tab, text="Gradientes Calculados (dW, db):").pack(pady=(5, 0))
        self.bp_grads_textbox = ctk.CTkTextbox(tab, height=200)
        self.bp_grads_textbox.pack(pady=5, padx=10, fill="both", expand=True)

    def setup_visualization_tab(self, tab):
        label = ctk.CTkLabel(tab, text="Visualización de la Red", font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(pady=10)

        self.vis_options_frame = ctk.CTkFrame(tab)
        self.vis_options_frame.pack(pady=5, fill="x")
        self.show_weights_var = ctk.CTkCheckBox(self.vis_options_frame, text="Mostrar Pesos en Aristas")
        self.show_weights_var.pack(side="left", padx=10)
        self.colored_nodes_var = ctk.CTkCheckBox(self.vis_options_frame, text="Nodos Coloreados por Capa")
        self.colored_nodes_var.pack(side="left", padx=10)
        self.colored_nodes_var.select()

        self.visualize_button = ctk.CTkButton(tab, text="Generar/Actualizar Visualización",
                                              command=self.visualize_network)
        self.visualize_button.pack(pady=10)

        self.fig_frame = ctk.CTkFrame(tab)
        self.fig_frame.pack(pady=10, padx=10, fill="both", expand=True)
        # El canvas de Matplotlib se añadirá aquí
        self.fig: Optional[Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None

    def setup_food101_tab(self):
        """Configura la pestaña de Food-101"""
        # Frame principal con scroll
        main_frame = ctk.CTkScrollableFrame(self.tab_food101)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ❌ BORRAR ESTA LÍNEA (causa error):
        # food_frame = self.tabview.tab("Food-101")

        # Título principal
        title_label = ctk.CTkLabel(main_frame, text="🍎 Food-101 Reconocimiento de Comida",
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10)

        # Frame para organizar en 2 columnas
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Columna izquierda - Controles
        left_column = ctk.CTkFrame(content_frame)
        left_column.pack(side="left", fill="y", padx=(0, 10), pady=10)

        # SECCIÓN 1: Configuración del dataset
        dataset_frame = ctk.CTkFrame(left_column)
        dataset_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(dataset_frame, text="📁 Configuración Dataset",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        # Selector de número de clases
        ctk.CTkLabel(dataset_frame, text="Número de clases de comida:").pack()
        self.food_classes_var = ctk.StringVar(value="10")
        classes_options = ["5", "10", "20", "50"]
        self.food_classes_combo = ctk.CTkComboBox(dataset_frame, values=classes_options,
                                                  variable=self.food_classes_var, width=200)
        self.food_classes_combo.pack(pady=5)

        # Selector de tamaño de dataset
        ctk.CTkLabel(dataset_frame, text="Tamaño del dataset:").pack()
        self.food_size_var = ctk.StringVar(value="pequeño")
        size_options = ["pequeño (50 imgs/clase)", "mediano (100 imgs/clase)", "grande (200 imgs/clase)"]
        self.food_size_combo = ctk.CTkComboBox(dataset_frame, values=size_options,
                                               variable=self.food_size_var, width=200)
        self.food_size_combo.pack(pady=5)

        self.download_food101_button = ctk.CTkButton(
            dataset_frame, text="Descargar y Preparar Food-101",
            command=self.download_food101, width=250
        )
        self.download_food101_button.pack(pady=5)

        self.food101_status_label = ctk.CTkLabel(dataset_frame, text="Dataset no descargado")
        self.food101_status_label.pack(pady=5)

        # SECCIÓN 2: Preprocesamiento
        preprocess_frame = ctk.CTkFrame(left_column)
        preprocess_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(preprocess_frame, text="⚙️ Preprocesamiento",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        self.food_preprocess_button = ctk.CTkButton(
            preprocess_frame, text="Aplicar Preprocesamiento",
            command=self.apply_food_preprocessing, width=250
        )
        self.food_preprocess_button.pack(pady=5)

        self.show_food_samples_button = ctk.CTkButton(
            preprocess_frame, text="Mostrar Muestras de Comida",
            command=self.show_food_samples, width=250
        )
        self.show_food_samples_button.pack(pady=5)

        # SECCIÓN 3: Data Augmentation
        augment_frame = ctk.CTkFrame(left_column)
        augment_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(augment_frame, text="🔄 Data Augmentation",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        self.setup_food_augmentation_button = ctk.CTkButton(
            augment_frame, text="Configurar Augmentation",
            command=self.setup_food_augmentation, width=250
        )
        self.setup_food_augmentation_button.pack(pady=5)

        self.show_food_augmented_button = ctk.CTkButton(
            augment_frame, text="Mostrar Antes/Después",
            command=self.show_food_augmented, width=250
        )
        self.show_food_augmented_button.pack(pady=5)

        # SECCIÓN 4: Modelo CNN
        model_frame = ctk.CTkFrame(left_column)
        model_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(model_frame, text="🧠 Modelo CNN Food-101",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        self.create_food_cnn_button = ctk.CTkButton(
            model_frame, text="Crear Modelo CNN",
            command=self.create_food101_cnn_model, width=250
        )
        self.create_food_cnn_button.pack(pady=5)

        self.show_food_model_button = ctk.CTkButton(
            model_frame, text="Mostrar Arquitectura",
            command=self.show_food_model_architecture, width=250
        )
        self.show_food_model_button.pack(pady=5)

        # Parámetros de entrenamiento
        train_params_frame = ctk.CTkFrame(model_frame)
        train_params_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(train_params_frame, text="Épocas:").pack()
        self.food_epochs_entry = ctk.CTkEntry(train_params_frame, width=100)
        self.food_epochs_entry.insert(0, "10")
        self.food_epochs_entry.pack(pady=2)

        ctk.CTkLabel(train_params_frame, text="Batch Size:").pack()
        self.food_batch_size_entry = ctk.CTkEntry(train_params_frame, width=100)
        self.food_batch_size_entry.insert(0, "16")
        self.food_batch_size_entry.pack(pady=2)

        self.train_food_cnn_button = ctk.CTkButton(
            model_frame, text="Entrenar Modelo",
            command=self.train_food_cnn_model, width=250
        )
        self.train_food_cnn_button.pack(pady=5)

        # Barra de progreso
        self.food_training_progress = ctk.CTkProgressBar(model_frame, width=250)
        self.food_training_progress.pack(pady=5)
        self.food_training_progress.set(0)

        self.save_food_cnn_button = ctk.CTkButton(
            model_frame, text="Guardar Modelo Food-101",
            command=self.save_food_cnn_model, width=250
        )
        self.save_food_cnn_button.pack(pady=5)

        # SECCIÓN 5: Predicción y Google Colab
        prediction_frame = ctk.CTkFrame(left_column)
        prediction_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(prediction_frame, text="🎯 Predicción y Google Colab",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        self.load_food_test_button = ctk.CTkButton(
            prediction_frame, text="Cargar Imagen de Comida",
            command=self.load_food_test_image, width=250
        )
        self.load_food_test_button.pack(pady=5)

        self.predict_food_button = ctk.CTkButton(
            prediction_frame, text="Predecir Comida",
            command=self.predict_food, width=250
        )
        self.predict_food_button.pack(pady=5)

        # Label para mostrar predicción
        self.food_prediction_label = ctk.CTkLabel(
            prediction_frame, text="Predicción: -",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.food_prediction_label.pack(pady=5)

        # ✅ BOTÓN GOOGLE COLAB CORREGIDO - USANDO prediction_frame
        colab_button = ctk.CTkButton(
            prediction_frame,
            text="🚀 Abrir en Google Colab",
            command=self.open_google_colab,
            width=250,
            fg_color="#F9AB00",
            hover_color="#E8A100"
        )
        colab_button.pack(pady=10)

        # ❌ BORRAR ESTA LÍNEA DUPLICADA:
        # self.food_prediction_label.pack(pady=5)

        # Columna derecha - Visualizaciones
        right_column = ctk.CTkFrame(content_frame)
        right_column.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=10)

        # Frame para mostrar visualizaciones
        self.food_visualization_frame = ctk.CTkFrame(right_column)
        self.food_visualization_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.food_visualization_label = ctk.CTkLabel(
            self.food_visualization_frame, text="Las visualizaciones de Food-101 aparecerán aquí",
            font=ctk.CTkFont(size=14)
        )
        self.food_visualization_label.pack(expand=True)

        # Variables para almacenar datos de Food-101
        self.food101_X_train = None
        self.food101_y_train = None
        self.food101_X_test = None
        self.food101_y_test = None
        self.food101_class_names = []
        self.food101_datagen = None
        self.food101_model = None
        self.food_test_image = None

    def add_layer(self):
        current_layers = len(self.layer_frames)
        if current_layers == 0:  # Caso inicial, añadir entrada y salida
            self.update_layer_frames(initial_layers=2)
        elif current_layers < 10:  # Limitar a 10 capas por simplicidad
            self.update_layer_frames(num_layers=current_layers + 1)
        else:
            messagebox.showwarning("Límite Alcanzado", "No se pueden añadir más de 10 capas.")

    def remove_layer(self):
        current_layers = len(self.layer_frames)
        if current_layers > 2:  # Siempre mantener al menos capa de entrada y salida
            self.update_layer_frames(num_layers=current_layers - 1)
        else:
            messagebox.showwarning("Mínimo de Capas", "Se requieren al menos una capa de entrada y una de salida.")

    def update_layer_frames(self, num_layers=None, initial_layers=None):
        if initial_layers is not None:
            current_num_visible_layers = initial_layers
        elif num_layers is not None:
            current_num_visible_layers = num_layers
        else:  # Si es llamado sin args, intenta usar el número actual de frames
            current_num_visible_layers = len(self.layer_frames) if self.layer_frames else 2

        # Limpiar frames existentes
        for frame in self.layer_frames:
            frame.destroy()
        self.layer_frames = []

        activation_options = ["Sigmoide", "ReLU", "Tanh", "Lineal", "Softmax"]

        for i in range(current_num_visible_layers):
            layer_frame = ctk.CTkFrame(self.layers_config_scrollable_frame)
            layer_frame.pack(pady=5, padx=5, fill="x")
            self.layer_frames.append(layer_frame)

            label_text = f"Capa {i} (Entrada)" if i == 0 else f"Capa {i} ({'Salida' if i == current_num_visible_layers - 1 else 'Oculta'})"
            ctk.CTkLabel(layer_frame, text=label_text).pack(side="left", padx=5)

            ctk.CTkLabel(layer_frame, text="Neuronas:").pack(side="left", padx=5)
            neurons_entry = ctk.CTkEntry(layer_frame, width=50)
            neurons_entry.pack(side="left", padx=5)
            neurons_entry.insert(0, "3" if i == 0 else (
                "2" if i == current_num_visible_layers - 1 else "4"))  # Valores por defecto
            setattr(layer_frame, "neurons_entry", neurons_entry)

            if i > 0:  # La capa de entrada no tiene activación previa modificable desde aquí
                ctk.CTkLabel(layer_frame, text="Activación:").pack(side="left", padx=5)
                activation_menu = ctk.CTkOptionMenu(layer_frame, values=activation_options)
                default_activation = "Softmax" if i == current_num_visible_layers - 1 else "ReLU"
                activation_menu.set(default_activation)
                activation_menu.pack(side="left", padx=5)
                setattr(layer_frame, "activation_menu", activation_menu)
            else:  # Para capa de entrada, solo almacenamos la "activación" como "Lineal" (o placeholder)
                setattr(layer_frame, "activation_menu", "Lineal")  # No es un widget, solo un valor

        self.update_weights_biases_frames_visibility()

    def update_weights_biases_frames_visibility(self):
        # Limpiar textboxes antiguos
        for tb in self.weight_textboxes + self.bias_textboxes:
            if tb.winfo_exists(): tb.master.destroy()  # Destruir el frame contenedor
        self.weight_textboxes = []
        self.bias_textboxes = []

        if not self.nn_calculator.layers_neurons or len(self.nn_calculator.layers_neurons) < 2:
            ctk.CTkLabel(self.weights_biases_scrollable_frame,
                         text="Define la arquitectura primero (mínimo 2 capas).").pack()
            return

        # Quitar cualquier mensaje previo
        for widget in self.weights_biases_scrollable_frame.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and "Define la arquitectura" in widget.cget("text"):
                widget.destroy()

        num_param_sets = len(self.nn_calculator.layers_neurons) - 1
        for i in range(num_param_sets):
            n_out = self.nn_calculator.layers_neurons[i + 1]
            n_in = self.nn_calculator.layers_neurons[i]

            # Frame para cada conjunto de W y b
            param_frame = ctk.CTkFrame(self.weights_biases_scrollable_frame)
            param_frame.pack(pady=10, padx=5, fill="x")

            # Weights
            ctk.CTkLabel(param_frame, text=f"Pesos W{i + 1} (Capa {i} a {i + 1}) [{n_out}x{n_in}]:").pack(anchor="w")
            w_textbox = ctk.CTkTextbox(param_frame, height=max(60, n_out * 20))
            w_textbox.pack(fill="x", expand=True, padx=5, pady=2)
            if self.nn_calculator.weights and i < len(self.nn_calculator.weights) and self.nn_calculator.weights[
                i] is not None:
                w_textbox.insert("1.0", self.format_matrix_for_display(self.nn_calculator.weights[i]))
            else:
                w_textbox.insert("1.0", f"# Ingrese {n_out}x{n_in} pesos, ej: [[1,2,3],[4,5,6]]")
            self.weight_textboxes.append(w_textbox)

            # Biases
            ctk.CTkLabel(param_frame, text=f"Biases b{i + 1} (Capa {i + 1}) [{n_out}x1]:").pack(anchor="w", pady=(5, 0))
            b_textbox = ctk.CTkTextbox(param_frame, height=max(40, n_out * 20))
            b_textbox.pack(fill="x", expand=True, padx=5, pady=2)
            if self.nn_calculator.biases and i < len(self.nn_calculator.biases) and self.nn_calculator.biases[
                i] is not None:
                b_textbox.insert("1.0", self.format_matrix_for_display(self.nn_calculator.biases[i]))
            else:
                b_textbox.insert("1.0", f"# Ingrese {n_out}x1 biases, ej: [[0.1],[0.2]]")
            self.bias_textboxes.append(b_textbox)

    def format_matrix_for_display(self, matrix: np.ndarray) -> str:
        if matrix is None: return ""
        return str(matrix.tolist()).replace("], [", "],\n [")

    def parse_input_matrix_from_textbox(self, textbox_content: str) -> Optional[np.ndarray]:
        content = textbox_content.strip()
        if not content or content.startswith("#"): return None
        try:
            # Permitir múltiples formatos: simple vector columna, o lista de listas
            if content.startswith("[[") and content.endswith("]]"):  # Matriz completa
                parsed_list = json.loads(content)
                return np.array(parsed_list, dtype=float)
            elif ";" in content and not content.startswith("["):  # Vector columna 1;2;3
                return np.array([list(map(float, content.split(';')))], dtype=float).T
            elif "," in content and not content.startswith("["):  # Vector fila 1,2,3 (convertir a columna)
                return np.array([list(map(float, content.split(',')))], dtype=float).T
            else:  # Suponer un solo número para vector de 1x1
                return np.array([[float(content)]], dtype=float)
        except Exception as e:
            messagebox.showerror("Error de Formato de Matriz",
                                 f"Error al parsear: '{content[:30]}...'\n{e}\nUse formato JSON de lista de listas (ej: [[1,2],[3,4]]) o vector columna (ej: 1;2;3).")
            return None

    def save_architecture(self):
        try:
            layers_neurons = []
            activations = []  # activations[0] para capa entrada, activations[i] para capa i (post Z_i)

            # Capa de entrada (Capa 0)
            input_neurons_str = self.layer_frames[0].neurons_entry.get()
            layers_neurons.append(int(input_neurons_str))
            activations.append("Lineal")  # Placeholder para capa de entrada

            # Capas ocultas y de salida
            for i in range(1, len(self.layer_frames)):
                frame = self.layer_frames[i]
                neurons_str = frame.neurons_entry.get()
                layers_neurons.append(int(neurons_str))
                activations.append(frame.activation_menu.get())

            self.nn_calculator.set_architecture(layers_neurons, activations)
            messagebox.showinfo("Arquitectura Guardada",
                                "Arquitectura actualizada. Configure los pesos y biases si es necesario.")
            self.update_weights_biases_frames_visibility()  # Actualizar la pestaña de parámetros
            # Resetear gradientes si la arquitectura cambia
            self.current_grads_W = None
            self.current_grads_b = None
            self.bp_update_params_button.configure(state="disabled")
            self.bp_grads_textbox.delete("1.0", "end")
            self.bp_cost_label.configure(text="Costo: N/A")

        except ValueError as e:
            messagebox.showerror("Error en Arquitectura", str(e))
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}")

    def save_parameters(self):
        if not self.nn_calculator.layers_neurons or len(self.nn_calculator.layers_neurons) < 2:
            messagebox.showerror("Error", "Primero defina y guarde una arquitectura válida.")
            return
        try:
            parsed_weights = []
            parsed_biases = []
            num_param_sets = len(self.nn_calculator.layers_neurons) - 1

            if len(self.weight_textboxes) != num_param_sets or len(self.bias_textboxes) != num_param_sets:
                messagebox.showerror("Error de UI",
                                     "Discrepancia en número de textboxes de parámetros. Intente guardar arquitectura de nuevo.")
                return

            for i in range(num_param_sets):
                w_text = self.weight_textboxes[i].get("1.0", "end-1c")
                b_text = self.bias_textboxes[i].get("1.0", "end-1c")

                W = self.parse_input_matrix_from_textbox(w_text)
                b = self.parse_input_matrix_from_textbox(b_text)

                if W is None or b is None:
                    messagebox.showerror("Error de Formato",
                                         f"Error en formato de W{i + 1} o b{i + 1}. Verifique las entradas.")
                    return

                # Validar dimensiones esperadas
                expected_w_shape = (self.nn_calculator.layers_neurons[i + 1], self.nn_calculator.layers_neurons[i])
                expected_b_shape = (self.nn_calculator.layers_neurons[i + 1], 1)

                if W.shape != expected_w_shape:
                    messagebox.showerror("Error de Dimensiones",
                                         f"W{i + 1} tiene forma {W.shape}, se esperaba {expected_w_shape}")
                    return
                if b.shape != expected_b_shape:
                    # Intentar corregir si es un vector fila como [1,2,3] en lugar de [[1],[2],[3]]
                    if b.ndim == 2 and b.shape[0] == 1 and b.shape[1] == expected_b_shape[0]:
                        b = b.T
                    elif b.ndim == 1 and b.shape[0] == expected_b_shape[0]:
                        b = b.reshape(expected_b_shape)

                    if b.shape != expected_b_shape:
                        messagebox.showerror("Error de Dimensiones",
                                             f"b{i + 1} tiene forma {b.shape}, se esperaba {expected_b_shape} (ej: [[v1],[v2]])")
                        return

                parsed_weights.append(W)
                parsed_biases.append(b)

            self.nn_calculator.set_parameters(parsed_weights, parsed_biases)
            messagebox.showinfo("Parámetros Guardados", "Pesos y biases actualizados.")
        except Exception as e:
            messagebox.showerror("Error Guardando Parámetros", str(e))

    def initialize_random_parameters(self):
        if not self.nn_calculator.layers_neurons or len(self.nn_calculator.layers_neurons) < 2:
            messagebox.showerror("Error", "Defina y guarde una arquitectura primero.")
            return

        try:
            weights = []
            biases = []
            for i in range(len(self.nn_calculator.layers_neurons) - 1):
                n_out = self.nn_calculator.layers_neurons[i + 1]
                n_in = self.nn_calculator.layers_neurons[i]
                # Xavier/Glorot initialization (good for tanh, sigmoid) or He (good for ReLU)
                # Simple random normal * 0.01 for now
                W = np.random.randn(n_out, n_in) * 0.01
                b = np.zeros((n_out, 1))
                weights.append(W)
                biases.append(b)

            self.nn_calculator.set_parameters(weights, biases)
            # Actualizar los textboxes en la pestaña de parámetros
            self.update_weights_biases_frames_visibility()
            messagebox.showinfo("Parámetros Inicializados",
                                "Pesos y biases inicializados aleatoriamente (W) y a cero (b).")
        except Exception as e:
            messagebox.showerror("Error Inicializando Parámetros", str(e))

    def run_forward_propagation_from_tab(self):
        if not self.nn_calculator.weights or self.nn_calculator.weights[0] is None:
            messagebox.showerror("Error",
                                 "Parámetros (pesos y biases) no establecidos. Defina arquitectura y luego guarde o inicialice parámetros.")
            return

        x_str = self.fp_input_X_textbox.get("1.0", "end-1c")
        X = self.parse_input_matrix_from_textbox(x_str)
        if X is None: return

        try:
            AL = self.nn_calculator.forward_propagation(X)
            results_text = f"Salida Final AL (Capa {len(self.nn_calculator.layers_neurons) - 1}):\n{self.format_matrix_for_display(AL)}\n"

            if self.show_intermediate_var_fp.get():
                results_text += "\n--- Resultados Intermedios ---\n"
                for key, value in self.nn_calculator.intermediate_results.items():
                    results_text += f"{key}:\n{self.format_matrix_for_display(value)}\n\n"

            self.fp_results_textbox.delete("1.0", "end")
            self.fp_results_textbox.insert("1.0", results_text)
        except Exception as e:
            messagebox.showerror("Error en Propagación Adelante", str(e))
            self.fp_results_textbox.delete("1.0", "end")
            self.fp_results_textbox.insert("1.0", f"Error: {e}")

    def run_backward_propagation_from_tab(self):
        if not self.nn_calculator.weights or self.nn_calculator.weights[0] is None:
            messagebox.showerror("Error", "Parámetros no establecidos. Configurelos primero.")
            return

        x_str = self.bp_input_X_textbox.get("1.0", "end-1c")
        y_true_str = self.bp_input_Y_textbox.get("1.0", "end-1c")

        X = self.parse_input_matrix_from_textbox(x_str)
        Y_true = self.parse_input_matrix_from_textbox(y_true_str)

        if X is None or Y_true is None: return

        try:
            # 1. Forward Propagation
            AL = self.nn_calculator.forward_propagation(X)

            if AL.shape != Y_true.shape:
                # Intentar remodelar Y_true si es un vector fila y AL es columna, o viceversa
                if Y_true.ndim == 2 and AL.ndim == 2 and Y_true.shape[0] == AL.shape[1] and Y_true.shape[1] == AL.shape[
                    0]:
                    Y_true = Y_true.T
                if AL.shape != Y_true.shape:  # Comprobar de nuevo
                    messagebox.showerror("Error de Dimensiones",
                                         f"Y_true ({Y_true.shape}) no coincide con salida AL ({AL.shape}).")
                    return

            # 2. Calcular Costo
            output_activation = self.nn_calculator.activations[-1]  # Activación de la última capa
            cost = self.nn_calculator.calculate_cost(AL, Y_true, output_activation)
            self.bp_cost_label.configure(text=f"Costo ({output_activation}): {cost:.6f}")

            # 3. Backward Propagation
            grads_W, grads_b = self.nn_calculator.backward_propagation(AL, Y_true)
            self.current_grads_W = grads_W
            self.current_grads_b = grads_b

            grads_text = "Gradientes dW:\n"
            for i, dW in enumerate(grads_W):
                grads_text += f"dW{i + 1}:\n{self.format_matrix_for_display(dW)}\n\n"
            grads_text += "Gradientes db:\n"
            for i, db in enumerate(grads_b):
                grads_text += f"db{i + 1}:\n{self.format_matrix_for_display(db)}\n\n"

            self.bp_grads_textbox.delete("1.0", "end")
            self.bp_grads_textbox.insert("1.0", grads_text)
            self.bp_update_params_button.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Error en Retropropagación", str(e))
            self.bp_cost_label.configure(text="Costo: Error")
            self.bp_grads_textbox.delete("1.0", "end")
            self.bp_grads_textbox.insert("1.0", f"Error: {e}")
            self.bp_update_params_button.configure(state="disabled")
            self.current_grads_W = None
            self.current_grads_b = None

    def update_nn_parameters_from_tab(self):
        if self.current_grads_W is None or self.current_grads_b is None:
            messagebox.showwarning("Sin Gradientes", "Calcule los gradientes primero.")
            return

        lr_str = self.bp_lr_entry.get()
        try:
            learning_rate = float(lr_str)
            if learning_rate <= 0:
                raise ValueError("Tasa de aprendizaje debe ser positiva.")
        except ValueError:
            messagebox.showerror("Error", "Tasa de aprendizaje inválida.")
            return

        try:
            self.nn_calculator.update_parameters(self.current_grads_W, self.current_grads_b, learning_rate)
            messagebox.showinfo("Éxito", "Pesos y biases actualizados.")

            # Actualizar visualización de parámetros en la pestaña "Pesos y Biases"
            self.update_weights_biases_frames_visibility()

            self.current_grads_W = None  # Limpiar para evitar re-aplicación
            self.current_grads_b = None
            self.bp_update_params_button.configure(state="disabled")
            self.bp_grads_textbox.delete("1.0", "end")
            # El costo se referirá al estado ANTES de la actualización, así que no lo borramos necesariamente
            # self.bp_cost_label.configure(text="Costo: N/A (Parámetros actualizados)")
        except Exception as e:
            messagebox.showerror("Error Actualizando Parámetros", str(e))

    def save_model(self):
        if not self.nn_calculator.layers_neurons:
            messagebox.showerror("Error", "No hay arquitectura definida para guardar.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Guardar Modelo de Red Neuronal"
        )
        if not filepath: return

        try:
            model_data = self.nn_calculator.to_json()
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=4)
            messagebox.showinfo("Modelo Guardado", f"Modelo guardado en {filepath}")
        except Exception as e:
            messagebox.showerror("Error Guardando Modelo", str(e))

    def load_model(self):
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Cargar Modelo de Red Neuronal"
        )
        if not filepath: return

        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)

            self.nn_calculator = NeuralNetworkCalculator.from_json(model_data)

            # Actualizar la GUI para reflejar el modelo cargado
            # 1. Pestaña de Arquitectura
            num_layers_loaded = len(self.nn_calculator.layers_neurons)
            self.update_layer_frames(num_layers=num_layers_loaded)  # Crea los frames
            # Ahora llenar los frames con los datos cargados
            for i in range(num_layers_loaded):
                self.layer_frames[i].neurons_entry.delete(0, "end")
                self.layer_frames[i].neurons_entry.insert(0, str(self.nn_calculator.layers_neurons[i]))
                if i > 0:  # Activaciones
                    self.layer_frames[i].activation_menu.set(self.nn_calculator.activations[i])

            # 2. Pestaña de Parámetros
            self.update_weights_biases_frames_visibility()  # Esto leerá de nn_calculator y poblará

            messagebox.showinfo("Modelo Cargado", f"Modelo cargado desde {filepath}")
            # Resetear estado de backprop
            self.current_grads_W = None
            self.current_grads_b = None
            self.bp_update_params_button.configure(state="disabled")
            self.bp_grads_textbox.delete("1.0", "end")
            self.bp_cost_label.configure(text="Costo: N/A")

        except Exception as e:
            messagebox.showerror("Error Cargando Modelo", str(e))

    def reset_all(self):
        if messagebox.askyesno("Confirmar Reset", "¿Está seguro de que desea resetear toda la configuración?"):
            self.nn_calculator = NeuralNetworkCalculator()
            self.current_grads_W = None
            self.current_grads_b = None

            # Limpiar entradas y resultados en pestañas
            self.fp_input_X_textbox.delete("1.0", "end")
            self.fp_results_textbox.delete("1.0", "end")
            self.bp_input_X_textbox.delete("1.0", "end")
            self.bp_input_Y_textbox.delete("1.0", "end")
            self.bp_lr_entry.delete(0, "end");
            self.bp_lr_entry.insert(0, "0.01")
            self.bp_grads_textbox.delete("1.0", "end")
            self.bp_cost_label.configure(text="Costo: N/A")
            self.bp_update_params_button.configure(state="disabled")

            # Resetear arquitectura a 2 capas por defecto
            self.update_layer_frames(initial_layers=2)
            # Resetear/limpiar visualización de pesos y biases
            self.update_weights_biases_frames_visibility()

            # Limpiar visualización
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            if self.fig:
                plt.close(self.fig)
                self.fig = None
                self.ax = None

            messagebox.showinfo("Reset Completo", "Toda la configuración ha sido reseteada.")

    def visualize_network(self):
        if not self.nn_calculator.layers_neurons:
            messagebox.showinfo("Visualización", "Defina una arquitectura primero.")
            return

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.fig:
            plt.close(self.fig)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        def sanitize_color_for_matplotlib(color_val):
            """Convierte valores de color de CustomTkinter a formatos válidos para Matplotlib"""
            # Si es None o vacío, usar fallback
            if not color_val:
                return "#2B2B2B"

            # Si es una tupla, extraer el primer elemento válido
            if isinstance(color_val, (tuple, list)):
                for item in color_val:
                    if isinstance(item, str) and item.strip():
                        color_val = item
                        break
                else:
                    return "#2B2B2B"  # Si no hay elementos válidos en la tupla

            # Asegurar que es string
            if not isinstance(color_val, str):
                return "#2B2B2B"

            color_val_lower = color_val.lower().strip()

            # Mapeo de colores especiales de CustomTkinter
            color_mapping = {
                "gray17": "#2B2B2B",
                "gray20": "#333333",
                "gray92": "#EBEBEB",
                "transparent": "#2B2B2B"
            }

            if color_val_lower in color_mapping:
                return color_mapping[color_val_lower]

            # Si ya es un color hexadecimal válido, devolverlo
            if color_val.startswith('#') and len(color_val) in [4, 7]:
                return color_val

            # Para otros colores nombrados, intentar devolverlos (Matplotlib los reconocerá)
            return color_val

        def extract_color_from_theme(theme_path, fallback_color):
            """Extrae color del tema de manera segura"""
            try:
                theme_val = ctk.ThemeManager.theme
                for key in theme_path:
                    theme_val = theme_val[key]

                if isinstance(theme_val, (tuple, list)):
                    if len(theme_val) > 1:
                        return theme_val[1]  # Preferir modo oscuro
                    elif len(theme_val) == 1:
                        return theme_val[0]  # Usar el único disponible
                    else:
                        return fallback_color  # Tupla vacía
                else:
                    return theme_val  # Es un valor directo
            except (KeyError, IndexError, TypeError, AttributeError):
                return fallback_color

        # Extraer colores del tema de manera segura
        raw_frame_bg_color = extract_color_from_theme(["CTkFrame", "fg_color"], "#2B2B2B")
        raw_label_text_color = extract_color_from_theme(["CTkLabel", "text_color"], "#DCE4EE")
        raw_button_text_color = extract_color_from_theme(["CTkButton", "text_color"], "#DCE4EE")

        # Sanitizar colores
        frame_bg_color_mpl = sanitize_color_for_matplotlib(raw_frame_bg_color)
        label_text_color_mpl = sanitize_color_for_matplotlib(raw_label_text_color)
        button_text_color_mpl = sanitize_color_for_matplotlib(raw_button_text_color)

        # Debugging - puedes comentar estas líneas después de que funcione
        print(f"Debug - Frame BG: {raw_frame_bg_color} -> {frame_bg_color_mpl}")
        print(f"Debug - Label Text: {raw_label_text_color} -> {label_text_color_mpl}")
        print(f"Debug - Button Text: {raw_button_text_color} -> {button_text_color_mpl}")

        # Aplicar colores a la figura
        self.fig.patch.set_facecolor(frame_bg_color_mpl)
        self.ax.set_facecolor(frame_bg_color_mpl)

        G = nx.DiGraph()
        layer_nodes = []
        node_display_colors_list = []
        pos = {}
        max_neurons_in_layer = max(self.nn_calculator.layers_neurons) if self.nn_calculator.layers_neurons else 1

        try:
            num_layers_for_palette = len(self.nn_calculator.layers_neurons)
            if num_layers_for_palette > 0:
                color_palette_nodes = plt.cm.get_cmap('viridis', num_layers_for_palette)
            else:
                color_palette_nodes = lambda x: "skyblue"
        except ValueError:
            color_palette_nodes = lambda x: "skyblue"

        for i, num_neurons in enumerate(self.nn_calculator.layers_neurons):
            nodes_in_this_layer = []
            for j in range(num_neurons):
                node_id = f"{i}_{j}"
                nodes_in_this_layer.append(node_id)
                G.add_node(node_id)
                y_pos = (max_neurons_in_layer - num_neurons) / 2 + j
                pos[node_id] = (i * 2, -y_pos * 1.5)
                if self.colored_nodes_var.get() and len(self.nn_calculator.layers_neurons) > 0:
                    try:
                        node_display_colors_list.append(color_palette_nodes(i))
                    except Exception:
                        node_display_colors_list.append("lightgreen")
            layer_nodes.append(nodes_in_this_layer)

        edge_labels = {}
        edge_colors_list = []
        edge_widths_list = []
        default_edge_color_val = label_text_color_mpl

        min_edge_width = 0.5
        max_edge_width = 4.0
        width_scale_factor = 2.0

        if G.number_of_nodes() > 0 and len(layer_nodes) > 1:
            for i in range(len(layer_nodes) - 1):
                for u_idx, u_node in enumerate(layer_nodes[i]):
                    for v_idx, v_node in enumerate(layer_nodes[i + 1]):
                        G.add_edge(u_node, v_node)
                        current_edge_color = default_edge_color_val
                        current_edge_width = 1.0

                        if self.show_weights_var.get() and self.nn_calculator.weights and \
                                i < len(self.nn_calculator.weights) and self.nn_calculator.weights[i] is not None:
                            try:
                                weight = self.nn_calculator.weights[i][v_idx, u_idx]
                                edge_labels[(u_node, v_node)] = f"{weight:.2f}"

                                if weight > 0.01:
                                    current_edge_color = 'skyblue'
                                elif weight < -0.01:
                                    current_edge_color = 'salmon'
                                else:
                                    current_edge_color = 'lightgray'

                                current_edge_width = abs(weight) * width_scale_factor
                                current_edge_width = max(min_edge_width, min(current_edge_width, max_edge_width))
                            except IndexError:
                                pass

                        edge_colors_list.append(current_edge_color)
                        edge_widths_list.append(current_edge_width)

        final_node_color_param = "skyblue"
        if self.colored_nodes_var.get() and node_display_colors_list and len(
                node_display_colors_list) == G.number_of_nodes():
            final_node_color_param = node_display_colors_list
        elif not self.colored_nodes_var.get() and G.number_of_nodes() > 0:
            final_node_color_param = "skyblue"

        effective_edge_color = default_edge_color_val
        effective_edge_width = 1.0

        if G.number_of_edges() > 0:
            if self.show_weights_var.get() and edge_colors_list and len(edge_colors_list) == G.number_of_edges():
                effective_edge_color = edge_colors_list

            if self.show_weights_var.get() and edge_widths_list and len(edge_widths_list) == G.number_of_edges():
                effective_edge_width = edge_widths_list
        else:
            effective_edge_color = []
            effective_edge_width = []

        if G.number_of_nodes() > 0:
            nx.draw(G, pos, ax=self.ax, with_labels=True, node_size=800,
                    node_color=final_node_color_param,
                    font_size=8, font_weight="bold", arrows=True,
                    arrowstyle="-|>", arrowsize=12,
                    edge_color=effective_edge_color,
                    width=effective_edge_width)

            if self.show_weights_var.get() and edge_labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=self.ax, font_size=7,
                                             font_color=button_text_color_mpl,
                                             bbox=dict(facecolor=frame_bg_color_mpl, edgecolor='none', alpha=0.7,
                                                       boxstyle='round,pad=0.2'))
        else:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No hay red para visualizar.", horizontalalignment='center',
                         verticalalignment='center',
                         transform=self.ax.transAxes, color=label_text_color_mpl)

        self.ax.set_title("Visualización de la Red Neuronal", color=label_text_color_mpl)
        plt.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def download_food101(self):
        """Descarga y prepara el dataset Food-101"""
        try:
            # Actualizar etiqueta de estado
            self.food101_status_label.configure(text="Descargando Food-101...")

            # Instalar tensorflow-datasets si no está instalado
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                print("Instalando tensorflow-datasets...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "tensorflow-datasets"
                ], check=True)
                import tensorflow_datasets as tfds

            # Descargar el dataset
            print("Descargando dataset Food-101...")
            (ds_train, ds_test), ds_info = tfds.load(
                'food101',
                split=['train', 'validation'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
                download=True
            )

            # Preparar los datos
            self.food101_class_names = ds_info.features['label'].names
            print(f"Clases disponibles: {len(self.food101_class_names)}")

            # Actualizar combo de clases
            self.food_classes_combo.configure(values=self.food101_class_names)

            # Convertir a numpy arrays (tomar una muestra pequeña para pruebas)
            print("Preparando datos de entrenamiento...")
            train_images = []
            train_labels = []

            # Tomar solo las primeras 1000 imágenes para pruebas rápidas
            for image, label in ds_train.take(1000):
                # Redimensionar imagen a 224x224
                image = tf.image.resize(image, [224, 224])
                # Normalizar pixeles
                image = tf.cast(image, tf.float32) / 255.0

                train_images.append(image.numpy())
                train_labels.append(label.numpy())

            self.food101_X_train = np.array(train_images)
            self.food101_y_train = np.array(train_labels)

            print("Preparando datos de prueba...")
            test_images = []
            test_labels = []

            # Tomar solo las primeras 200 imágenes de prueba
            for image, label in ds_test.take(200):
                # Redimensionar imagen a 224x224
                image = tf.image.resize(image, [224, 224])
                # Normalizar pixeles
                image = tf.cast(image, tf.float32) / 255.0

                test_images.append(image.numpy())
                test_labels.append(label.numpy())

            self.food101_X_test = np.array(test_images)
            self.food101_y_test = np.array(test_labels)

            print(f"Datos cargados:")
            print(f"- Entrenamiento: {self.food101_X_train.shape}")
            print(f"- Prueba: {self.food101_X_test.shape}")

            self.food101_status_label.configure(
                text=f"Food-101 descargado: {len(self.food101_X_train)} train, {len(self.food101_X_test)} test"
            )

            messagebox.showinfo("Éxito", "Dataset Food-101 descargado correctamente!")

        except subprocess.CalledProcessError as e:
            error_msg = f"Error instalando tensorflow-datasets: {str(e)}"
            print(error_msg)
            self.food101_status_label.configure(text="Error en instalación")
            messagebox.showerror("Error", error_msg)

        except Exception as e:
            error_msg = f"Error descargando Food-101: {str(e)}"
            print(error_msg)
            self.food101_status_label.configure(text="Error en descarga")
            messagebox.showerror("Error", error_msg)

    def prepare_food_dataset(self, dataset, num_classes, samples_per_class, split_name):
            """Prepara el dataset Food-101"""
            images = []
            labels = []

            # Contador por clase
            class_counts = [0] * num_classes
            total_target = num_classes * samples_per_class

            for image, label in dataset:
                label_val = label.numpy()

                # Solo tomar las primeras num_classes clases
                if label_val < num_classes and class_counts[label_val] < samples_per_class:
                    # Redimensionar imagen a 64x64
                    image_resized = tf.image.resize(image, [64, 64])
                    images.append(image_resized.numpy())
                    labels.append(label_val)
                    class_counts[label_val] += 1

                    # Mostrar progreso
                    total_collected = sum(class_counts)
                    if total_collected % 10 == 0:
                        progress = total_collected / total_target
                        self.food_training_progress.set(min(progress, 1.0))
                        self.update()

                # Terminar si tenemos suficientes muestras
                if all(count >= samples_per_class for count in class_counts):
                    break

            self.food_training_progress.set(0)
            return np.array(images), np.array(labels)

    def apply_food_preprocessing(self):
        """Preprocesa imágenes de Food-101 manteniendo originales y procesadas"""
        try:
            if self.food101_X_train is None:
                messagebox.showerror("Error", "Primero descarga las imágenes de Food-101")
                return

            # ✅ GUARDAR ORIGINALES (para visualización)
            self.food101_X_train_original = self.food101_X_train.copy()
            self.food101_X_test_original = self.food101_X_test.copy()

            # ✅ PROCESAR PARA MODELO (normalizar 0-1)
            self.food101_X_train = self.food101_X_train.astype('float32') / 255.0
            self.food101_X_test = self.food101_X_test.astype('float32') / 255.0

            messagebox.showinfo("Éxito", f"Preprocesamiento Food-101 completado:\n"
                                         f"• Originales guardadas para visualización\n"
                                         f"• Normalizadas (0-1) para entrenamiento\n"
                                         f"• Train: {self.food101_X_train.shape}\n"
                                         f"• Test: {self.food101_X_test.shape}")

            # Actualizar estado del botón de preprocesamiento
            self.food_preprocess_button.configure(text="✅ Preprocesamiento aplicado")

        except Exception as e:
            messagebox.showerror("Error", f"Error en preprocesamiento: {str(e)}")

    def show_food_samples(self):
        """Muestra muestras de imágenes de Food-101"""
        try:
            if self.food101_X_train is None:
                messagebox.showerror("Error", "Primero descarga las imágenes de Food-101")
                return

            # Limpiar frame de visualización
            for widget in self.food_visualization_frame.winfo_children():
                widget.destroy()

            # Configurar grid
            filas = 3
            columnas = 4
            num_samples = filas * columnas

            # Crear figura
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle('Muestras del Dataset Food-101 (Imágenes Originales)',
                         fontsize=16, weight='bold')

            # ✅ USAR IMÁGENES ORIGINALES SI ESTÁN DISPONIBLES
            if hasattr(self, 'food101_X_train_original') and self.food101_X_train_original is not None:
                # Usar originales guardadas
                images_to_show = self.food101_X_train_original[:num_samples]
            else:
                # Usar las actuales (pueden estar normalizadas o no)
                images_to_show = self.food101_X_train[:num_samples]

            # Mostrar muestras
            for i in range(num_samples):
                ax = fig.add_subplot(filas, columnas, i + 1)

                image = images_to_show[i]

                # ✅ MANEJAR DIFERENTES FORMATOS
                if image.max() <= 1.0:
                    # Imagen normalizada, desnormalizar para mostrar
                    display_image = (image * 255).astype(np.uint8)
                else:
                    # Imagen ya en rango 0-255
                    display_image = image.astype(np.uint8)

                ax.imshow(display_image)

                # Mostrar nombre de la clase
                class_name = self.food101_class_names[self.food101_y_train[i]]
                ax.set_title(f'{class_name.replace("_", " ").title()}', fontsize=10)
                ax.axis('off')

            plt.tight_layout()

            # Mostrar en GUI
            canvas = FigureCanvasTkAgg(fig, self.food_visualization_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Actualizar estado del botón
            self.show_food_samples_button.configure(text="✅ Muestras mostradas")

        except Exception as e:
            messagebox.showerror("Error", f"Error mostrando muestras: {str(e)}")

    def setup_food_augmentation(self):
        """Configura data augmentation para Food-101"""
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator

            # ✅ CONFIGURAR AUGMENTATION DESDE ORIGINALES
            self.food101_datagen = ImageDataGenerator(
                rotation_range=20,  # Rotación hasta 20 grados
                width_shift_range=0.2,  # Desplazamiento horizontal 20%
                height_shift_range=0.2,  # Desplazamiento vertical 20%
                zoom_range=0.2,  # Zoom 20%
                horizontal_flip=True,  # Flip horizontal
                fill_mode='nearest',  # Rellenar con píxeles cercanos
                brightness_range=[0.8, 1.2],  # Variación de brillo
                channel_shift_range=0.1  # Variación de canales de color
            )

            # ✅ FIT CON ORIGINALES (no normalizadas)
            if hasattr(self, 'food101_X_train_original') and self.food101_X_train_original is not None:
                self.food101_datagen.fit(self.food101_X_train_original)
                messagebox.showinfo("Éxito", "Data Augmentation configurado para Food-101\n"
                                             "• Configurado desde imágenes originales (0-255)\n"
                                             "• Transformaciones: rotación, zoom, flip, brillo\n"
                                             "• Listo para generar augmentaciones")

                # Actualizar estado del botón
                self.setup_food_augmentation_button.configure(text="✅ Augmentation configurado")
            else:
                messagebox.showerror("Error", "Primero aplica preprocesamiento para guardar originales")

        except ImportError:
            messagebox.showerror("Error", "TensorFlow no está instalado. Instala con: pip install tensorflow")
        except Exception as e:
            messagebox.showerror("Error", f"Error configurando augmentation: {str(e)}")

    def show_food_augmented(self):
        """Muestra antes/después de augmentation para Food-101"""
        try:
            if self.food101_datagen is None or self.food101_X_train is None:
                messagebox.showerror("Error", "Primero configura data augmentation")
                return

            # ✅ VERIFICAR QUE TENEMOS ORIGINALES
            if not hasattr(self, 'food101_X_train_original') or self.food101_X_train_original is None:
                messagebox.showerror("Error", "Ejecuta preprocesamiento primero para guardar originales")
                return

            # Limpiar frame de visualización
            for widget in self.food_visualization_frame.winfo_children():
                widget.destroy()

            filas = 2
            columnas = 4
            num = columnas

            # Crear figura con subplots
            fig = plt.figure(figsize=(16, 8))

            # ✅ ANTES - Usar imágenes ORIGINALES (0-255)
            fig.text(0.5, 0.95, 'ANTES (Imágenes Originales de Comida)',
                     ha='center', fontsize=14, weight='bold', color='blue')

            for i in range(num):
                ax = fig.add_subplot(4, columnas, i + 1)

                # Usar originales - ya están en formato correcto (0-255)
                original_image = self.food101_X_train_original[i]
                ax.imshow(original_image.astype(np.uint8))

                class_name = self.food101_class_names[self.food101_y_train[i]]
                ax.set_title(f'{class_name.replace("_", " ").title()}', fontsize=9)
                ax.axis('off')

            # ✅ DESPUÉS - Generar augmentación desde originales
            fig.text(0.5, 0.48, 'DESPUÉS (Con Data Augmentation)',
                     ha='center', fontsize=14, weight='bold', color='red')

            # Generar augmentación desde originales (0-255)
            original_batch = self.food101_X_train_original[:num]
            labels_batch = self.food101_y_train[:num]

            # Generar solo un batch de augmentaciones
            for X_aug, Y_aug in self.food101_datagen.flow(original_batch, labels_batch,
                                                          batch_size=num, shuffle=False):
                for i in range(num):
                    ax = fig.add_subplot(4, columnas, i + 1 + num)

                    # X_aug[i] ya está en formato correcto desde originales
                    augmented_image = X_aug[i]

                    # Asegurar que esté en rango 0-255
                    if augmented_image.max() <= 1.0:
                        augmented_image = (augmented_image * 255).astype(np.uint8)
                    else:
                        augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

                    ax.imshow(augmented_image)

                    class_name = self.food101_class_names[Y_aug[i]]
                    ax.set_title(f'{class_name.replace("_", " ").title()} (Augmented)', fontsize=9)
                    ax.axis('off')
                break  # Solo procesar el primer batch

            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.02)

            # Mostrar en GUI
            canvas = FigureCanvasTkAgg(fig, self.food_visualization_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Actualizar estado del botón
            self.show_food_augmented_button.configure(text="✅ Comparación mostrada")

        except Exception as e:
            messagebox.showerror("Error", f"Error mostrando augmentation: {str(e)}")
            import traceback
            print(f"Error detallado: {traceback.format_exc()}")

    def create_food101_cnn_model(self):
        """Crea modelo CNN para Food-101 con arquitectura corregida"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam

            if self.food101_X_train is None:
                messagebox.showerror("Error", "Primero descarga y preprocesa las imágenes")
                return

            # ✅ ARQUITECTURA CNN CORREGIDA
            self.food101_model = Sequential([
                # Primera capa convolucional
                Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                BatchNormalization(),
                MaxPooling2D(2, 2),

                # Segunda capa convolucional
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),

                # Tercera capa convolucional
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),

                # Cuarta capa convolucional
                Conv2D(256, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),

                # ✅ APLANAR ANTES DE CAPAS DENSAS
                Flatten(),

                # Capas densas
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.3),

                # ✅ CAPA DE SALIDA: 101 clases para Food-101
                Dense(101, activation='softmax')
            ])

            # ✅ COMPILAR MODELO
            self.food101_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Mostrar resumen del modelo
            model_summary = []
            self.food101_model.summary(print_fn=lambda x: model_summary.append(x))
            summary_text = '\n'.join(model_summary)

            messagebox.showinfo("Éxito", f"Modelo CNN creado exitosamente!\n\n"
                                         f"Arquitectura:\n"
                                         f"• 4 capas convolucionales con BatchNorm\n"
                                         f"• MaxPooling después de cada conv\n"
                                         f"• Flatten para conectar con densas\n"
                                         f"• 2 capas densas con Dropout\n"
                                         f"• Salida: 101 clases (Food-101)\n\n"
                                         f"Parámetros totales: {self.food101_model.count_params():,}")

            # Actualizar botón
            self.create_food_cnn_button.configure(text="✅ Modelo CNN creado")

            # Mostrar arquitectura detallada en consola
            print("=== RESUMEN DEL MODELO CNN ===")
            print(summary_text)

        except ImportError:
            messagebox.showerror("Error", "TensorFlow no está instalado. Instala con: pip install tensorflow")
        except Exception as e:
            messagebox.showerror("Error", f"Error creando modelo CNN: {str(e)}")
            import traceback
            print(f"Error detallado: {traceback.format_exc()}")

    def show_food_model_architecture(self):
            """Muestra la arquitectura del modelo Food-101"""
            try:
                if self.food101_model is None:
                    messagebox.showerror("Error", "Primero crea el modelo Food-101")
                    return

                # Limpiar frame de visualización
                for widget in self.food_visualization_frame.winfo_children():
                    widget.destroy()

                # Crear text widget para mostrar el resumen
                text_widget = tk.Text(self.food_visualization_frame, wrap=tk.WORD, font=('Courier', 10))
                text_widget.pack(fill="both", expand=True, padx=10, pady=10)

                # Capturar el resumen del modelo
                import io
                import sys

                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()

                self.food101_model.summary()

                sys.stdout = old_stdout
                model_summary = buffer.getvalue()

                text_widget.insert(tk.END, model_summary)
                text_widget.config(state=tk.DISABLED)

            except Exception as e:
                messagebox.showerror("Error", f"Error mostrando arquitectura: {str(e)}")

    def train_food_cnn_model(self):
        """Entrena el modelo CNN de Food-101"""
        try:
            if self.food101_model is None:
                messagebox.showerror("Error", "Primero crea el modelo CNN")
                return

            if self.food101_X_train is None:
                messagebox.showerror("Error", "Primero descarga y preprocesa las imágenes")
                return

            # ✅ VERIFICAR FORMAS DE LOS DATOS
            print(f"Forma X_train: {self.food101_X_train.shape}")
            print(f"Forma y_train: {self.food101_y_train.shape}")
            print(f"Forma X_test: {self.food101_X_test.shape}")
            print(f"Forma y_test: {self.food101_y_test.shape}")

            # Obtener parámetros de entrenamiento
            try:
                epochs = int(self.food_epochs_entry.get()) if self.food_epochs_entry.get() else 5
                batch_size = int(self.food_batch_size_entry.get()) if self.food_batch_size_entry.get() else 32
            except ValueError:
                epochs = 5
                batch_size = 32

            # ✅ VERIFICAR COMPATIBILIDAD DE DATOS
            if self.food101_X_train.shape[1:] != (224, 224, 3):
                messagebox.showerror("Error", f"Forma incorrecta de imágenes: {self.food101_X_train.shape[1:]}\n"
                                              f"Se esperaba: (224, 224, 3)")
                return

            # Configurar callbacks
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
            ]

            messagebox.showinfo("Entrenamiento", f"Iniciando entrenamiento...\n"
                                                 f"Épocas: {epochs}\n"
                                                 f"Batch size: {batch_size}\n"
                                                 f"Datos: {len(self.food101_X_train)} imágenes\n"
                                                 f"Clases: {len(self.food101_class_names)}")

            # ✅ ENTRENAR MODELO
            history = self.food101_model.fit(
                self.food101_X_train, self.food101_y_train,
                validation_data=(self.food101_X_test, self.food101_y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Evaluar modelo
            test_loss, test_accuracy = self.food101_model.evaluate(
                self.food101_X_test, self.food101_y_test, verbose=0
            )

            messagebox.showinfo("Entrenamiento Completado",
                                f"✅ Entrenamiento finalizado!\n\n"
                                f"Precisión en test: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n"
                                f"Pérdida en test: {test_loss:.4f}\n"
                                f"Épocas completadas: {len(history.history['loss'])}")

            # Actualizar botón
            self.train_food_cnn_button.configure(text="✅ Modelo entrenado")

            # Mostrar gráficas de entrenamiento
            self.plot_training_history(history)

        except Exception as e:
            messagebox.showerror("Error", f"Error entrenando modelo: {str(e)}")
            import traceback
            print(f"Error detallado: {traceback.format_exc()}")

    def plot_training_history(self, history):
        """Muestra gráficas del historial de entrenamiento"""
        try:
            # Limpiar frame de visualización
            for widget in self.food_visualization_frame.winfo_children():
                widget.destroy()

            # Crear figura con subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Gráfica de precisión
            ax1.plot(history.history['accuracy'], label='Entrenamiento', color='blue')
            ax1.plot(history.history['val_accuracy'], label='Validación', color='red')
            ax1.set_title('Precisión del Modelo')
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Precisión')
            ax1.legend()
            ax1.grid(True)

            # Gráfica de pérdida
            ax2.plot(history.history['loss'], label='Entrenamiento', color='blue')
            ax2.plot(history.history['val_loss'], label='Validación', color='red')
            ax2.set_title('Pérdida del Modelo')
            ax2.set_xlabel('Época')
            ax2.set_ylabel('Pérdida')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            # Mostrar en GUI
            canvas = FigureCanvasTkAgg(fig, self.food_visualization_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            print(f"Error mostrando historial: {str(e)}")

    def save_food_cnn_model(self):
            """Guarda el modelo Food-101 entrenado"""
            try:
                if self.food101_model is None:
                    messagebox.showerror("Error", "No hay modelo Food-101 para guardar")
                    return

                file_path = filedialog.asksaveasfilename(
                    defaultextension=".h5",
                    filetypes=[("H5 files", "*.h5"), ("All files", "*.*")],
                    title="Guardar modelo Food-101"
                )

                if file_path:
                    self.food101_model.save(file_path)
                    messagebox.showinfo("Éxito", f"Modelo Food-101 guardado en: {file_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Error guardando modelo: {str(e)}")

    def load_food_test_image(self):
        """Carga una imagen de comida para hacer predicción"""
        try:
            file_path = filedialog.askopenfilename(
                title="Seleccionar imagen de comida para predicción",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
            )

            if file_path:
                from PIL import Image

                # Cargar imagen
                image = Image.open(file_path)
                image = image.convert('RGB')  # Asegurar RGB

                # ✅ CAMBIO CRÍTICO: Redimensionar a 224x224 (igual que entrenamiento)
                image = image.resize((224, 224))  # Era 64x64, ahora 224x224

                # Convertir a array numpy y normalizar
                image_array = np.array(image) / 255.0
                self.food_test_image = image_array

                # Mostrar imagen cargada en la interfaz
                self.display_loaded_image(image_array)

                messagebox.showinfo("Éxito", f"Imagen cargada para predicción\n"
                                             f"Dimensiones: {image_array.shape}\n"
                                             f"Formato: RGB normalizado (0-1)")

                # Actualizar botón
                self.load_food_test_button.configure(text="✅ Imagen cargada")

        except Exception as e:
            messagebox.showerror("Error", f"Error cargando imagen: {str(e)}")
            import traceback
            print(f"Error detallado: {traceback.format_exc()}")

    def predict_food(self):
        """Hace predicción de comida con la imagen cargada"""
        try:
            if self.food101_model is None:
                messagebox.showerror("Error", "Primero entrena o carga un modelo")
                return

            if self.food_test_image is None:
                messagebox.showerror("Error", "Primero carga una imagen para predicción")
                return

            # ✅ VERIFICAR DIMENSIONES DE LA IMAGEN
            print(f"Forma de imagen cargada: {self.food_test_image.shape}")

            if self.food_test_image.shape != (224, 224, 3):
                messagebox.showerror("Error", f"Imagen con dimensiones incorrectas: {self.food_test_image.shape}\n"
                                              f"Se esperaba: (224, 224, 3)")
                return

            # ✅ PREPARAR IMAGEN PARA PREDICCIÓN (agregar dimensión de batch)
            image_batch = np.expand_dims(self.food_test_image, axis=0)  # (1, 224, 224, 3)
            print(f"Forma para predicción: {image_batch.shape}")

            # ✅ HACER PREDICCIÓN
            predictions = self.food101_model.predict(image_batch, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]

            # Obtener nombre de la clase
            predicted_class_name = self.food101_class_names[predicted_class_idx]

            # ✅ OBTENER TOP 5 PREDICCIONES
            top5_indices = np.argsort(predictions[0])[-5:][::-1]
            top5_predictions = []

            for idx in top5_indices:
                class_name = self.food101_class_names[idx]
                class_confidence = predictions[0][idx]
                top5_predictions.append((class_name, class_confidence))

            # Mostrar resultados
            result_text = f"🍴 PREDICCIÓN DE COMIDA\n\n"
            result_text += f"🏆 Predicción principal:\n"
            result_text += f"   {predicted_class_name.replace('_', ' ').title()}\n"
            result_text += f"   Confianza: {confidence:.2%}\n\n"
            result_text += f"📊 Top 5 predicciones:\n"

            for i, (class_name, conf) in enumerate(top5_predictions, 1):
                result_text += f"   {i}. {class_name.replace('_', ' ').title()}: {conf:.2%}\n"

            # Mostrar en label de predicción
            if hasattr(self, 'food_prediction_label'):
                self.food_prediction_label.configure(text=result_text)

            messagebox.showinfo("Predicción Completa", result_text)

            # Actualizar botón
            self.predict_food_button.configure(text="✅ Predicción realizada")

        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción: {str(e)}")
            import traceback
            print(f"Error detallado: {traceback.format_exc()}")

    def display_loaded_image(self, image_array):
        """Muestra la imagen cargada en la interfaz"""
        try:
            # Limpiar frame de visualización
            for widget in self.food_visualization_frame.winfo_children():
                widget.destroy()

            # Crear figura
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            # Mostrar imagen (desnormalizar para visualización)
            display_image = (image_array * 255).astype(np.uint8)
            ax.imshow(display_image)
            ax.set_title('Imagen Cargada para Predicción', fontsize=14, weight='bold')
            ax.axis('off')

            # Agregar información
            info_text = f"Dimensiones: {image_array.shape}\n"
            info_text += f"Rango: [{image_array.min():.3f}, {image_array.max():.3f}]\n"
            info_text += f"Formato: RGB normalizado"

            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            # Mostrar en GUI
            canvas = FigureCanvasTkAgg(fig, self.food_visualization_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            print(f"Error mostrando imagen: {str(e)}")

    def open_google_colab(self):
        """Abre Google Colab con un cuaderno vacío"""
        try:
            import webbrowser

            # 🔗 URL básica de Google Colab que abre un notebook vacío
            colab_url = "https://colab.research.google.com/"

            # Abrir Google Colab en el navegador
            webbrowser.open(colab_url)

            # Mostrar mensaje de confirmación
            messagebox.showinfo("Google Colab",
                                "🚀 ¡Google Colab abierto correctamente!\n\n"
                                "📝 Se ha abierto un cuaderno vacío\n"
                                "🔥 GPU/TPU disponibles si las activas\n"
                                "💡 Puedes empezar a escribir tu código\n\n"
                                "Tip: Ve a Runtime > Change runtime type para activar GPU")

        except Exception as e:
            # Manejo de errores si no se puede abrir el navegador
            messagebox.showerror("Error", f"No se pudo abrir Google Colab: {str(e)}")

            # Mostrar instrucciones manuales como fallback
            messagebox.showinfo("Acceso Manual",
                                "Por favor, abre manualmente:\n\n"
                                "🌐 https://colab.research.google.com/\n\n"
                                "Desde allí podrás crear un nuevo notebook.")

    def setup_ia_assistant_tab(self):
        """Configurar el tab de IA Assistant"""
        # Frame principal con scroll
        main_scroll = ctk.CTkScrollableFrame(self.tab_ia_assistant)
        main_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # Título principal
        title_label = ctk.CTkLabel(main_scroll, text="🤖 Asistente de IA con Transformers",
                                   font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=20)

        # Estado de carga de modelos
        self.ia_status_frame = ctk.CTkFrame(main_scroll)
        self.ia_status_frame.pack(fill="x", padx=10, pady=10)

        self.ia_status_label = ctk.CTkLabel(self.ia_status_frame,
                                            text="Estado: Modelos no cargados",
                                            font=ctk.CTkFont(size=12))
        self.ia_status_label.pack(pady=10)

        # Botón para cargar modelos
        self.load_models_button = ctk.CTkButton(self.ia_status_frame,
                                                text="Cargar Modelos de IA",
                                                command=self.load_ia_models,
                                                font=ctk.CTkFont(size=14, weight="bold"))
        self.load_models_button.pack(pady=10)

        # Separador
        separator1 = ctk.CTkFrame(main_scroll, height=2)
        separator1.pack(fill="x", padx=10, pady=20)

        # SECCIÓN 1: RESUMEN DE TEXTO
        summarizer_frame = ctk.CTkFrame(main_scroll)
        summarizer_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Título del resumen
        summarizer_title = ctk.CTkLabel(summarizer_frame,
                                        text="📝 Resumir Texto",
                                        font=ctk.CTkFont(size=18, weight="bold"))
        summarizer_title.pack(pady=15)

        # Área de entrada de texto
        ctk.CTkLabel(summarizer_frame, text="Ingrese el texto a resumir:",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10)

        self.input_text_box = ctk.CTkTextbox(summarizer_frame, height=150)
        self.input_text_box.pack(fill="x", padx=10, pady=5)

        # Configuración de resumen
        config_frame = ctk.CTkFrame(summarizer_frame)
        config_frame.pack(fill="x", padx=10, pady=10)

        # Longitud máxima del resumen
        ctk.CTkLabel(config_frame, text="Longitud máxima:").pack(side="left", padx=5)
        self.max_length_var = ctk.StringVar(value="150")
        max_length_entry = ctk.CTkEntry(config_frame, textvariable=self.max_length_var, width=80)
        max_length_entry.pack(side="left", padx=5)

        # Longitud mínima del resumen
        ctk.CTkLabel(config_frame, text="Longitud mínima:").pack(side="left", padx=5)
        self.min_length_var = ctk.StringVar(value="50")
        min_length_entry = ctk.CTkEntry(config_frame, textvariable=self.min_length_var, width=80)
        min_length_entry.pack(side="left", padx=5)

        # Botón de resumir
        self.summarize_button = ctk.CTkButton(summarizer_frame,
                                              text="🔍 Generar Resumen",
                                              command=self.summarize_text,
                                              font=ctk.CTkFont(size=14, weight="bold"))
        self.summarize_button.pack(pady=10)

        # Área de resultado del resumen
        ctk.CTkLabel(summarizer_frame, text="Resumen generado:",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10)

        self.summary_result_box = ctk.CTkTextbox(summarizer_frame, height=100)
        self.summary_result_box.pack(fill="x", padx=10, pady=5)

        # Separador
        separator2 = ctk.CTkFrame(main_scroll, height=2)
        separator2.pack(fill="x", padx=10, pady=20)

        # SECCIÓN 2: CHAT DE PREGUNTAS Y RESPUESTAS
        chat_frame = ctk.CTkFrame(main_scroll)
        chat_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Título del chat
        chat_title = ctk.CTkLabel(chat_frame,
                                  text="💬 Chat de Preguntas y Respuestas",
                                  font=ctk.CTkFont(size=18, weight="bold"))
        chat_title.pack(pady=15)

        # Área de conversación
        self.chat_display = ctk.CTkTextbox(chat_frame, height=300)
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=5)
        self.chat_display.configure(state="disabled")

        # Frame para entrada de pregunta
        question_frame = ctk.CTkFrame(chat_frame)
        question_frame.pack(fill="x", padx=10, pady=10)

        # Entrada de pregunta
        self.question_entry = ctk.CTkEntry(question_frame,
                                           placeholder_text="Escribe tu pregunta aquí...",
                                           font=ctk.CTkFont(size=12))
        self.question_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # Botón de enviar
        self.send_question_button = ctk.CTkButton(question_frame,
                                                  text="Enviar",
                                                  command=self.send_question,
                                                  font=ctk.CTkFont(size=12, weight="bold"),
                                                  width=80)
        self.send_question_button.pack(side="right")

        # Botón para limpiar conversación
        self.clear_chat_button = ctk.CTkButton(chat_frame,
                                               text="🗑️ Limpiar Conversación",
                                               command=self.clear_conversation,
                                               font=ctk.CTkFont(size=12))
        self.clear_chat_button.pack(pady=10)

        # Configurar enter key para enviar pregunta
        self.question_entry.bind("<Return>", lambda event: self.send_question())

        # Inicializar el chat con mensaje de bienvenida
        self.add_message_to_chat("🤖 IA Assistant",
                                 "¡Hola! Soy tu asistente de IA. Puedes hacerme preguntas sobre cualquier tema. Para empezar, asegúrate de cargar los modelos usando el botón superior.")

    def load_ia_models(self):
        """Cargar los modelos de IA en un hilo separado"""
        if self.is_loading_models:
            return

        self.is_loading_models = True
        self.load_models_button.configure(state="disabled", text="Cargando modelos...")
        self.ia_status_label.configure(text="Estado: Cargando modelos de IA... (Esto puede tomar varios minutos)")

        # Ejecutar carga en hilo separado
        thread = threading.Thread(target=self._load_models_thread)
        thread.daemon = True
        thread.start()

    def _load_models_thread(self):
        """Cargar modelos en hilo separado"""
        try:
            # Cargar modelo de resumen
            self.ia_status_label.configure(text="Estado: Cargando modelo de resumen...")
            self.summarizer = pipeline("summarization",
                                       model="facebook/bart-large-cnn",
                                       device=-1)  # CPU

            # Cargar modelo de Q&A con mejor configuración
            self.ia_status_label.configure(text="Estado: Cargando modelo de preguntas y respuestas...")
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Cargar tokenizer y modelo por separado para mejor control
            self.qa_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.qa_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

            # Configurar tokens especiales
            if self.qa_tokenizer.pad_token is None:
                self.qa_tokenizer.pad_token = self.qa_tokenizer.eos_token

            # Crear pipeline con configuración mejorada
            self.qa_pipeline = pipeline("text-generation",
                                        model=self.qa_model,
                                        tokenizer=self.qa_tokenizer,
                                        device=-1)

            # Actualizar UI en el hilo principal
            self.after(0, self._models_loaded_successfully)

        except Exception as e:
            error_msg = f"Error al cargar modelos: {str(e)}"
            self.after(0, lambda: self._models_loading_failed(error_msg))

    def _models_loaded_successfully(self):
        """Callback cuando los modelos se cargan exitosamente"""
        self.is_loading_models = False
        self.load_models_button.configure(state="normal", text="Modelos Cargados ✓")
        self.load_models_button.configure(fg_color="green")
        self.ia_status_label.configure(text="Estado: Modelos cargados correctamente ✓")

        # Habilitar botones
        self.summarize_button.configure(state="normal")
        self.send_question_button.configure(state="normal")

        # Actualizar mensaje de bienvenida
        self.add_message_to_chat("🤖 IA Assistant",
                                 "¡Los modelos se han cargado correctamente! Ahora puedes usar todas las funcionalidades.")

    def _models_loading_failed(self, error_msg):
        """Callback cuando falla la carga de modelos"""
        self.is_loading_models = False
        self.load_models_button.configure(state="normal", text="Reintentar Carga")
        self.ia_status_label.configure(text=f"Estado: Error - {error_msg}")
        messagebox.showerror("Error", error_msg)

    def summarize_text(self):
        """Resumir el texto ingresado"""
        if not self.summarizer:
            messagebox.showwarning("Advertencia", "Primero debe cargar los modelos de IA")
            return

        text = self.input_text_box.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Advertencia", "Por favor ingrese un texto para resumir")
            return

        try:
            # Configurar parámetros
            max_length = int(self.max_length_var.get()) if self.max_length_var.get().isdigit() else 150
            min_length = int(self.min_length_var.get()) if self.min_length_var.get().isdigit() else 50

            # Deshabilitar botón durante procesamiento
            self.summarize_button.configure(state="disabled", text="Procesando...")

            # Crear hilo para el resumen
            thread = threading.Thread(target=self._summarize_thread, args=(text, max_length, min_length))
            thread.daemon = True
            thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar el resumen: {str(e)}")
            self.summarize_button.configure(state="normal", text="🔍 Generar Resumen")

    def _summarize_thread(self, text, max_length, min_length):
        """Ejecutar resumen en hilo separado"""
        try:
            # Dividir texto largo si es necesario
            if len(text) > 1024:
                # Para textos muy largos, tomar las primeras 1024 palabras
                words = text.split()
                text = ' '.join(words[:1024])

            summary = self.summarizer(text,
                                      max_length=max_length,
                                      min_length=min_length,
                                      do_sample=False)

            summary_text = summary[0]['summary_text']

            # Actualizar UI en hilo principal
            self.after(0, lambda: self._display_summary(summary_text))

        except Exception as e:
            error_msg = f"Error durante el resumen: {str(e)}"
            self.after(0, lambda: self._summary_failed(error_msg))

    def _display_summary(self, summary_text):
        """Mostrar el resumen generado"""
        self.summary_result_box.delete("1.0", "end")
        self.summary_result_box.insert("1.0", summary_text)
        self.summarize_button.configure(state="normal", text="🔍 Generar Resumen")

    def _summary_failed(self, error_msg):
        """Manejar error en resumen"""
        messagebox.showerror("Error", error_msg)
        self.summarize_button.configure(state="normal", text="🔍 Generar Resumen")

    def send_question(self):
        """Enviar pregunta al chat"""
        if not self.qa_pipeline:
            messagebox.showwarning("Advertencia", "Primero debe cargar los modelos de IA")
            return

        question = self.question_entry.get().strip()
        if not question:
            return

        # Limpiar entrada
        self.question_entry.delete(0, "end")

        # Agregar pregunta al chat
        self.add_message_to_chat("👤 Tú", question)

        # Deshabilitar botón durante procesamiento
        self.send_question_button.configure(state="disabled", text="...")

        # Procesar pregunta en hilo separado
        thread = threading.Thread(target=self._process_question_thread, args=(question,))
        thread.daemon = True
        thread.start()

    def _process_question_thread(self, question):
        """Procesar pregunta en hilo separado - Versión mejorada"""
        try:
            # Limpiar y preparar la pregunta
            question = question.strip()
            if not question:
                raise Exception("Pregunta vacía")

            # Crear contexto más simple y efectivo
            # DialoGPT funciona mejor con conversaciones cortas
            conversation_context = ""
            if hasattr(self, 'conversation_history') and self.conversation_history:
                # Solo usar los últimos 2-3 intercambios para evitar contexto muy largo
                recent_messages = self.conversation_history[-4:]  # 2 intercambios
                for msg in recent_messages:
                    if msg['role'] == 'Usuario':
                        conversation_context += f"{msg['content']}{self.qa_tokenizer.eos_token}"
                    elif msg['role'] == 'IA Assistant':
                        conversation_context += f"{msg['content']}{self.qa_tokenizer.eos_token}"

            # Preparar el prompt
            full_prompt = f"{conversation_context}{question}{self.qa_tokenizer.eos_token}"

            # Generar respuesta con parámetros optimizados
            response = self.qa_pipeline(
                full_prompt,
                max_length=len(full_prompt) + 80,  # Más conservador
                min_length=len(full_prompt) + 10,  # Asegurar respuesta mínima
                num_return_sequences=1,
                temperature=0.8,  # Más creativo
                top_p=0.9,  # Nucleus sampling
                do_sample=True,
                pad_token_id=self.qa_tokenizer.eos_token_id,
                eos_token_id=self.qa_tokenizer.eos_token_id
            )

            # Extraer respuesta
            generated_text = response[0]['generated_text']

            # Extraer solo la nueva respuesta
            if full_prompt in generated_text:
                assistant_response = generated_text[len(full_prompt):].strip()
            else:
                # Fallback: tomar todo después del último token EOS
                parts = generated_text.split(self.qa_tokenizer.eos_token)
                assistant_response = parts[-1].strip() if parts else ""

            # Limpiar la respuesta
            assistant_response = assistant_response.replace(self.qa_tokenizer.eos_token, "").strip()

            # Validar respuesta
            if not assistant_response or len(assistant_response) < 5:
                assistant_response = self._generate_fallback_response(question)

            # Actualizar UI en hilo principal
            self.after(0, lambda: self._display_response(assistant_response))

        except Exception as e:
            error_msg = f"Error al procesar la pregunta: {str(e)}"
            self.after(0, lambda: self._question_failed(error_msg))

    def _display_response(self, response):
        """Mostrar respuesta del asistente"""
        self.add_message_to_chat("🤖 IA Assistant", response)
        self.send_question_button.configure(state="normal", text="Enviar")

    def _question_failed(self, error_msg):
        """Manejar error en pregunta"""
        self.add_message_to_chat("🤖 IA Assistant", f"Lo siento, ocurrió un error: {error_msg}")
        self.send_question_button.configure(state="normal", text="Enviar")

    def add_message_to_chat(self, sender, message):
        """Agregar mensaje al chat"""
        from datetime import datetime

        # Guardar en historial
        self.conversation_history.append({
            "role": sender,
            "content": message
        })

        # Mostrar en UI
        self.chat_display.configure(state="normal")

        # Formato del mensaje
        timestamp = datetime.now().strftime("%H:%M")
        formatted_message = f"[{timestamp}] {sender}:\n{message}\n\n"

        self.chat_display.insert("end", formatted_message)
        self.chat_display.configure(state="disabled")

        # Scroll hacia abajo
        self.chat_display.see("end")

    def clear_conversation(self):
        """Limpiar la conversación"""
        self.conversation_history.clear()
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

        # Mensaje de bienvenida
        self.add_message_to_chat("🤖 IA Assistant", "Conversación limpiada. ¡Puedes empezar una nueva conversación!")

if __name__ == '__main__':
    app = App()
    app.mainloop()