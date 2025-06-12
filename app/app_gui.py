import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional

# Importar la calculadora de red neuronal
from neural_network.nn_calculator import NeuralNetworkCalculator

# Importaciones para visualización (si las usas)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from matplotlib.figure import Figure

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

        # --- TabView Principal ---
        self.tab_view = ctk.CTkTabview(self.main_container)
        self.tab_view.pack(pady=10, padx=10, fill="both", expand=True)

        self.tab_architecture = self.tab_view.add("1. Arquitectura")
        self.tab_parameters = self.tab_view.add("2. Pesos y Biases")
        self.tab_forward_prop = self.tab_view.add("3. Prop. Adelante")
        self.tab_back_prop = self.tab_view.add("4. Retropropagación")
        self.tab_visualization = self.tab_view.add("5. Visualización")

        # --- Inicializar contenido de las pestañas ---
        self.layer_frames: List[ctk.CTkFrame] = []  # Para entradas de neuronas y activaciones
        self.weight_textboxes: List[ctk.CTkTextbox] = []
        self.bias_textboxes: List[ctk.CTkTextbox] = []

        self.setup_architecture_tab(self.tab_architecture)
        self.setup_parameters_tab(self.tab_parameters)  # Renombrado de setup_weights_biases_tab
        self.setup_forward_prop_tab(self.tab_forward_prop)
        self.setup_back_prop_tab(self.tab_back_prop)
        self.setup_visualization_tab(self.tab_visualization)

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

if __name__ == '__main__':
    app = App()
    app.mainloop()