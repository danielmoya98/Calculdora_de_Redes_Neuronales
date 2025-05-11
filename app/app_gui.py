import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from matplotlib.figure import Figure
from neural_network.nn_calculator import NeuralNetworkCalculator

# Configuración de CustomTkinter
ctk.set_appearance_mode("System")  # Modo: "System", "Dark" o "Light"
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana principal
        self.title("Calculadora de Red Neuronal Feedforward")
        self.geometry("1200x800")

        # Crear la calculadora de red neuronal
        self.nn_calculator = NeuralNetworkCalculator()

        # Variables para la GUI
        self.layer_frames = []  # Lista de frames para cada capa
        self.weight_textboxes = []  # Lista de textboxes para matrices de pesos
        self.bias_textboxes = []  # Lista de textboxes para vectores de biases

        # Crear el contenedor principal
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Crear pestañas
        self.tabview = ctk.CTkTabview(self.main_container)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Agregar pestañas (modificar esta parte para incluir "Visualización")
        self.tabview.add("Arquitectura")
        self.tabview.add("Pesos y Biases")
        self.tabview.add("Cálculo")
        self.tabview.add("Visualización")  # Nueva pestaña para visualización

        # Configurar pestañas
        self.setup_architecture_tab()
        self.setup_weights_biases_tab()
        self.setup_calculation_tab()
        self.setup_visualization_tab()


        # Estado inicial: Solo entrada y salida
        self.update_layer_frames()

    def setup_architecture_tab(self):
        """Configura la pestaña de arquitectura de la red"""
        architecture_frame = self.tabview.tab("Arquitectura")

        # Frame superior para controles
        controls_frame = ctk.CTkFrame(architecture_frame)
        controls_frame.pack(fill="x", padx=10, pady=10)

        # Botones para agregar y eliminar capas
        add_layer_btn = ctk.CTkButton(
            controls_frame,
            text="Agregar Capa",
            command=self.add_layer
        )
        add_layer_btn.pack(side="left", padx=10, pady=10)

        remove_layer_btn = ctk.CTkButton(
            controls_frame,
            text="Eliminar Última Capa",
            command=self.remove_layer
        )
        remove_layer_btn.pack(side="left", padx=10, pady=10)

        # Botón para guardar la arquitectura
        save_architecture_btn = ctk.CTkButton(
            controls_frame,
            text="Guardar Arquitectura",
            command=self.save_architecture
        )
        save_architecture_btn.pack(side="right", padx=10, pady=10)

        # Frame para las capas
        self.layers_frame = ctk.CTkScrollableFrame(architecture_frame)
        self.layers_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_weights_biases_tab(self):
        """Configura la pestaña de pesos y biases"""
        weights_biases_frame = self.tabview.tab("Pesos y Biases")

        # Frame superior para controles
        controls_frame = ctk.CTkFrame(weights_biases_frame)
        controls_frame.pack(fill="x", padx=10, pady=10)

        # Botón para guardar pesos y biases
        save_params_btn = ctk.CTkButton(
            controls_frame,
            text="Guardar Pesos y Biases",
            command=self.save_parameters
        )
        save_params_btn.pack(side="left", padx=10, pady=10)

        # Botón para inicializar pesos y biases con valores aleatorios
        random_params_btn = ctk.CTkButton(
            controls_frame,
            text="Inicializar Aleatoriamente",
            command=self.initialize_random_parameters
        )
        random_params_btn.pack(side="left", padx=10, pady=10)

        # Frame para los pesos y biases
        self.weights_biases_frame = ctk.CTkScrollableFrame(weights_biases_frame)
        self.weights_biases_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_calculation_tab(self):
        """Configura la pestaña de cálculo"""
        calculation_frame = self.tabview.tab("Cálculo")

        # Frame para la entrada
        input_frame = ctk.CTkFrame(calculation_frame)
        input_frame.pack(fill="x", padx=10, pady=10)

        # Etiqueta para la entrada
        input_label = ctk.CTkLabel(
            input_frame,
            text="Vector de entrada (valores separados por comas):"
        )
        input_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Campo de texto para la entrada
        self.input_textbox = ctk.CTkTextbox(input_frame, height=50)
        self.input_textbox.pack(fill="x", padx=10, pady=(0, 10))

        # Botón para calcular
        calculate_btn = ctk.CTkButton(
            input_frame,
            text="Calcular",
            command=self.calculate
        )
        calculate_btn.pack(padx=10, pady=10)

        # Frame para los resultados
        results_frame = ctk.CTkFrame(calculation_frame)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Etiqueta para los resultados
        results_label = ctk.CTkLabel(
            results_frame,
            text="Resultados:"
        )
        results_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Campo de texto para los resultados
        self.results_textbox = ctk.CTkTextbox(results_frame)
        self.results_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.results_textbox.configure(state="disabled")

        # Botón para mostrar/ocultar resultados intermedios
        self.show_intermediate_var = tk.BooleanVar(value=True)
        show_intermediate_checkbox = ctk.CTkCheckBox(
            results_frame,
            text="Mostrar resultados intermedios",
            variable=self.show_intermediate_var,
            command=self.calculate  # Recalcular cuando cambie
        )
        show_intermediate_checkbox.pack(anchor="w", padx=10, pady=(0, 10))

        # Frame para operaciones adicionales
        operations_frame = ctk.CTkFrame(calculation_frame)
        operations_frame.pack(fill="x", padx=10, pady=10)

        # Botones para guardar y cargar modelo
        save_model_btn = ctk.CTkButton(
            operations_frame,
            text="Guardar Modelo",
            command=self.save_model
        )
        save_model_btn.pack(side="left", padx=10, pady=10)

        load_model_btn = ctk.CTkButton(
            operations_frame,
            text="Cargar Modelo",
            command=self.load_model
        )
        load_model_btn.pack(side="left", padx=10, pady=10)

        # Botón para limpiar todo
        reset_btn = ctk.CTkButton(
            operations_frame,
            text="Resetear/Limpiar Todo",
            command=self.reset_all
        )
        reset_btn.pack(side="right", padx=10, pady=10)

    def add_layer(self):
        """Agrega una nueva capa a la red"""
        if len(self.layer_frames) < 2:
            # Si solo tenemos entrada, agregar salida (capa 2)
            self.update_layer_frames()
        else:
            # Agregar capa oculta antes de la capa de salida
            self.layer_frames.insert(-1, None)
            self.update_layer_frames()

    def remove_layer(self):
        """Elimina la última capa oculta de la red"""
        if len(self.layer_frames) > 2:
            # Eliminar la última capa oculta (no la de salida)
            self.layer_frames.pop(-2)
            self.update_layer_frames()

    def update_layer_frames(self):
        """Actualiza los frames de las capas en la interfaz"""
        # Limpiar el frame de capas
        for widget in self.layers_frame.winfo_children():
            widget.destroy()

        # Inicializar la lista de frames si está vacía
        if not self.layer_frames:
            self.layer_frames = [None, None]  # Entrada y salida mínimo

        # Crear/actualizar frames para cada capa
        for i, frame in enumerate(self.layer_frames):
            layer_frame = ctk.CTkFrame(self.layers_frame)
            layer_frame.pack(fill="x", padx=10, pady=10)

            # Tipo de capa (entrada, oculta, salida)
            if i == 0:
                layer_type = "Entrada"
            elif i == len(self.layer_frames) - 1:
                layer_type = "Salida"
            else:
                layer_type = f"Oculta {i}"

            # Etiqueta de la capa
            layer_label = ctk.CTkLabel(
                layer_frame,
                text=f"Capa {i + 1} ({layer_type}):"
            )
            layer_label.pack(side="left", padx=10, pady=10)

            # Campo para el número de neuronas
            neurons_label = ctk.CTkLabel(
                layer_frame,
                text="Neuronas:"
            )
            neurons_label.pack(side="left", padx=(20, 5), pady=10)

            neurons_entry = ctk.CTkEntry(layer_frame, width=60)
            neurons_entry.pack(side="left", padx=(0, 20), pady=10)
            neurons_entry.insert(0, "3")  # Valor por defecto

            # Dropdown para la función de activación (excepto para la capa de entrada)
            if i > 0:
                activation_label = ctk.CTkLabel(
                    layer_frame,
                    text="Activación:"
                )
                activation_label.pack(side="left", padx=(20, 5), pady=10)

                activation_options = ["Sigmoide", "ReLU", "Tanh", "Lineal", "Softmax"]
                default_activation = "ReLU" if i < len(self.layer_frames) - 1 else "Softmax"

                activation_dropdown = ctk.CTkOptionMenu(
                    layer_frame,
                    values=activation_options
                )
                activation_dropdown.pack(side="left", padx=(0, 20), pady=10)
                activation_dropdown.set(default_activation)

            # Actualizar la lista de frames
            self.layer_frames[i] = layer_frame

        # Actualizar la pestaña de pesos y biases
        self.update_weights_biases_frames()

    def update_weights_biases_frames(self):
        """Actualiza los frames de pesos y biases en la interfaz"""
        # Limpiar el frame de pesos y biases
        for widget in self.weights_biases_frame.winfo_children():
            widget.destroy()

        # Reiniciar listas de textboxes
        self.weight_textboxes = []
        self.bias_textboxes = []

        # Crear frames para cada par de capas conectadas
        for i in range(len(self.layer_frames) - 1):
            # Crear frame para este par de capas
            layer_params_frame = ctk.CTkFrame(self.weights_biases_frame)
            layer_params_frame.pack(fill="x", padx=10, pady=10)

            # Obtener el número de neuronas en cada capa
            from_neurons = int(self.layer_frames[i].winfo_children()[2].get())
            to_neurons = int(self.layer_frames[i + 1].winfo_children()[2].get())

            # Etiqueta para los pesos
            weights_label = ctk.CTkLabel(
                layer_params_frame,
                text=f"Pesos Capa {i + 1}->{i + 2} (dim: {to_neurons}x{from_neurons}):"
            )
            weights_label.pack(anchor="w", padx=10, pady=(10, 5))

            # Textbox para la matriz de pesos
            weights_textbox = ctk.CTkTextbox(layer_params_frame, height=80)
            weights_textbox.pack(fill="x", padx=10, pady=(0, 10))
            self.weight_textboxes.append(weights_textbox)

            # Etiqueta para los biases
            bias_label = ctk.CTkLabel(
                layer_params_frame,
                text=f"Biases Capa {i + 2} (dim: {to_neurons}):"
            )
            bias_label.pack(anchor="w", padx=10, pady=(10, 5))

            # Textbox para el vector de biases
            bias_textbox = ctk.CTkTextbox(layer_params_frame, height=40)
            bias_textbox.pack(fill="x", padx=10, pady=(0, 10))
            self.bias_textboxes.append(bias_textbox)

    def save_architecture(self):
        """Guarda la arquitectura de la red en el modelo"""
        try:
            # Obtener número de neuronas por capa
            layers_neurons = []
            for frame in self.layer_frames:
                neurons = int(frame.winfo_children()[2].get())
                layers_neurons.append(neurons)

            # Obtener funciones de activación por capa
            activations = ["Ninguna"]  # La capa de entrada no tiene activación
            for i in range(1, len(self.layer_frames)):
                activation = self.layer_frames[i].winfo_children()[4].get()
                activations.append(activation)

            # Configurar el modelo
            self.nn_calculator.set_architecture(layers_neurons, activations)

            # Actualizar la pestaña de pesos y biases
            self.update_weights_biases_frames()

            # Cambiar a la pestaña de pesos y biases
            self.tabview.set("Pesos y Biases")

            messagebox.showinfo("Arquitectura Guardada", "La arquitectura de la red ha sido guardada correctamente.")

            self.tabview.set("Visualización")
            self.visualize_network()

        except ValueError as e:
            messagebox.showerror("Error", f"Error al guardar la arquitectura: {str(e)}")



    def save_parameters(self):
        """Guarda los pesos y biases en el modelo"""
        try:
            weights = []
            biases = []

            # Obtener las matrices de pesos y vectores de biases
            for i in range(len(self.weight_textboxes)):
                # Pesos
                weight_text = self.weight_textboxes[i].get("1.0", "end-1c").strip()
                if not weight_text:
                    messagebox.showerror("Error", f"La matriz de pesos para las capas {i + 1}->{i + 2} está vacía.")
                    return

                weight_rows = weight_text.split("\n")
                weight_matrix = []

                for row in weight_rows:
                    if not row.strip():
                        continue
                    weight_row = [float(val.strip()) for val in row.split(",")]
                    weight_matrix.append(weight_row)

                weight_matrix = np.array(weight_matrix)

                # Verificar dimensiones
                from_neurons = self.nn_calculator.layers_neurons[i]
                to_neurons = self.nn_calculator.layers_neurons[i + 1]

                if weight_matrix.shape != (to_neurons, from_neurons):
                    messagebox.showerror(
                        "Error",
                        f"Dimensiones incorrectas para la matriz de pesos {i + 1}->{i + 2}. "
                        f"Se esperaba ({to_neurons}, {from_neurons}), "
                        f"pero se obtuvo {weight_matrix.shape}."
                    )
                    return

                weights.append(weight_matrix)

                # Biases
                bias_text = self.bias_textboxes[i].get("1.0", "end-1c").strip()
                if not bias_text:
                    messagebox.showerror("Error", f"El vector de biases para la capa {i + 2} está vacío.")
                    return

                bias_vector = np.array([float(val.strip()) for val in bias_text.split(",")])

                # Verificar dimensiones
                if bias_vector.shape[0] != to_neurons:
                    messagebox.showerror(
                        "Error",
                        f"Dimensiones incorrectas para el vector de biases {i + 2}. "
                        f"Se esperaba ({to_neurons},), "
                        f"pero se obtuvo {bias_vector.shape}."
                    )
                    return

                biases.append(bias_vector.reshape(-1, 1))

            # Configurar el modelo
            self.nn_calculator.set_parameters(weights, biases)

            # Cambiar a la pestaña de cálculo
            self.tabview.set("Cálculo")

            messagebox.showinfo("Parámetros Guardados", "Los pesos y biases han sido guardados correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar los parámetros: {str(e)}")

    def initialize_random_parameters(self):
        """Inicializa los pesos y biases con valores aleatorios"""
        try:
            # Verificar que la arquitectura esté configurada
            if not self.nn_calculator.layers_neurons:
                messagebox.showerror("Error", "La arquitectura de la red no está configurada.")
                return

            # Inicializar pesos y biases aleatorios
            weights = []
            biases = []

            for i in range(len(self.nn_calculator.layers_neurons) - 1):
                from_neurons = self.nn_calculator.layers_neurons[i]
                to_neurons = self.nn_calculator.layers_neurons[i + 1]

                # Inicializar pesos con distribución normal (Xavier/Glorot)
                weight_matrix = np.random.randn(to_neurons, from_neurons) * np.sqrt(1 / from_neurons)
                weights.append(weight_matrix)

                # Inicializar biases con ceros
                bias_vector = np.zeros((to_neurons, 1))
                biases.append(bias_vector)

                # Actualizar textboxes
                self.weight_textboxes[i].delete("1.0", "end")
                weight_str = "\n".join([",".join([str(round(val, 4)) for val in row]) for row in weight_matrix])
                self.weight_textboxes[i].insert("1.0", weight_str)

                self.bias_textboxes[i].delete("1.0", "end")
                bias_str = ",".join([str(round(val[0], 4)) for val in bias_vector])
                self.bias_textboxes[i].insert("1.0", bias_str)

            # Configurar el modelo
            self.nn_calculator.set_parameters(weights, biases)

            messagebox.showinfo("Parámetros Inicializados", "Los pesos y biases han sido inicializados aleatoriamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al inicializar los parámetros: {str(e)}")

    def calculate(self):
        """Realiza el cálculo de propagación hacia adelante"""
        try:
            # Verificar que el modelo esté configurado
            if not self.nn_calculator.layers_neurons or not self.nn_calculator.weights or not self.nn_calculator.biases:
                messagebox.showerror("Error", "El modelo no está completamente configurado.")
                return

            # Obtener el vector de entrada
            input_text = self.input_textbox.get("1.0", "end-1c").strip()
            if not input_text:
                messagebox.showerror("Error", "El vector de entrada está vacío.")
                return

            input_vector = np.array([float(val.strip()) for val in input_text.split(",")])

            # Verificar dimensiones
            if input_vector.shape[0] != self.nn_calculator.layers_neurons[0]:
                messagebox.showerror(
                    "Error",
                    f"Dimensiones incorrectas para el vector de entrada. "
                    f"Se esperaba ({self.nn_calculator.layers_neurons[0]},), "
                    f"pero se obtuvo {input_vector.shape}."
                )
                return

            # Reshape para operaciones matriciales
            input_vector = input_vector.reshape(-1, 1)

            # Realizar propagación hacia adelante
            output = self.nn_calculator.forward_propagation(input_vector)

            # Continuación de la clase App
            # Habilitar la edición del textbox de resultados
            self.results_textbox.configure(state="normal")
            self.results_textbox.delete("1.0", "end")

            # Formatear el resultado final
            output_str = "Salida de la red:\n"
            output_formatted = np.array2string(output, precision=6, separator=', ')
            output_str += output_formatted + "\n\n"

            # Mostrar resultados intermedios si está activado
            if self.show_intermediate_var.get():
                output_str += "Resultados intermedios:\n"
                for key, value in self.nn_calculator.intermediate_results.items():
                    output_str += f"{key}:\n"
                    value_formatted = np.array2string(value, precision=6, separator=', ')
                    output_str += value_formatted + "\n\n"

            self.results_textbox.insert("1.0", output_str)

            # Deshabilitar la edición del textbox de resultados
            self.results_textbox.configure(state="disabled")
        except ValueError as e:
            messagebox.showerror("Error", f"Error en el cálculo: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {str(e)}")

    def save_model(self):
        """Guarda el modelo completo en un archivo JSON"""
        try:
            # Verificar que el modelo esté configurado
            if not self.nn_calculator.layers_neurons:
                messagebox.showerror("Error", "El modelo no está configurado.")
                return

            # Abrir diálogo para guardar archivo
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")]
            )

            if not file_path:
                return  # El usuario canceló

            # Convertir modelo a diccionario serializable
            model_dict = self.nn_calculator.to_json()

            # Guardar en archivo JSON
            with open(file_path, 'w') as f:
                json.dump(model_dict, f, indent=4)

            messagebox.showinfo("Modelo Cargado", f"El modelo ha sido cargado desde {file_path}")
            self.tabview.set("Visualización")
            self.visualize_network()

        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar el modelo: {str(e)}")

    def load_model(self):
        """Carga un modelo desde un archivo JSON"""
        try:
            # Abrir diálogo para cargar archivo
            file_path = filedialog.askopenfilename(
                filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")]
            )

            if not file_path:
                return  # El usuario canceló

            # Cargar desde archivo JSON
            with open(file_path, 'r') as f:
                model_dict = json.load(f)

            # Crear modelo desde diccionario
            self.nn_calculator = NeuralNetworkCalculator.from_json(model_dict)

            # Actualizar la interfaz
            self.layer_frames = [None] * len(self.nn_calculator.layers_neurons)
            self.update_layer_frames()

            # Actualizar los campos en la pestaña de arquitectura
            for i, frame in enumerate(self.layer_frames):
                # Número de neuronas
                frame.winfo_children()[2].delete(0, "end")
                frame.winfo_children()[2].insert(0, str(self.nn_calculator.layers_neurons[i]))

                # Función de activación (excepto para la capa de entrada)
                if i > 0:
                    activation = self.nn_calculator.activations[i]
                    frame.winfo_children()[4].set(activation)

            # Actualizar textboxes de pesos y biases
            for i in range(len(self.nn_calculator.weights)):
                # Pesos
                self.weight_textboxes[i].delete("1.0", "end")
                weight_matrix = self.nn_calculator.weights[i]
                weight_str = "\n".join([",".join([str(val) for val in row]) for row in weight_matrix])
                self.weight_textboxes[i].insert("1.0", weight_str)

                # Biases
                self.bias_textboxes[i].delete("1.0", "end")
                bias_vector = self.nn_calculator.biases[i]
                bias_str = ",".join([str(val[0]) for val in bias_vector])
                self.bias_textboxes[i].insert("1.0", bias_str)

            messagebox.showinfo("Modelo Cargado", f"El modelo ha sido cargado desde {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el modelo: {str(e)}")

    def reset_all(self):
        """Resetea toda la aplicación a su estado inicial"""
        if messagebox.askyesno("Confirmar",
                               "¿Estás seguro de que deseas resetear toda la aplicación? Se perderán todos los datos no guardados."):
            # Crear una nueva calculadora
            self.nn_calculator = NeuralNetworkCalculator()

            # Resetear la interfaz
            self.layer_frames = []
            self.update_layer_frames()

            # Limpiar campos de texto
            self.input_textbox.delete("1.0", "end")
            self.results_textbox.configure(state="normal")
            self.results_textbox.delete("1.0", "end")
            self.results_textbox.configure(state="disabled")

            # Volver a la pestaña de arquitectura
            self.tabview.set("Arquitectura")

            messagebox.showinfo("Aplicación Reseteada", "La aplicación ha sido reseteada a su estado inicial.")

            # Añadir esta nueva función a la clase App
    def setup_visualization_tab(self):
        """Configura la pestaña de visualización de la red"""
        visualization_frame = self.tabview.tab("Visualización")

        # Frame para controles
        controls_frame = ctk.CTkFrame(visualization_frame)
        controls_frame.pack(fill="x", padx=10, pady=10)

        # Botón para generar visualización
        visualize_btn = ctk.CTkButton(
            controls_frame,
            text="Visualizar Red Neuronal",
            command=self.visualize_network
        )
        visualize_btn.pack(side="left", padx=10, pady=10)

        # Opciones de visualización
        self.show_weights_var = tk.BooleanVar(value=True)
        show_weights_checkbox = ctk.CTkCheckBox(
            controls_frame,
            text="Mostrar pesos",
            variable=self.show_weights_var
        )
        show_weights_checkbox.pack(side="left", padx=20, pady=10)

        self.colored_nodes_var = tk.BooleanVar(value=True)
        colored_nodes_checkbox = ctk.CTkCheckBox(
            controls_frame,
            text="Colorear por activación",
            variable=self.colored_nodes_var
        )
        colored_nodes_checkbox.pack(side="left", padx=20, pady=10)

        # Frame para la figura de Matplotlib
        self.fig_frame = ctk.CTkFrame(visualization_frame)
        self.fig_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Inicializar figura de Matplotlib
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.ax = self.fig.add_subplot(111)

        # Canvas para Matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    # Añadir esta función para visualizar la red
    def visualize_network(self):
        """Crea y muestra una visualización de la arquitectura de la red"""
        try:
            # Verificar que la arquitectura esté configurada
            if not self.nn_calculator.layers_neurons:
                messagebox.showerror("Error", "La arquitectura de la red no está configurada.")
                return

            # Limpiar la figura
            self.ax.clear()

            # Crear un grafo dirigido con NetworkX
            G = nx.DiGraph()

            # Posiciones para dibujar las capas
            pos = {}
            layer_sizes = self.nn_calculator.layers_neurons
            max_neurons = max(layer_sizes)
            layer_spacing = 1.0

            # Color de nodos según función de activación
            activation_colors = {
                "Ninguna": "lightgray",
                "Sigmoide": "lightblue",
                "ReLU": "lightgreen",
                "Tanh": "lightyellow",
                "Lineal": "lightcoral",
                "Softmax": "orchid"
            }

            # Añadir nodos al grafo
            node_labels = {}
            node_colors = []

            for layer_idx, num_neurons in enumerate(layer_sizes):
                layer_name = "Entrada" if layer_idx == 0 else "Salida" if layer_idx == len(layer_sizes) - 1 else f"Oculta {layer_idx}"

                # Determinar el espacio vertical para los nodos
                vertical_spacing = max_neurons / (num_neurons + 1) if num_neurons > 1 else 0.5

                for neuron_idx in range(num_neurons):
                    # Calcular posición del nodo
                    x = layer_idx * layer_spacing
                    y = (neuron_idx + 1) * vertical_spacing

                    # Crear ID único para el nodo
                    node_id = f"{layer_idx}_{neuron_idx}"

                    # Añadir nodo al grafo
                    G.add_node(node_id)
                    pos[node_id] = (x, y)

                    # Etiqueta para el nodo (opcional, puede hacerlo más recargado)
                    if layer_idx == 0:
                        node_labels[node_id] = f"X{neuron_idx+1}"
                    elif layer_idx == len(layer_sizes) - 1:
                        node_labels[node_id] = f"Y{neuron_idx+1}"
                    else:
                        node_labels[node_id] = f"h{layer_idx}_{neuron_idx+1}"

                    # Color según función de activación
                    if self.colored_nodes_var.get():
                        # La primera capa no tiene activación asignada
                        activation = "Ninguna" if layer_idx == 0 else self.nn_calculator.activations[layer_idx]
                        node_colors.append(activation_colors.get(activation, "gray"))
                    else:
                        # Colores por capa (entrada, ocultas, salida)
                        if layer_idx == 0:
                            node_colors.append("lightblue")  # Entrada
                        elif layer_idx == len(layer_sizes) - 1:
                            node_colors.append("lightgreen")  # Salida
                        else:
                            node_colors.append("lightyellow")  # Ocultas

            # Añadir conexiones entre capas con pesos (si están disponibles)
            edge_weights = []
            for layer_idx in range(len(layer_sizes) - 1):
                for from_neuron in range(layer_sizes[layer_idx]):
                    for to_neuron in range(layer_sizes[layer_idx + 1]):
                        from_id = f"{layer_idx}_{from_neuron}"
                        to_id = f"{layer_idx+1}_{to_neuron}"

                        G.add_edge(from_id, to_id)

                        # Añadir peso como etiqueta si están disponibles y la opción está activada
                        if (self.show_weights_var.get() and
                            layer_idx < len(self.nn_calculator.weights) and
                            self.nn_calculator.weights[layer_idx] is not None):
                            try:
                                weight = self.nn_calculator.weights[layer_idx][to_neuron, from_neuron]
                                # Redondear para una visualización más limpia
                                weight_rounded = round(weight, 2)
                                edge_weights.append(abs(weight))  # Para grosor de línea

                                # El ancho de la conexión refleja la magnitud del peso
                                width = 0.5 + 2 * abs(weight) / max(abs(weight), 1)

                                # Color rojo para pesos negativos, azul para positivos
                                color = "red" if weight < 0 else "blue"

                                # Dibujar la conexión con el color y grosor adecuados
                                nx.draw_networkx_edges(
                                    G, pos,
                                    edgelist=[(from_id, to_id)],
                                    width=width,
                                    edge_color=color,
                                    alpha=0.7,
                                    ax=self.ax
                                )
                            except (IndexError, ValueError):
                                # Si hay algún error con los pesos, dibuja una conexión normal
                                edge_weights.append(1.0)
                        else:
                            edge_weights.append(1.0)

            # Dibujar el grafo (nodos y conexiones sin pesos)
            if not self.show_weights_var.get():
                nx.draw_networkx_edges(G, pos, ax=self.ax, alpha=0.6)

            # Dibujar los nodos
            nx.draw_networkx_nodes(
                G, pos,
                node_size=700,
                node_color=node_colors,
                edgecolors='black',
                alpha=0.8,
                ax=self.ax
            )

            # Añadir etiquetas a los nodos
            nx.draw_networkx_labels(
                G, pos,
                labels=node_labels,
                font_size=8,
                font_weight='bold',
                ax=self.ax
            )

            # Añadir etiquetas a las capas
            for i, size in enumerate(layer_sizes):
                if i == 0:
                    layer_name = "Entrada"
                elif i == len(layer_sizes) - 1:
                    layer_name = "Salida"
                else:
                    layer_name = f"Capa oculta {i}"

                # Colocar etiqueta en la parte superior de cada capa
                self.ax.text(
                    i * layer_spacing,
                    max_neurons + 0.5,
                    layer_name,
                    ha='center',
                    va='center',
                    fontsize=12,
                    fontweight='bold'
                )

                # Si hay función de activación, mostrarla para capas que no sean la de entrada
                if i > 0 and i < len(self.nn_calculator.activations):
                    activation_name = self.nn_calculator.activations[i]
                    self.ax.text(
                        i * layer_spacing,
                        -0.5,
                        f"Act: {activation_name}",
                        ha='center',
                        va='center',
                        fontsize=10,
                        color='blue'
                    )

            # Configuración estética de la figura
            self.ax.set_title("Arquitectura de la Red Neuronal", fontsize=14)
            self.ax.axis('off')  # Ocultar ejes

            # Agregar leyenda de activaciones si están coloreadas
            if self.colored_nodes_var.get():
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=act)
                    for act, color in activation_colors.items()
                    if act in self.nn_calculator.activations + ["Ninguna"]
                ]
                self.ax.legend(handles=legend_elements, loc='best')

            # Redibujar el canvas
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar la red: {str(e)}")
