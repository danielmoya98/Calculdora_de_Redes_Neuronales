import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any # Mantendremos las anotaciones de tipo

class NeuralNetworkCalculator:
    def __init__(self):
        self.layers_neurons = []  # Lista con número de neuronas por capa
        self.activations = []  # Lista con funciones de activación por capa
        self.weights = []  # Lista de matrices de pesos
        self.biases = []  # Lista de vectores de biases
        self.intermediate_results = {}  # Almacena resultados intermedios

    def set_architecture(self, layers_neurons: List[int], activations: List[str]) -> None:
        """
        Configura la arquitectura de la red neuronal.

        Args:
            layers_neurons: Lista con el número de neuronas por capa
            activations: Lista con las funciones de activación por capa
        """
        self.layers_neurons = layers_neurons
        self.activations = activations

        # Inicializar pesos y biases con valores vacíos
        self.weights = [None] * (len(layers_neurons) - 1)
        self.biases = [None] * (len(layers_neurons) - 1)

    def set_parameters(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        """
        Configura los pesos y biases de la red.

        Args:
            weights: Lista de matrices de pesos
            biases: Lista de vectores de biases
        """
        self.weights = weights
        self.biases = biases

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Función de activación Sigmoide"""
        # Evitar overflow en exp para valores muy negativos de z
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """Derivada de la función Sigmoide"""
        # Evitar overflow en exp para valores muy negativos de z
        z_clipped = np.clip(z, -500, 500)
        sigmoid_z = 1 / (1 + np.exp(-z_clipped))
        return sigmoid_z * (1 - sigmoid_z)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """Función de activación ReLU"""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivada de la función ReLU"""
        return np.where(z > 0, 1, 0)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """Función de activación Tanh"""
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        """Derivada de la función Tanh"""
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        """Función de activación Lineal (Identidad)"""
        return z

    @staticmethod
    def linear_derivative(z: np.ndarray) -> np.ndarray:
        """Derivada de la función Lineal"""
        return np.ones_like(z)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Función de activación Softmax"""
        # Estabilidad numérica: restar el máximo de z antes de exponenciar
        # Esto evita overflow con números grandes y no cambia el resultado de softmax
        if z.ndim == 1: # Caso vector
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        elif z.ndim == 2: # Caso matriz (batch)
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            raise ValueError("La entrada de Softmax debe ser 1D o 2D.")


    @staticmethod
    def softmax_derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivada de Softmax (aproximación para uso en dZ = dA * f'(Z)).
        La derivada completa de softmax es una matriz Jacobiana.
        Para el cálculo de dZ en backprop, a menudo se combina dL/dA * dA/dZ.
        Si la función de coste es Entropía Cruzada con Softmax, la derivada dL/dZ es A - Y.
        Esta función devuelve S * (1 - S) como una aproximación si se usa genéricamente,
        pero es importante ser consciente de su contexto en backpropagation.
        """
        s = NeuralNetworkCalculator.softmax(z)
        return s * (1 - s)

    def get_activation_function(self, name: str) -> callable:
        """
        Retorna la función de activación según su nombre.

        Args:
            name: Nombre de la función de activación

        Returns:
            Función de activación correspondiente
        """
        activation_functions = {
            "Sigmoide": self.sigmoid,
            "ReLU": self.relu,
            "Tanh": self.tanh,
            "Lineal": self.linear,
            "Softmax": self.softmax
        }
        return activation_functions.get(name, self.linear) # Default to linear if name not found

    def get_activation_derivative(self, name: str) -> callable:
        """
        Retorna la derivada de la función de activación según su nombre.

        Args:
            name: Nombre de la función de activación

        Returns:
            Derivada de la función de activación correspondiente
        """
        activation_derivatives = {
            "Sigmoide": self.sigmoid_derivative,
            "ReLU": self.relu_derivative,
            "Tanh": self.tanh_derivative,
            "Lineal": self.linear_derivative,
            "Softmax": self.softmax_derivative # Usar con precaución, ver nota en la función
        }
        return activation_derivatives.get(name, self.linear_derivative) # Default

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza la propagación hacia adelante en la red neuronal.

        Args:
            X: Vector de entrada (debe tener dimensiones compatibles con la capa de entrada)
               X debe ser una columna vector (n_features, 1)

        Returns:
            Salida de la red neuronal (columna vector)
        """
        # Reiniciar resultados intermedios
        self.intermediate_results = {}

        if not self.layers_neurons:
            raise ValueError("La arquitectura de la red no ha sido definida. Llama a set_architecture primero.")
        if not self.weights or not self.biases or self.weights[0] is None or self.biases[0] is None:
            # Podrías también inicializar con parámetros aleatorios aquí si es deseado
            raise ValueError("Los parámetros (pesos y biases) no han sido establecidos. Llama a set_parameters o inicialízalos.")

        # Asegurar que X sea un vector columna
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != self.layers_neurons[0]:
            raise ValueError(f"La entrada X debe tener {self.layers_neurons[0]} filas (features), pero tiene {X.shape[0]}")

        A = X
        self.intermediate_results['A0'] = A

        for l in range(len(self.weights)):
            Wl = self.weights[l]
            bl = self.biases[l]

            # Verificar dimensiones
            if Wl.shape[1] != A.shape[0]:
                raise ValueError(f"Incompatibilidad de dimensiones en la capa {l+1}: W{l+1} tiene {Wl.shape[1]} columnas, A{l} tiene {A.shape[0]} filas.")

            Z = np.dot(Wl, A) + bl
            self.intermediate_results[f'Z{l + 1}'] = Z

            activation_name = self.activations[l+1] # La capa 0 es la entrada, las activaciones aplican desde la capa 1
            activation_function = self.get_activation_function(activation_name)
            A = activation_function(Z)
            self.intermediate_results[f'A{l + 1}'] = A
        return A

    def to_json(self) -> Dict:
        """
        Convierte el modelo a un formato JSON serializable.

        Returns:
            Diccionario con la configuración del modelo
        """
        model_dict = {
            "layers_neurons": self.layers_neurons,
            "activations": self.activations,
            "weights": [w.tolist() if w is not None else None for w in self.weights],
            "biases": [b.tolist() if b is not None else None for b in self.biases]
        }
        return model_dict

    @classmethod
    def from_json(cls, model_dict: Dict) -> 'NeuralNetworkCalculator':
        """
        Crea una instancia de NeuralNetworkCalculator a partir de un diccionario.

        Args:
            model_dict: Diccionario con la configuración del modelo

        Returns:
            Instancia de NeuralNetworkCalculator
        """
        nn = cls()
        nn.layers_neurons = model_dict.get("layers_neurons", [])
        nn.activations = model_dict.get("activations", [])

        weights_data = model_dict.get("weights", [])
        biases_data = model_dict.get("biases", [])

        nn.weights = [np.array(w) if w is not None else None for w in weights_data]
        nn.biases = [np.array(b) if b is not None else None for b in biases_data]

        return nn