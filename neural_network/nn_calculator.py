import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any


class NeuralNetworkCalculator:
    def __init__(self):
        self.layers_neurons = []
        self.activations = []
        self.weights = []
        self.biases = []
        self.intermediate_results = {}  # Almacena Z y A de cada capa

    def set_architecture(self, layers_neurons: List[int], activations: List[str]) -> None:
        if len(layers_neurons) == 0:
            self.layers_neurons = []
            self.activations = []
            self.weights = []
            self.biases = []
            return

        if len(layers_neurons) != len(activations):
            raise ValueError("El número de capas en layers_neurons debe coincidir con el de activations.")

        self.layers_neurons = layers_neurons
        self.activations = activations  # activations[0] es para la capa de entrada (no se usa), activations[i] es para la capa i

        # Inicializar pesos y biases como None, se llenarán con set_parameters o initialize_random_parameters
        # Hay len(layers_neurons) - 1 conjuntos de pesos/biases
        self.weights = [None] * (len(layers_neurons) - 1)
        self.biases = [None] * (len(layers_neurons) - 1)

    def set_parameters(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        if len(weights) != len(self.layers_neurons) - 1 or len(biases) != len(self.layers_neurons) - 1:
            raise ValueError("La cantidad de matrices de pesos o biases no coincide con la arquitectura.")
        self.weights = weights
        self.biases = biases

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        s = NeuralNetworkCalculator.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1, 0)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        return z

    @staticmethod
    def linear_derivative(z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        if z.ndim == 1:
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        elif z.ndim == 2:
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            raise ValueError("La entrada de Softmax debe ser 1D o 2D.")

    @staticmethod
    def softmax_derivative(z: np.ndarray) -> np.ndarray:
        # Nota: Esta es una derivada puntual (elemento a elemento), s(z)(1-s(z)).
        # La derivada completa de softmax para backprop es más compleja (Jacobiana)
        # o se simplifica cuando se combina con la entropía cruzada (dL/dZ = A - Y).
        # Se incluye por completitud, pero para backprop con softmax, usar la combinada es mejor.
        s = NeuralNetworkCalculator.softmax(z)
        return s * (1 - s)

    def get_activation_function(self, name: str) -> callable:
        name_lower = name.lower()
        activation_functions = {
            "sigmoide": self.sigmoid, "sigmoid": self.sigmoid,
            "relu": self.relu,
            "tanh": self.tanh,
            "lineal": self.linear, "linear": self.linear,
            "softmax": self.softmax
        }
        return activation_functions.get(name_lower, self.linear)

    def get_activation_derivative(self, name: str) -> callable:
        name_lower = name.lower()
        activation_derivatives = {
            "sigmoide": self.sigmoid_derivative, "sigmoid": self.sigmoid_derivative,
            "relu": self.relu_derivative,
            "tanh": self.tanh_derivative,
            "lineal": self.linear_derivative, "linear": self.linear_derivative,
            "softmax": self.softmax_derivative  # Usar con precaución
        }
        return activation_derivatives.get(name_lower, self.linear_derivative)

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        self.intermediate_results = {}
        if not self.layers_neurons:
            raise ValueError("Arquitectura no definida. Llama a set_architecture.")
        if not self.weights or not self.biases or self.weights[0] is None or self.biases[0] is None:
            raise ValueError("Parámetros no establecidos. Llama a set_parameters o inicializa.")

        if X.ndim == 1: X = X.reshape(-1, 1)
        if X.shape[0] != self.layers_neurons[0]:
            raise ValueError(f"Entrada X con {X.shape[0]} filas, se esperan {self.layers_neurons[0]}.")

        A = X
        self.intermediate_results['A0'] = A  # A0 es la entrada X

        # L es el número de capas (incluyendo entrada y salida)
        # Hay L-1 conjuntos de pesos/biases, indexados de 0 a L-2
        # W[i] y b[i] son para la capa i+1 (activación i+1)
        for l in range(len(self.weights)):  # l va de 0 a L-2
            Wl = self.weights[l]
            bl = self.biases[l]

            if Wl.shape[1] != A.shape[0]:
                raise ValueError(f"Dimensión incompatible capa {l + 1}: W{l + 1} ({Wl.shape}) vs A{l} ({A.shape}).")

            Z = np.dot(Wl, A) + bl
            activation_name = self.activations[
                l + 1]  # activaciones[0] es capa entrada, activaciones[l+1] es para capa l+1
            activation_function = self.get_activation_function(activation_name)
            A_prev = A  # Guardamos A de la capa anterior para A{l}
            A = activation_function(Z)

            self.intermediate_results[f'Z{l + 1}'] = Z  # Z para la capa l+1
            self.intermediate_results[f'A{l + 1}'] = A  # A para la capa l+1
        return A  # Salida final de la red

    def _softmax_cross_entropy_cost(self, AL: np.ndarray, Y_true: np.ndarray) -> float:
        """Costo de entropía cruzada para salida Softmax."""
        m = Y_true.shape[1] if Y_true.ndim > 1 else 1
        epsilon = 1e-8  # Para evitar log(0)
        cost = -(1 / m) * np.sum(Y_true * np.log(AL + epsilon))
        return np.squeeze(cost)

    def _sigmoid_cross_entropy_cost(self, AL: np.ndarray, Y_true: np.ndarray) -> float:
        """Costo de entropía cruzada para salida Sigmoide (clasificación binaria)."""
        m = Y_true.shape[1] if Y_true.ndim > 1 else 1
        epsilon = 1e-8
        cost = -(1 / m) * np.sum(Y_true * np.log(AL + epsilon) + (1 - Y_true) * np.log(1 - AL + epsilon))
        return np.squeeze(cost)

    def calculate_cost(self, AL: np.ndarray, Y_true: np.ndarray, output_activation: str) -> float:
        """Calcula el costo basado en la activación de la capa de salida."""
        output_activation_lower = output_activation.lower()
        if output_activation_lower == "softmax":
            return self._softmax_cross_entropy_cost(AL, Y_true)
        elif output_activation_lower == "sigmoide" or output_activation_lower == "sigmoid":
            return self._sigmoid_cross_entropy_cost(AL, Y_true)
        else:
            # Para otras activaciones (ej. Lineal para regresión, podrías usar MSE)
            # Aquí, por simplicidad, si no es softmax/sigmoide, no calculamos un costo específico.
            # O podrías definir MSE: np.mean((AL - Y_true)**2)
            raise ValueError(
                f"Función de costo no implementada para '{output_activation}'. Soportado: Softmax, Sigmoide.")

    def _softmax_cross_entropy_cost_derivative_dZL(self, AL: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """Derivada dL/dZL para Softmax con Entropía Cruzada = AL - Y_true."""
        return AL - Y_true

    def backward_propagation(self, AL: np.ndarray, Y_true: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not self.intermediate_results:
            raise ValueError("Ejecuta forward_propagation primero.")
        if AL.shape != Y_true.shape:
            Y_true = Y_true.reshape(AL.shape)  # Intentar remodelar si es posible

        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        num_param_sets = len(self.weights)  # L-1, donde L es el número total de capas (incl. entrada)
        # Corresponde a índices 0 a L-2 para W, b

        m = AL.shape[1] if AL.ndim > 1 else 1  # Número de ejemplos

        # --- Capa de Salida (l = num_param_sets - 1 para W, b; capa L = num_param_sets para Z, A) ---
        # ZL es Z[num_param_sets], AL es A[num_param_sets]
        # W_L es weights[num_param_sets - 1], b_L es biases[num_param_sets - 1]
        # A_L-1 es A[num_param_sets - 1] (activación de la capa oculta anterior)

        idx_last_params = num_param_sets - 1  # Índice para weights, biases, grads_W, grads_b
        idx_last_layer_activation = num_param_sets  # Índice para Z, A en intermediate_results (Z1, A1, ..., ZL, AL)
        # self.activations[idx_last_layer_activation] es la activación de salida

        output_activation_name = self.activations[idx_last_layer_activation]

        if output_activation_name.lower() == "softmax":
            # Para Softmax con Entropía Cruzada, dL/dZL = AL - Y
            dZL = self._softmax_cross_entropy_cost_derivative_dZL(AL, Y_true)
        elif output_activation_name.lower() == "sigmoide" or output_activation_name.lower() == "sigmoid":
            # Para Sigmoide con Entropía Cruzada, dL/dAL = -(Y/AL - (1-Y)/(1-AL))
            # dZL = dL/dAL * g'(ZL)
            epsilon = 1e-8
            dAL = - (np.divide(Y_true, AL + epsilon) - np.divide(1 - Y_true, 1 - AL + epsilon))
            ZL = self.intermediate_results[f'Z{idx_last_layer_activation}']
            g_prime_ZL = self.get_activation_derivative(output_activation_name)(ZL)
            dZL = dAL * g_prime_ZL
        else:
            # Genérico: Asumir que dL/dAL es (AL - Y_true) como en MSE, o se necesita una forma más general
            # Para simplificar, si no es softmax/sigmoide, no se puede calcular dZL directamente sin más info.
            # Esto debería mejorarse si se soportan otras funciones de costo/activación de salida para backprop.
            # Por ahora, si es 'lineal', podríamos usar dL/dAL = AL - Y_true y g'(ZL) = 1
            if output_activation_name.lower() == "lineal" or output_activation_name.lower() == "linear":
                dAL = AL - Y_true  # Asumiendo costo MSE implícito
                ZL = self.intermediate_results[f'Z{idx_last_layer_activation}']
                g_prime_ZL = self.get_activation_derivative(output_activation_name)(ZL)
                dZL = dAL * g_prime_ZL
            else:
                raise ValueError(
                    f"Retropropagación no implementada para la función de activación de salida '{output_activation_name}' con el costo actual.")

        A_prev = self.intermediate_results[f'A{idx_last_layer_activation - 1}']  # A de la capa anterior

        grads_W[idx_last_params] = (1 / m) * np.dot(dZL, A_prev.T)
        grads_b[idx_last_params] = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
        dAPrev = np.dot(self.weights[idx_last_params].T, dZL)  # dL/dA para la capa anterior

        # --- Capas Ocultas (hacia atrás) ---
        # l va de num_param_sets - 1 (índice de capa de activación) hacia abajo hasta 1
        # param_idx (para W,b) va de num_param_sets - 2 hacia abajo hasta 0
        for l_act_idx in range(idx_last_layer_activation - 1, 0, -1):  # l_act_idx = L-1, L-2, ..., 1
            param_idx = l_act_idx - 1  # Índice para W, b, grads_W, grads_b

            Zl = self.intermediate_results[f'Z{l_act_idx}']
            activation_name_l = self.activations[l_act_idx]
            g_prime_Zl = self.get_activation_derivative(activation_name_l)(Zl)

            dZl = dAPrev * g_prime_Zl

            A_prev = self.intermediate_results[f'A{l_act_idx - 1}']  # A0 es la entrada X

            grads_W[param_idx] = (1 / m) * np.dot(dZl, A_prev.T)
            grads_b[param_idx] = (1 / m) * np.sum(dZl, axis=1, keepdims=True)

            if param_idx > 0 or l_act_idx > 1:  # No calcular dAPrev para la capa antes de la primera oculta si A0 es la entrada
                dAPrev = np.dot(self.weights[param_idx].T, dZl)

        return grads_W, grads_b

    def update_parameters(self, grads_W: List[np.ndarray], grads_b: List[np.ndarray], learning_rate: float) -> None:
        if len(grads_W) != len(self.weights) or len(grads_b) != len(self.biases):
            raise ValueError("Las listas de gradientes no coinciden con el número de parámetros.")

        for l in range(len(self.weights)):
            if self.weights[l] is not None and grads_W[l] is not None:
                self.weights[l] -= learning_rate * grads_W[l]
            if self.biases[l] is not None and grads_b[l] is not None:
                self.biases[l] -= learning_rate * grads_b[l]

    def to_json(self) -> Dict:
        model_dict = {
            "layers_neurons": self.layers_neurons,
            "activations": self.activations,
            "weights": [w.tolist() if w is not None else None for w in self.weights],
            "biases": [b.tolist() if b is not None else None for b in self.biases]
        }
        return model_dict

    @classmethod
    def from_json(cls, model_dict: Dict) -> 'NeuralNetworkCalculator':
        nn = cls()
        nn.layers_neurons = model_dict.get("layers_neurons", [])
        nn.activations = model_dict.get("activations", [])
        weights_data = model_dict.get("weights", [])
        biases_data = model_dict.get("biases", [])
        nn.weights = [np.array(w) if w is not None else None for w in weights_data]
        nn.biases = [np.array(b) if b is not None else None for b in biases_data]
        return nn