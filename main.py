# Importar la clase App desde el módulo app_gui dentro del paquete app
from app.app_gui import App
# Si necesitaras instanciar NeuralNetworkCalculator directamente aquí (aunque normalmente se haría dentro de App):
# from neural_network.nn_calculator import NeuralNetworkCalculator

def main():
    # Aquí asumo que la clase de tu interfaz gráfica se llama 'App'
    # y que está definida en app/app_gui.py
    app_instance = App()
    app_instance.mainloop()

if __name__ == "__main__":
    main()