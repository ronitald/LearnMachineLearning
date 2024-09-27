import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class OnlineTemperaturePredictor:
    def __init__(self):
        # Inicializar el modelo y el escalador
        self.scaler = StandardScaler()
        self.model = SGDRegressor(max_iter=1, tol=None, warm_start=True)
        self.is_fitted = False

    def fit_initial_data(self, X, y):
        # Escalar los datos iniciales
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar el modelo con los datos iniciales
        self.model.partial_fit(X_scaled, y)
        self.is_fitted = True

    def update_model(self, X_new, y_new):
        # Escalar los nuevos datos utilizando el mismo escalador
        X_new_scaled = self.scaler.transform(X_new)
        
        # Actualizar el modelo con los nuevos datos
        self.model.partial_fit(X_new_scaled, y_new)

    def predict_temperature(self, time_in_minutes):
        # Predecir la temperatura para un nuevo tiempo
        time_scaled = self.scaler.transform([[time_in_minutes]])
        prediction = self.model.predict(time_scaled)
        return prediction[0]

# Función principal para simular el aprendizaje en línea
def simulate_online_learning():
    # Datos iniciales: tiempo en minutos y temperatura en grados Celsius
    initial_time = np.array([[0], [10], [20], [30], [40]])
    initial_temperature = np.array([20, 22, 24, 23, 25])

    # Crear una instancia del predictor de temperatura en línea
    predictor = OnlineTemperaturePredictor()

    # Entrenar el modelo con los datos iniciales
    predictor.fit_initial_data(initial_time, initial_temperature)

    # Nuevos datos que llegan secuencialmente
    new_time_data = np.array([[50], [60], [70]])
    new_temperature_data = np.array([26, 27, 28])

    # Simular el proceso de aprendizaje en línea
    for time, temp in zip(new_time_data, new_temperature_data):
        predictor.update_model(np.array([[time[0]]]), np.array([temp]))
        print(f"Modelo actualizado con tiempo={time[0]} y temperatura={temp}")

    # Hacer una predicción para un nuevo tiempo
    predicted_temp = predictor.predict_temperature(80)
    print(f"Predicción de temperatura para 80 minutos: {predicted_temp:.2f}°C")

# Ejecutar la simulación
simulate_online_learning()
