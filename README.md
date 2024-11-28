Fashion MNIST Model Prediction API

Este proyecto utiliza el conjunto de datos Fashion MNIST para entrenar un modelo de red neuronal convolucional (CNN) usando TensorFlow y lo expone como una API a través de un servidor HTTP utilizando BaseHTTPRequestHandler de Python.
Descripción

El modelo se entrena sobre el conjunto de datos Fashion MNIST, que consiste en imágenes de prendas de ropa, y predice la clase de la prenda a partir de una imagen enviada a través de una solicitud POST.
Características

    Modelo de red neuronal convolucional (CNN) entrenado con TensorFlow.
    API para realizar predicciones de clase en imágenes de ropa.
    Soporta solicitudes POST con imágenes en formato JSON.
    Servidor HTTP que devuelve la clase de la prenda predicha.

Requisitos

Para ejecutar este proyecto, necesitas tener instalado:

    Python 3.x
    TensorFlow
    TensorFlow Datasets
    NumPy
    PIL (Pillow)
    HTTPServer
    idx2numpy

Puedes instalar las dependencias necesarias con:

pip install tensorflow tensorflow-datasets numpy pillow idx2numpy

Uso
Entrenamiento del Modelo

El código entrena el modelo utilizando el conjunto de datos Fashion MNIST. Los datos de entrenamiento se cargan, se normalizan y se entrenan durante 20 épocas. El modelo se guarda en el directorio modelo al finalizar el entrenamiento.
Servir la API

El servidor HTTP expone un endpoint para predecir la clase de una imagen de ropa a través de una solicitud POST. Envíe una imagen de 28x28 píxeles en formato JSON para obtener la predicción.
Endpoint para Predicción (POST)

    URL: /predict

    Método: POST

    Entrada: JSON con una matriz de píxeles de una imagen de 28x28 (escala de grises).

{
  "data": [[0, 0, 0, ..., 0, 0, 0], ...]
}

Respuesta: JSON con la clase predicha.

    {
      "prediction": "T-shirt/top"
    }

Endpoint de Inicio (GET)

    URL: /
    Método: GET
    Descripción: Servirá la página index.html.

Ejecutar el Servidor

Para iniciar el servidor, ejecuta el siguiente comando:

python app.py

Esto iniciará el servidor en el puerto 8000. El servidor estará disponible en http://localhost:8000.
Guardar y Cargar Imágenes

El proyecto también incluye una función para cargar imágenes de un archivo de tipo .ubyte (formato del conjunto de datos MNIST) y guardarlas en formato PNG. Puedes modificar la ruta a tus archivos en el código para guardar las imágenes deseadas.
Estructura del Proyecto

/fashion-mnist-model-api
│
├── app.py                   # Código principal para el modelo y servidor
├── modelo/                  # Directorio donde se guarda el modelo entrenado
├── index.html               # Página HTML de inicio (si se solicita por GET)
└── requirements.txt         # Dependencias del proyecto

Contribuciones

Si deseas contribuir, por favor haz un fork del repositorio y envía un pull request.
Licencia

Este proyecto está bajo la Licencia MIT.
