import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from PIL import Image
import idx2numpy

datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas

datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

num_ej_entrenamiento = metadatos.splits["train"].num_examples
TAMANO_LOTE = 64

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)

print("Comenzando entrenamiento del modelo...")
historial = modelo.fit(datos_entrenamiento, epochs=20, steps_per_epoch=math.ceil(num_ej_entrenamiento / TAMANO_LOTE))
print("Entrenamiento del modelo completado.")

tf.keras.models.save_model(
    modelo,
    'modelo',
    overwrite=True,
    include_optimizer=True,
    save_format='tf'
)

class PredictionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print("Recibiendo solicitud POST...")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        print("Datos recibidos:", data)

        image_data = data.get('data', [])
        if not image_data:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Missing image data')
            return

        input_data = np.array(image_data).reshape((1, 28, 28, 1))
        print("Datos de entrada:", input_data)

        prediction = modelo.predict(input_data)
        print("Predicción antes de argmax:", prediction)
        class_index = np.argmax(prediction)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        class_name = class_names[class_index]
        print("Predicción:", class_name)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {'prediction': class_name}
        print("Respuesta enviada:", response)
        self.wfile.write(json.dumps(response).encode('utf-8'))


    def do_GET(self):
        if self.path == '/':
            try:
                with open('index.html', 'rb') as file:
                    content = file.read()
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content)
            except IOError:
                self.send_error(404, 'File Not Found: %s' % self.path)
        elif self.path == '/favicon.ico':
            self.send_error(404, 'File Not Found: %s' % self.path)
        else:
            self.send_error(501, 'Unsupported method (%s)' % self.command)


file_path = "C:\\Users\\Admin\\Desktop\\zalando\\fashion-mnist\\data\\fashion\\train-images-idx3-ubyte/train-images-idx3-ubyte"
images = idx2numpy.convert_from_file(file_path)

imagen_primera = images[10]
imagen_segunda = images[11]

ruta_guardado = "C:\\Users\\Admin\\Desktop\\zalando\\imagen_primera.png"
Image.fromarray(imagen_primera).resize((28, 28)).save(ruta_guardado)

ruta_guardado2 = "C:\\Users\\Admin\\Desktop\\zalando\\imagen_segunda.png"
Image.fromarray(imagen_segunda).resize((28, 28)).save(ruta_guardado2)

print("Las imágenes se han guardado correctamente en:", ruta_guardado, ruta_guardado2)

server_address = ('', 8000)
httpd = HTTPServer(server_address, PredictionHandler)
print('Servidor en ejecución en el puerto 8000...')
httpd.serve_forever()
