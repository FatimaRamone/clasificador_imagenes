import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Obtener la ruta absoluta del directorio actual
current_directory = os.path.dirname(os.path.abspath(__file__))

# Directorio donde se guardarán las imágenes
save_directory = os.path.join(current_directory, 'images')

# Crear el directorio si no existe
os.makedirs(save_directory, exist_ok=True)

# Cargar el dataset Fashion MNIST
(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

# Guardar algunas imágenes en el directorio
num_images_to_save = 50
for i in range(num_images_to_save):
    label = train_labels[i]
    image = train_images[i]
    image_path = os.path.join(save_directory, f"{i}.png")
    plt.imsave(image_path, image, cmap='gray')
    print(f"Guardada imagen: {image_path}")
