import tensorflow_datasets as tfds

# Obtener la lista de nombres de constructores de conjuntos de datos
dataset_names = tfds.list_builders()

# Imprimir la lista de nombres de conjuntos de datos
print("Conjuntos de datos disponibles en TensorFlow Datasets:")
for dataset_name in dataset_names:
    print(dataset_name)
