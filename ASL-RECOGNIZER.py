# Import libraries and setup keras environment (with jax)

import numpy as np
import os
import pandas as pd
import shutil
import random

os.environ["KERAS_BACKEND"] = "tensorflow"
# Keras must only be imported after the backend has been configured.
# We can also use "tensorflow" or "torch"
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model

# Read the CSV file into a pandas dataframe
df = pd.read_csv('/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/_train_classes.csv')
# df = pd.read_csv('/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/_valid_classes.csv')
# df = pd.read_csv('/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/_test_classes.csv')

# Initialize variables for maximum and minimum values
valMax = 0
valActu = 0
valMin = float('inf')
valTotal = 0

# Loop through each column in the dataframe, starting from the second column
for col in df.columns[1:]:
    # Print the name of the current column
    print(f"Column {col}:")
    # Get the maximum value count for the current column
    valActu = df[col].value_counts().max()
    # Print the maximum value count for the current column
    print(valActu)
    valTotal += valActu
    # Update the maximum and minimum values if necessary
    if valActu > valMax:
        valMax = valActu
    if valActu < valMin:
        valMin = valActu

# Print the maximum and minimum values
print("The maximum value found is:")
print(valMax)
print("The minimum value found is:")
print(valMin)
print("The sum of value found is:")
print(valTotal)

def count_files_in_subfolders(folder_path):
    try:
        # Créer un dictionnaire pour stocker le nombre de fichiers dans chaque sous-dossier
        files_count_per_subfolder = {}

        # Parcourir tous les sous-dossiers dans le dossier principal
        for subdir_name in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir_name)
            # Vérifier si l'élément est un dossier
            if os.path.isdir(subdir_path):
                # Compter le nombre de fichiers dans le sous-dossier
                files_count = len([name for name in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, name))])
                # Stocker le nombre de fichiers dans le dictionnaire
                files_count_per_subfolder[subdir_name] = files_count

        return files_count_per_subfolder
    except OSError:
        print("Erreur : Impossible d'accéder au dossier ou le dossier n'existe pas.")
        return {}

# Exemple d'utilisation
dossier_principal = '/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/train'
nombre_fichiers_par_sous_dossier = count_files_in_subfolders(dossier_principal)
print("Nombre de fichiers dans chaque sous-dossier :")
for subdir, count in nombre_fichiers_par_sous_dossier.items():
    print(f"- {subdir}: {count}")

    def equalize_files_in_subfolders(folder_path):
  try:
    files_count_per_subfolder = count_files_in_subfolders(folder_path)
    # Déterminer le nombre maximum de fichiers parmi tous les sous-dossiers
    max_files = max(files_count_per_subfolder.values())

    # Parcourir chaque sous-dossier
    for subdir_name, current_count in files_count_per_subfolder.items():
      subdir_path = os.path.join(folder_path, subdir_name)
      # Calculer combien de fichiers supplémentaires sont nécessaires
      files_to_add = max_files - current_count
      # Si des fichiers supplémentaires sont nécessaires
      if files_to_add > 0:
        # Liste de fichiers dans le sous-dossier
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        # Dupliquer des fichiers aléatoires pour atteindre le nombre maximum
        for i in range(files_to_add):
          # Sélectionner un fichier aléatoire
          file_to_copy = random.choice(files)
          # Chemin complet du fichier à copier
          src_file = os.path.join(subdir_path, file_to_copy)
          # Chemin où le fichier sera dupliqué
          dst_file = os.path.join(subdir_path, f"copy_{i+1}_{file_to_copy}")
          # Copier le fichier
          shutil.copy(src_file, dst_file)

    print("Duplication terminée avec succès.")
  except OSError as e:
    print(f"Erreur : {e}")

equalize_files_in_subfolders("/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/train")
equalize_files_in_subfolders("/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/valid")
nombre_fichiers_par_sous_dossier = count_files_in_subfolders("/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/train")
print("Nombre de fichiers dans chaque sous-dossier :")
for subdir, count in nombre_fichiers_par_sous_dossier.items():
    print(f"- {subdir}: {count}")

# Count the number of missing values in each column of the dataframe
missing_values = df.isnull().sum()

# Print the number of missing values for each column
print(missing_values)

# Réorganiser les images en sous-dossiers

# Chemins vers vos dossiers et fichiers CSV
base_dir = '/content/drive/MyDrive/ASL_Recognition_Data'
folders = ['train', 'valid', 'test']

for folder in folders:
    # Lire le fichier CSV
    df = pd.read_csv(os.path.join(base_dir, folder, '_classes.csv'))

    # Parcourir chaque ligne du DataFrame
    for index, row in df.iterrows():
        # Obtenir le nom du fichier et les classes auxquelles il appartient
        filename = row['filename']
        classes = row[row == 1].index.tolist()

        # Pour chaque classe, créer un sous-dossier (s'il n'existe pas déjà) et déplacer l'image
        for class_name in classes:
            class_dir = os.path.join(base_dir, folder, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.move(os.path.join(base_dir, folder, filename), os.path.join(class_dir, filename))

    # Renommer le fichier _classes.csv et le déplacer en dehors des dossiers de data
    newClassesFileName = "_"+folder+"_classes.csv"
    shutil.move(os.path.join(base_dir, folder, "_classes.csv"),os.path.join(base_dir,folder, newClassesFileName))
    shutil.move(os.path.join(base_dir, folder, newClassesFileName),os.path.join(base_dir, newClassesFileName))


# Configuration des paramètres
# Une règle empirique est de choisir une taille de batch qui est une puissance de deux (par exemple, 16, 32, 64, 128, 256, etc.).
# Choisissez également un nombre d’époques qui est un multiple de la taille du batch (par exemple, 2, 4, 8, 16, 32, etc.)

batch_size = 32 # 32
epochs = 2 # 8
img_width = 640
img_height = 640
image_c = 3

# Setup data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/valid',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Build sequential model
#model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, image_c)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(29, activation='softmax'))

# Initialize the model
model = Sequential()

# Add Convolutional layers
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(640, 640, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())

# Add Fully Connected layers
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=106, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

train_samples = train_generator.n
validation_samples = validation_generator.n


history = model.fit(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size)

model.save("/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/asl_model_simple.h5")

# Load the saved model
saved_model_path = "/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/asl_model.h5"
if os.path.exists(saved_model_path):
    model = load_model(saved_model_path)
    print("Model loaded successfully!")
else:
    # Build sequential model if the saved model doesn't exist
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, image_c)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(29, activation='softmax'))

# ... (Continue with the rest of the code)
train_samples = train_generator.n
validation_samples = validation_generator.n

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size)

# Save the model after training
model.save(saved_model_path)
print("Model saved successfully!")

# Test the trained model
test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/valid',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

saved_model_path = "/content/drive/MyDrive/ASL_Recognition_Data_Nath_Lulu/asl_model.h5"
if os.path.exists(saved_model_path):
    model = load_model(saved_model_path)
    print("Model loaded successfully!")

test_samples = test_generator.n
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_samples // batch_size)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
