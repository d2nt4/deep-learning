import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wget
import random
from tqdm import tqdm

# Descargar el dataset
# Descargamos el archivo de texto que utilizaremos como conjunto de datos (Don Quijote de la Mancha).
wget.download('https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/el_quijote.txt')

# Leer el texto
# Leemos el texto descargado y lo almacenamos en una variable.
with open("el_quijote.txt", "r", encoding='utf-8') as f:
    text = f.read()

# Tokenizador
# Definimos los caracteres válidos que se utilizarán para la tokenización
all_characters = string.printable + "ñÑáÁéÉíÍóÓúÚ¿¡"

#-----------------------------------------------------------------------------------#

# Clase Tokenizer
# Esta clase convierte texto en una secuencia de índices numéricos y viceversa.
class Tokenizer():
    def __init__(self):
        self.all_characters = all_characters # Conjunto de caracteres válidos
        self.n_characters = len(self.all_characters) # Número de caracteres válidos

    def text_to_seq(self, string):
        # Convierte texto en una secuencia de índices numéricos según su posición en `all_characters`.
        seq = []
        for c in range(len(string)):
            try:
                seq.append(self.all_characters.index(string[c]))
            except:
                continue
        return seq

    def seq_to_text(self, seq):
        # Convierte una secuencia de índices numéricos de vuelta a texto.
        text = ''
        for c in range(len(seq)):
            text += self.all_characters[seq[c]]
        return text

# Instanciamos el tokenizador
tokenizer = Tokenizer()

# Preparar datos
# Convertimos el texto en una secuencia de índices numéricos.
text_encoded = tokenizer.text_to_seq(text)

# Dividimos los datos en entrenamiento y prueba (80% y 20% respectivamente).
train_size = len(text_encoded) * 80 // 100
train = text_encoded[:train_size]
test = text_encoded[train_size:]

#-----------------------------------------------------------------------------------#

# Función para generar ventanas de texto
# Crea fragmentos de texto con longitud `window_size + 1` (último carácter como etiqueta).
def windows(text, window_size=100):
    start_index = 0
    end_index = len(text) - window_size
    text_windows = []
    while start_index < end_index:
        text_windows.append(text[start_index:start_index + window_size + 1])
        start_index += 1
    return text_windows

# Generamos ventanas para entrenamiento y validación
train_text_encoded_windows = windows(train)
test_text_encoded_windows = windows(test)

#-----------------------------------------------------------------------------------#

# Dataset personalizado
# Define cómo acceder a los datos en formato compatible con PyTorch.
class CharRNNDataset(Dataset):
    def __init__(self, text_encoded_windows, train=True):
        self.text = text_encoded_windows
        self.train = train

    def __len__(self):
        # Devuelve la cantidad de ejemplos en el dataset.
        return len(self.text)

    def __getitem__(self, ix):
        # Devuelve un ejemplo del dataset.
        if self.train:
            return torch.tensor(self.text[ix][:-1]), torch.tensor(self.text[ix][-1])  # Entrada y etiqueta
        return torch.tensor(self.text[ix]) # Solo entrada

# Creamos datasets para entrenamiento y validación
dataset = {
    'train': CharRNNDataset(train_text_encoded_windows),
    'val': CharRNNDataset(test_text_encoded_windows)
}

# Creamos dataloaders para iterar sobre los datos
dataloader = {
    'train': DataLoader(dataset['train'], batch_size=512, shuffle=True, pin_memory=True),
    'val': DataLoader(dataset['val'], batch_size=2048, shuffle=False, pin_memory=True),
}

#-----------------------------------------------------------------------------------#

# Modelo CharRNN
# Define la arquitectura de la red neuronal recurrente.
class CharRNN(nn.Module):
    def __init__(self, input_size, embedding_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.Embedding(input_size, embedding_size) # Capa de embedding
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True) # Capa LSTM
        self.fc = nn.Linear(hidden_size, input_size) #Capa totalmente conectada

    def forward(self, x):
        x = self.encoder(x) # Codificar la entrada
        x, h = self.rnn(x) # Pasar por la capa LSTM
        y = self.fc(x[:, -1, :]) #Predecir el siguiente carácter
        return y

#-----------------------------------------------------------------------------------#

# Entrenamiento
# Configuración del dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función de entrenamiento
def fit(model, dataloader, epochs=10):
    model.to(device) # Mover modelo a GPU si está disponible
    print(f"Modelo en dispositivo: {next(model.parameters()).device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Optimizador
    criterion = nn.CrossEntropyLoss() # Función de pérdida

    for epoch in range(1, epochs + 1):
        # Entrenamiento
        model.train()
        train_loss = []
        bar = tqdm(dataloader['train'])
        for batch in bar:
            X, y = batch
            X, y = X.to(device), y.to(device) # Mover datos a GPU si está disponible

            optimizer.zero_grad() # Reiniciar gradientes
            y_hat = model(X) # Predicción
            loss = criterion(y_hat, y) # Calcular pérdida
            loss.backward() # Retropropagación
            optimizer.step() # Actualizar pesos

            train_loss.append(loss.item())
            bar.set_description(f"loss {np.mean(train_loss):.5f}")

        # Validación
        val_loss = []
        model.eval()
        bar = tqdm(dataloader['val'])
        with torch.no_grad():
            for batch in bar:
                X, y = batch
                X, y = X.to(device), y.to(device) # Mover datos a GPU si está disponible
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                bar.set_description(f"val_loss {np.mean(val_loss):.5f}")
        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f}")

#-----------------------------------------------------------------------------------#

# Predicción
def predict(model, X):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X).to(device) # Mover datos a GPU si está disponible
        pred = model(X.unsqueeze(0)) # Agregar dimensión de batch
        return pred

# Instanciar y entrenar el modelo
model = CharRNN(input_size=tokenizer.n_characters)
fit(model, dataloader, epochs=10)

# Generación de texto
X_new = "En un lugar de la mancha, "
for i in range(1000):
    X_new_encoded = tokenizer.text_to_seq(X_new[-100:]) # Codificar los últimos 100 caracteres
    y_pred = predict(model, X_new_encoded) # Predecir el siguiente carácter
    y_pred = y_pred.view(-1).div(1).exp() # Distribución de probabilidad
    top_i = torch.multinomial(y_pred, 1)[0] # Seleccionar el carácter con mayor probabilidad
    predicted_char = tokenizer.all_characters[top_i]
    X_new += predicted_char

print(X_new)
