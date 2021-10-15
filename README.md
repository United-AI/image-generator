# image-generator
---
Ein Projekt unseres Vereins zur Generierung von Bildern.
## To-do's
Projektname
Trainieren auf 120x120 Pixeln
Datensatz

## Tutorial zum Downloaden:
---

### Schritt 1: Path
---
Erstellen eines Ordners im gewünschten path

Mit der commandozeile in den Ordner gehen
```
cd /PATH_ORDNER
```

### Schritt 2: git clone
---
Clonen sie den Repository mit:
```
git clone https://github.com/United-AI/image-generator
```
Falls dein PC kein git hat:
https://git-scm.com/downloads

## Installation:
---

### Schritt 1: Python libraries
---
Stell sicher das python 3.9 auf deinen PC installiert ist
Falls nicht:
https://www.python.org/downloads/

Im Repository Ordner gibst du folgenden command ein:
```
pip install -r requirements.txt
```
(Ab hier optional für Grafikkarten nutzung. Nur für NVIDIA Grafikkarten mit CUDA Enabled) 

### Schritt 3: CUDA
---
Checken ob deine Grafikkarte CUDA Enabled ist: https://developer.nvidia.com/cuda-gpus
Downloaden der CUDA Software: https://developer.nvidia.com/cuda-toolkit-archive (11.2.2 empfohlen)

### Schritt 4: cuDNN
---
Ohne der cuDNN library, erkennt tensorflow deine Grafikkarte nicht
Downloaden der cuDNN library (Registrierung ins Developer Programm erforderlich): https://developer.nvidia.com/cudnn
Installation der cuDNN library: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

### Schritt 5: Checken ob tensorflow deine Grafikkarte erkennt
Öffnen von python:
```
python
```
Import tensorflow:
```
import tensorflow as tf
```
Type in command:
```
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
Falls der output 0 ist, erkennt tensorflow deine Grafikkarte nicht. Check deine CUDA und cuDNN installation.
---



## Letzter Schritt: Starten des Training vorgangs
```
python main.py
```
---


