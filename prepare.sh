#!/bin/bash

# Mettre à jour la liste des paquets
sudo apt update

# Installer les dépendances Python
echo "Installation des dépendances Python..."
pip install --break-system-packages scikit-learn h2o seaborn matplotlib pandas

# Installer Tkinter (choisir l'une des deux options)
echo "Installation de Tkinter..."
# pip install tk (tkinter n'est pas disponible sur pip, il faut utiliser l'une des commandes suivantes)
sudo apt install -y python3-tk

# Installer JDK et les drivers Nvidia
echo "Installation de JDK et des drivers Nvidia..."
sudo apt install -y default-jdk
sudo apt-get -y install nvidia-cuda-toolkit
sudo apt install -y nvidia-driver

# Installer les dernières dépendances Python
echo "Installation des dernières dépendances Python..."
pip install --break-system-packages xgboost cuda

echo "Installation terminée !"
