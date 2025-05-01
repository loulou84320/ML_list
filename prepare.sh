#!/bin/bash
set -e

# Afficher un message stylisé
echo -e "\e[1;34m========================================\e[0m"
echo -e "\e[1;34m Configuration environnement ML avec GUI\e[0m"
echo -e "\e[1;34m========================================\e[0m"

# Vérifier si l'utilisateur a les droits sudo
if ! command -v sudo &> /dev/null; then
    echo -e "\e[1;31mLe script nécessite que sudo soit installé et disponible.\e[0m"
    exit 1
fi

# Vérifier si on est sur un système Linux
if [[ "$(uname)" != "Linux" ]]; then
    echo -e "\e[1;31mCe script est conçu pour fonctionner uniquement sur Linux.\e[0m"
    exit 1
fi

# Vérifier si l'OS est Ubuntu/Debian ou CentOS/RHEL
if command -v apt-get &> /dev/null; then
    PACKAGE_MANAGER="apt-get"
    INSTALL_CMD="sudo apt-get install -y"
    UPDATE_CMD="sudo apt-get update"
elif command -v yum &> /dev/null; then
    PACKAGE_MANAGER="yum"
    INSTALL_CMD="sudo yum install -y"
    UPDATE_CMD="sudo yum update -y"
else
    echo -e "\e[1;31mSystème d'exploitation non supporté. Ce script fonctionne avec Ubuntu/Debian ou CentOS/RHEL.\e[0m"
    exit 1
fi

echo -e "\e[1;32m[Info]\e[0m Utilisation de $PACKAGE_MANAGER comme gestionnaire de paquets"

# Mise à jour des paquets
echo -e "\e[1;32m[Étape 1/10]\e[0m Mise à jour des paquets système..."
eval $UPDATE_CMD

# Installation des paquets de base
echo -e "\e[1;32m[Étape 2/10]\e[0m Installation des paquets essentiels..."
if [[ "$PACKAGE_MANAGER" == "apt-get" ]]; then
    $INSTALL_CMD build-essential cmake unzip curl wget git python3-dev python3-pip
    # Packages pour Tkinter et autres dépendances graphiques
    $INSTALL_CMD python3-tk python3-pil python3-pil.imagetk
    $INSTALL_CMD libgl1-mesa-glx # Pour matplotlib
elif [[ "$PACKAGE_MANAGER" == "yum" ]]; then
    $INSTALL_CMD gcc gcc-c++ cmake unzip curl wget git python3-devel
    # Packages pour Tkinter et autres dépendances graphiques
    $INSTALL_CMD python3-tkinter
    $INSTALL_CMD mesa-libGL # Pour matplotlib
fi

# Vérification du GPU NVIDIA
echo -e "\e[1;32m[Étape 3/10]\e[0m Vérification du GPU NVIDIA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "\e[1;33m[Attention]\e[0m La commande nvidia-smi n'est pas disponible. Installation des drivers NVIDIA..."
    
    if [[ "$PACKAGE_MANAGER" == "apt-get" ]]; then
        $INSTALL_CMD ubuntu-drivers-common
        GPU_DRIVER=$(ubuntu-drivers devices | grep -o "nvidia-driver-[0-9]\+" | head -1)
        if [[ ! -z "$GPU_DRIVER" ]]; then
            $INSTALL_CMD $GPU_DRIVER
        else
            echo -e "\e[1;33m[Attention]\e[0m Aucun driver NVIDIA recommandé n'a été trouvé."
            $INSTALL_CMD nvidia-driver-525  # Version récente et stable
        fi
    elif [[ "$PACKAGE_MANAGER" == "yum" ]]; then
        $INSTALL_CMD epel-release
        $INSTALL_CMD kmod-nvidia
    fi
    
    echo -e "\e[1;33m[Attention]\e[0m Vous devrez peut-être redémarrer votre système après l'installation des drivers NVIDIA."
else
    echo -e "\e[1;32m[Info]\e[0m NVIDIA GPU détecté:"
    nvidia-smi
fi

# Installation de CUDA Toolkit
echo -e "\e[1;32m[Étape 4/10]\e[0m Vérification de CUDA Toolkit..."

if ! command -v nvcc &> /dev/null; then
    echo -e "\e[1;33m[Attention]\e[0m CUDA Toolkit n'est pas installé. Installation en cours..."
    
    if [[ "$PACKAGE_MANAGER" == "apt-get" ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        $INSTALL_CMD cuda-12-2
        echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    elif [[ "$PACKAGE_MANAGER" == "yum" ]]; then
        sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
        sudo dnf install -y cuda-12-2
        echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    
    echo -e "\e[1;33m[Action requise]\e[0m Source le fichier bashrc pour mettre à jour le PATH:"
    echo "source ~/.bashrc"
else
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "\e[1;32m[Info]\e[0m CUDA Toolkit $CUDA_VERSION est déjà installé"
fi

# Installation de cuDNN
echo -e "\e[1;32m[Étape 5/10]\e[0m Vérification/Installation de cuDNN..."

# Create a temporary directory for downloads
mkdir -p ~/cudnn_temp
cd ~/cudnn_temp

if [[ "$PACKAGE_MANAGER" == "apt-get" ]]; then
    echo -e "\e[1;33m[Info]\e[0m Installation de cuDNN via les paquets apt..."
    $INSTALL_CMD libcudnn8 libcudnn8-dev
elif [[ "$PACKAGE_MANAGER" == "yum" ]]; then
    echo -e "\e[1;33m[Info]\e[0m Installation de cuDNN via les paquets yum..."
    sudo dnf install -y libcudnn8 libcudnn8-devel
fi

echo -e "\e[1;32m[Info]\e[0m cuDNN installé ou existant"
cd ..
rm -rf ~/cudnn_temp

# Création d'un environnement virtuel pour Python
echo -e "\e[1;32m[Étape 6/10]\e[0m Configuration de l'environnement Python..."

# Installer pip si nécessaire
if ! command -v pip3 &> /dev/null; then
    echo -e "\e[1;33m[Info]\e[0m Installation de pip3..."
    if [[ "$PACKAGE_MANAGER" == "apt-get" ]]; then
        $INSTALL_CMD python3-pip
    elif [[ "$PACKAGE_MANAGER" == "yum" ]]; then
        $INSTALL_CMD python3-pip
    fi
fi

# Installer virtualenv
pip3 install --user virtualenv

# Créer l'environnement virtuel
VENV_NAME="ml_gpu_env"
python3 -m virtualenv ~/$VENV_NAME

echo -e "\e[1;32m[Info]\e[0m Environnement virtuel '$VENV_NAME' créé"
echo "Pour l'activer: source ~/$VENV_NAME/bin/activate"

# Installation des packages Python requis
echo -e "\e[1;32m[Étape 7/10]\e[0m Installation des packages Python de base..."

# Activer l'environnement virtuel pour l'installation
source ~/$VENV_NAME/bin/activate

# Installer les packages de base
pip install --upgrade pip
pip install numpy pandas scikit-learn

# Installation des packages de visualisation et GUI
echo -e "\e[1;32m[Étape 8/10]\e[0m Installation des packages pour GUI et visualisation..."

pip install matplotlib seaborn Pillow
# Vérifier si tk est correctement installé
python -c "import tkinter as tk; root = tk.Tk(); root.destroy()" || echo -e "\e[1;31mAttention: Tkinter semble ne pas fonctionner correctement\e[0m"

# Installer XGBoost avec support GPU
echo -e "\e[1;32m[Étape 9/10]\e[0m Installation de XGBoost avec support GPU..."
pip install xgboost

# Installer H2O
echo -e "\e[1;32m[Étape 10/10]\e[0m Installation de H2O framework..."
pip install h2o

# Vérifier les installations
echo -e "\e[1;32m[Vérification]\e[0m Test des installations..."

# Vérifier tous les imports requis
python -c "
modules = [
    'pandas', 'numpy', 'matplotlib.pyplot', 'seaborn', 'tkinter', 
    'matplotlib.backends.backend_tkagg', 'matplotlib.figure', 'h2o'
]
success = True
for module in modules:
    try:
        __import__(module)
        print(f'✓ {module} importé avec succès')
    except ImportError as e:
        print(f'✗ Erreur lors de l'importation de {module}: {e}')
        success = False

if success:
    print('Tous les modules nécessaires ont été importés avec succès!')
else:
    print('Certains modules n'ont pas pu être importés')
"

# Tester XGBoost avec GPU
python -c "
import xgboost as xgb
try:
    gpu_count = xgb.cuda.get_device_count()
    print(f'XGBoost détecte {gpu_count} GPU(s)')
    # Test rapide avec GPU
    data = xgb.DMatrix([[1, 2, 3], [4, 5, 6]], label=[0, 1])
    params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
    model = xgb.train(params, data, num_boost_round=1)
    print('Test GPU XGBoost réussi!')
except Exception as e:
    print(f'Erreur lors du test GPU XGBoost: {e}')
    print('XGBoost fonctionnera en mode CPU uniquement')
"

# Tester H2O
python -c "
import h2o
try:
    h2o.init(nthreads=-1, max_mem_size='1G')
    print('H2O initialisé avec succès')
    h2o.cluster().show_status()
    h2o.shutdown()
except Exception as e:
    print(f'Erreur lors du test H2O: {e}')
"

# Tester Tkinter avec Matplotlib
python -c "
try:
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    
    # Créer une fenêtre Tkinter de test
    root = tk.Tk()
    root.title('Test GUI')
    
    # Créer une figure Matplotlib
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3, 4, 5], [10, 1, 20, 3, 25])
    
    # Intégrer la figure dans Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    print('Test de Tkinter avec Matplotlib réussi!')
    
    # Fermer après 1 seconde
    root.after(1000, root.destroy)
    root.mainloop()
except Exception as e:
    print(f'Erreur lors du test Tkinter+Matplotlib: {e}')
"

# Désactiver l'environnement virtuel
deactivate

echo -e "\e[1;34m========================================\e[0m"
echo -e "\e[1;32m[TERMINÉ]\e[0m Configuration de l'environnement ML avec GUI complète!"
echo -e "\e[1;34m========================================\e[0m"
echo ""
echo -e "\e[1;33mPour utiliser cet environnement:\e[0m"
echo "1. Activez l'environnement virtuel: source ~/$VENV_NAME/bin/activate"
echo "2. Exécutez votre code Python avec interface graphique: python votre_script.py"
echo "3. Pour quitter l'environnement: deactivate"
echo ""
echo -e "\e[1;33mNotes importantes:\e[0m"
echo "- Si les drivers NVIDIA ont été installés, redémarrez le système avant utilisation"
echo "- Si vous rencontrez des problèmes d'affichage, assurez-vous d'être connecté avec un environnement graphique"
echo "- Pour utilisation à distance, configurez le transfert X11 (ssh -X) ou utilisez un serveur VNC"
echo ""
