import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
import os
import time
import threading
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import traceback
import warnings
warnings.filterwarnings('ignore')

class PredicteurListesBackend:
    def __init__(self, max_num=70, listes_par_ligne=20, log_callback=None):
        self.max_num = max_num
        self.listes_par_ligne = listes_par_ligne
        self.model_dir = 'modeles_listes'
        self.feature_dir = 'features'
        self.stats_dir = 'statistiques'
        self.h2o_initialized = False
        self.log_callback = log_callback
        
        # Création des répertoires nécessaires
        for dir_path in [self.model_dir, self.feature_dir, self.stats_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
    def log(self, message):
        """Enregistre un message dans la console et appelle le callback de log s'il existe"""
        print(message)
        if self.log_callback:
            self.log_callback(message)
    
    def initialiser_h2o(self, memory="4G", progress_callback=None):
        """Initialise H2O avec les paramètres spécifiés"""
        if self.h2o_initialized:
            self.log("H2O déjà initialisé")
            return True
        
        try:
            self.log("Initialisation de H2O en cours...")
            if progress_callback:
                progress_callback(10)
            
            h2o.init(nthreads=-1,  # Utilise tous les cœurs CPU disponibles
                    max_mem_size=memory,  
                    enable_assertions=False)
                    
            if progress_callback:
                progress_callback(50)
            
            # Vérifier si XGBoost est disponible
            try:
                from h2o.estimators.xgboost import H2OXGBoostEstimator
                self.log("XGBoost est disponible dans H2O")
                
                # Configurer XGBoost pour GPU lors de la création de modèles
                self.xgb_params = {
                    "tree_method": "gpu_hist",
                    "gpu_id": 0,
                    "predictor": "gpu_predictor"
                }
            except ImportError:
                self.log("XGBoost n'est pas disponible dans H2O")
                self.xgb_params = None
            
            self.log("H2O initialisé. Vérification du support GPU...")
            if progress_callback:
                progress_callback(70)
                
            self.verifier_support_gpu(progress_callback)
            
            self.h2o_initialized = True
            if progress_callback:
                progress_callback(100)
                
            return True
            
        except Exception as e:
            self.log(f"Erreur lors de l'initialisation de H2O: {e}")
            if progress_callback:
                progress_callback(0)
            return False

    def verifier_support_gpu(self, progress_callback=None):
        """Vérifie si le GPU est disponible pour H2O"""
        try:
            # Version adaptée pour H2O 3.38.0.1
            self.log(f"Version H2O: {h2o.__version__}")
            if progress_callback:
                progress_callback(75)
            
            # Vérifier si cluster().nodes existe et est itérable
            try:
                nodes = h2o.cluster().nodes
                if nodes is not None:
                    self.log(f"Nombre de nœuds: {len(nodes)}")
                else:
                    self.log("Information sur les nœuds non disponible")
            except:
                self.log("Impossible d'obtenir des informations sur les nœuds du cluster")
            
            # Utiliser une approche plus sûre pour cluster_status
            try:
                cluster_info = h2o.cluster_status()
                if cluster_info and isinstance(cluster_info, dict):
                    if 'mem_value_size' in cluster_info:
                        self.log(f"Mémoire totale: {cluster_info['mem_value_size']}")
                    else:
                        self.log("Info mémoire non disponible dans cluster_status")
                else:
                    self.log("cluster_status n'a pas retourné un dictionnaire valide")
            except Exception as e:
                self.log(f"Impossible d'obtenir le statut du cluster: {e}")
            
            if progress_callback:
                progress_callback(85)
            
            # Vérifier si XGBoost est disponible avec GPU
            try:
                import xgboost as xgb
                gpu_count = 0
                
                try:
                    # Tenter de détecter les GPUs disponibles
                    gpu_count = xgb.cuda.get_device_count()
                    self.log(f"GPUs détectés via xgb.cuda: {gpu_count}")
                except:
                    try:
                        # Alternative pour les anciennes versions ou autre méthode
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'], 
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        if result.returncode == 0 and result.stdout.strip():
                            gpu_count = int(result.stdout.strip())
                            self.log(f"GPUs détectés via nvidia-smi: {gpu_count}")
                    except:
                        # Dernière tentative
                        try:
                            from xgboost.core import EarlyStopException
                            from xgboost import gpu_predictor
                            self.log("Modules GPU XGBoost importés avec succès")
                            gpu_count = 1
                        except Exception as e:
                            self.log(f"Échec de l'importation des modules GPU XGBoost: {e}")
                            gpu_count = 0
                
                if progress_callback:
                    progress_callback(90)
                
                if gpu_count > 0:
                    self.log("Support GPU détecté pour XGBoost!")
                    
                    # Test avec configuration de repli
                    try:
                        # Créer un petit dataset
                        X = np.random.rand(100, 10)
                        y = np.random.randint(0, 2, 100)
                        
                        # Essayer d'abord avec GPU
                        try:
                            params_gpu = {
                                'tree_method': 'gpu_hist',
                                'gpu_id': 0,
                                'predictor': 'gpu_predictor'
                            }
                            
                            dtrain = xgb.DMatrix(X, label=y)
                            bst_gpu = xgb.train(params_gpu, dtrain, num_boost_round=2)
                            self.log("Test XGBoost GPU réussi!")
                        except Exception as gpu_error:
                            self.log(f"Test GPU a échoué: {gpu_error}")
                            self.log("Tentative de repli sur CPU...")
                            
                            params_cpu = {
                                'tree_method': 'hist',
                                'predictor': 'cpu_predictor'
                            }
                            
                            dtrain = xgb.DMatrix(X, label=y)
                            bst_cpu = xgb.train(params_cpu, dtrain, num_boost_round=2)
                            self.log("Repli sur CPU réussi")
                            self.xgb_params = None
                            
                    except Exception as e:
                        self.log(f"Erreur lors des tests XGBoost: {e}")
                        self.xgb_params = None
                else:
                    self.log("Aucun GPU détecté pour XGBoost. Utilisation du CPU.")
                    self.xgb_params = None
            except ImportError:
                self.log("XGBoost n'est pas disponible, utilisation CPU uniquement")
                self.xgb_params = None
                
            if progress_callback:
                progress_callback(95)
                
        except Exception as e:
            self.log(f"Erreur lors de la vérification du GPU: {e}")
            traceback.print_exc()
            self.xgb_params = None

    def get_leaderboard(self, aml):
        """Récupère le tableau de leaderboard des modèles entraînés
        
        Args:
            aml: Modèle H2OAutoML entraîné
            
        Returns:
            DataFrame: Tableau des performances des modèles
        """
        try:
            if aml is None:
                self.log("Aucun modèle AutoML fourni pour récupérer le leaderboard")
                return pd.DataFrame()
                
            # Récupération du leaderboard à partir de l'objet AutoML
            lb = aml.leaderboard
            
            # Conversion en pandas DataFrame pour faciliter l'affichage
            lb_df = lb.as_data_frame()
            
            self.log(f"Leaderboard récupéré avec {len(lb_df)} modèles")
            return lb_df
        except Exception as e:
            self.log(f"Erreur lors de la récupération du leaderboard: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def predire_listes(self, model, data, n_predictions=5, feature_cols=None, num_max=None):
        """Prédit les prochaines listes en utilisant un modèle entraîné
        
        Args:
            model: Modèle AutoML entraîné (H2OAutoML ou son leader)
            data: Données d'entrée (DataFrame)
            n_predictions: Nombre de prédictions à faire
            feature_cols: Colonnes à utiliser pour les prédictions
            num_max: Valeur maximale pour les nombres prédits (par défaut self.max_num)
            
        Returns:
            DataFrame: Prédictions générées
        """
        try:
            if not self.h2o_initialized:
                self.initialiser_h2o()
                
            self.log("Prédiction des prochaines listes...")
            
            if model is None:
                raise ValueError("Aucun modèle fourni pour les prédictions")
                
            # Si c'est un modèle AutoML, utiliser son leader
            if hasattr(model, 'leader'):
                best_model = model.leader
                self.log(f"Utilisation du modèle leader: {best_model.model_id}")
            else:
                best_model = model
                
            # Déterminer la valeur maximale pour les numéros
            if num_max is None:
                num_max = self.max_num
                
            # Préparer les dernières données pour la prédiction
            latest_data = data.tail(n_predictions).copy()
            
            # Liste pour stocker les prédictions
            predictions = []
            
            # Préparer les données de prediction
            for i in range(n_predictions):
                # Créer le vecteur de features à partir des données les plus récentes
                if feature_cols:
                    pred_data = latest_data[feature_cols].iloc[-1:]
                else:
                    # Si aucune colonne spécifiée, utiliser toutes les colonnes numériques
                    pred_data = latest_data.select_dtypes(include=[np.number]).iloc[-1:]
                
                # Convertir en H2OFrame
                h2o_pred_data = h2o.H2OFrame(pred_data)
                
                # Faire la prédiction
                pred = best_model.predict(h2o_pred_data)
                pred_values = pred.as_data_frame()
                
                # Traitement des résultats selon le type de problème
                if 'predict' in pred_values.columns:
                    # Régression ou classification binaire
                    predicted_value = pred_values['predict'].iloc[0]
                    if isinstance(predicted_value, (int, float)):
                        predicted_value = min(max(round(predicted_value), 1), num_max)
                elif len(pred_values.columns) > 0:
                    # Classification multi-classes ou autre format
                    # Trouver les indices des plus grandes probabilités
                    probas = pred_values.iloc[0].values
                    top_indices = np.argsort(probas)[-5:][::-1]  # Les 5 meilleurs en ordre décroissant
                    predicted_value = [int(idx + 1) for idx in top_indices if idx < num_max]
                else:
                    # Fallback si le format de prédiction n'est pas reconnu
                    self.log("Format de prédiction non reconnu")
                    predicted_value = np.random.randint(1, num_max + 1, 5)
                
                # Créer une nouvelle ligne avec la prédiction
                new_pred = {
                    'prediction': predicted_value,
                    'timestamp': pd.Timestamp.now()
                }
                predictions.append(new_pred)
                
                # Mettre à jour les données pour la prochaine prédiction
                # Dans un cas réel, vous pourriez attendre de nouvelles données au lieu de prédire à partir de prédictions
                
            # Créer un DataFrame avec les prédictions
            pred_df = pd.DataFrame(predictions)
            self.log(f"Généré {len(predictions)} prédictions")
            
            return pred_df
        
        except Exception as e:
            self.log(f"Erreur lors de la prédiction: {e}")
            traceback.print_exc()
            return pd.DataFrame()


    def charger_donnees(self, filepath, progress_callback=None):
        """Charge les données à partir d'un fichier CSV et les prépare pour l'analyse"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
            
            self.log(f"Chargement des données depuis {filepath}...")
            if progress_callback:
                progress_callback(10)
                
            # Déterminer le type de fichier CSV (en-têtes ou pas, séparateurs)
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                
            # Déterminer le séparateur
            delimiter = ',' if ',' in first_line else ';'
            
            # Déterminer si le fichier a des en-têtes ou non
            has_header = False
            try:
                # Si première ligne est entièrement numérique, alors pas d'en-tête
                parts = first_line.split(delimiter)
                [float(part) for part in parts]  # Tente de convertir en nombres
            except ValueError:
                has_header = True
                
            # Lire le fichier avec les paramètres déterminés
            if has_header:
                df = pd.read_csv(filepath, delimiter=delimiter)
            else:
                # Créer des noms de colonnes génériques
                n_cols = len(first_line.split(delimiter))
                col_names = [f"num_{i+1}" for i in range(n_cols)]
                df = pd.read_csv(filepath, delimiter=delimiter, names=col_names, header=None)
                
            if progress_callback:
                progress_callback(50)
                
            # Nettoyer les données - s'assurer que toutes les colonnes sont numériques
            df_cleaned = df.copy()
            
            # Détecter les colonnes contenant des donées
            number_columns = []
            for col in df_cleaned.columns:
                try:
                    # Convertir en numérique, les non-numériques deviennent NaN
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    
                    # Si la colonne contient principalement des nombres entre 1 et max_num, c'est un numéro de liste
                    valid_numbers = df_cleaned[(df_cleaned[col] >= 1) & (df_cleaned[col] <= self.max_num)][col].count()
                    if valid_numbers / len(df_cleaned) > 0.5:  # Si plus de 50% sont valides
                        number_columns.append(col)
                except:
                    continue
                    
            if not number_columns:
                raise ValueError("Aucune colonne valide de numéros de listes trouvée dans le fichier")
                
            # Sélectionner uniquement les colonnes contenant des numéros
            df_cleaned = df_cleaned[number_columns]
            
            # Supprimer les lignes avec des NaN
            df_cleaned.dropna(inplace=True)
            
            self.log(f"Données chargées avec succès. {len(df_cleaned)} donées et {len(number_columns)} numéros par liste.")
            
            if progress_callback:
                progress_callback(100)
                
            return df_cleaned
            
        except Exception as e:
            self.log(f"Erreur lors du chargement des données: {e}")
            traceback.print_exc()
            if progress_callback:
                progress_callback(0)
            return None

    def methode_statistique(self, data, n_prediction=5, progress_callback=None):
        """Analyse statistique des données pour prédire les prochaines listes"""
        try:
            if data is None or len(data) == 0:
                raise ValueError("Aucune donnée valide fournie pour l'analyse statistique")
                
            self.log("Analyse statistique des listes précédentes...")
            if progress_callback:
                progress_callback(10)
                
            # Convertir le DataFrame en un tableau de numéros
            all_numbers = []
            for _, row in data.iterrows():
                all_numbers.extend(row.values)
                
            # Compter les occurrences de chaque numéro
            counter = Counter(all_numbers)
            
            # Calculer les fréquences
            total_draws = len(data)
            total_numbers = len(all_numbers)
            
            stats = []
            for num in range(1, self.max_num + 1):
                count = counter.get(num, 0)
                frequency = (count / total_numbers) * 100
                stats.append({'number': num, 'count': count, 'frequency': frequency})
                
            stats_df = pd.DataFrame(stats)
            
            if progress_callback:
                progress_callback(50)
                
            # Analyse des tendances récentes (derniers 10% des listes)
            recent_data = data.iloc[int(len(data) * 0.9):]
            recent_numbers = []
            for _, row in recent_data.iterrows():
                recent_numbers.extend(row.values)
                
            recent_counter = Counter(recent_numbers)
            
            # Ajouter les tendances récentes au DataFrame de statistiques
            for i, num in enumerate(range(1, self.max_num + 1)):
                recent_count = recent_counter.get(num, 0)
                recent_total = len(recent_numbers)
                recent_frequency = (recent_count / recent_total) * 100 if recent_total > 0 else 0
                stats_df.at[i, 'recent_count'] = recent_count
                stats_df.at[i, 'recent_frequency'] = recent_frequency
                stats_df.at[i, 'trend'] = recent_frequency - stats_df.at[i, 'frequency']
            
            # Sauvegarder les statistiques
            stats_file = os.path.join(self.stats_dir, f'stats_{time.strftime("%Y%m%d_%H%M%S")}.csv')
            stats_df.to_csv(stats_file, index=False)
            self.log(f"Statistiques sauvegardées dans {stats_file}")
            
            if progress_callback:
                progress_callback(80)
                
            # Prédire les prochains numéros
            # Stratégie: combiner les numéros fréquents, ceux avec une tendance à la hausse et quelques aléatoires
            
            # 60% basés sur la fréquence
            most_common = stats_df.nlargest(int(n_prediction * 0.6), 'frequency')['number'].tolist()
            
            # 30% basés sur la tendance à la hausse
            trending_up = stats_df[stats_df['trend'] > 0].nlargest(int(n_prediction * 0.3), 'trend')['number'].tolist()
            
            # 10% aléatoires
            remaining = list(set(range(1, self.max_num + 1)) - set(most_common) - set(trending_up))
            random_picks = np.random.choice(remaining, size=min(int(n_prediction * 0.1) + 1, len(remaining)), replace=False).tolist()
            
            # Combiner et s'assurer que nous avons n_prediction numéros uniques
            prediction = most_common + trending_up + random_picks
            prediction = list(dict.fromkeys(prediction))[:n_prediction]  # Eliminer les doublons
            
            # S'assurer que nous avons exactement n_prediction numéros
            while len(prediction) < n_prediction:
                remaining_nums = list(set(range(1, self.max_num + 1)) - set(prediction))
                prediction.append(np.random.choice(remaining_nums))
                
            if progress_callback:
                progress_callback(100)
                
            self.log(f"Prédiction basée sur les statistiques: {sorted(prediction)}")
            return stats_df, sorted(prediction)
            
        except Exception as e:
            self.log(f"Erreur lors de l'analyse statistique: {e}")
            traceback.print_exc()
            if progress_callback:
                progress_callback(0)
            return None, []

    def preparer_donnees_ml(self, data, horizons=[1, 3, 5], progress_callback=None):
        """Prépare les données pour l'apprentissage automatique en créant des features temporelles"""
        if not self.h2o_initialized:
            self.log("H2O n'est pas initialisé. Initialisation en cours...")
            self.initialiser_h2o()
            
        try:
            self.log("Préparation des données pour le machine learning...")
            if progress_callback:
                progress_callback(10)
                
            # Vérifier si les données sont organisées sous forme de séries chronologiques
            if len(data) < max(horizons) + 1:
                raise ValueError(f"Pas assez de données pour les horizons spécifiés: {horizons}")
                
            # Convertir le DataFrame en une liste de listes de numéros
            all_draws = []
            for _, row in data.iterrows():
                draw = sorted([int(num) for num in row.values if not np.isnan(num)])
                all_draws.append(draw)

            if progress_callback:
                progress_callback(20)
                
            # Créer des features basées sur les numéros précédents
            X = []
            y = []
            
            for horizon in horizons:
                for i in range(len(all_draws) - horizon):
                    feature_vector = []
                    
                    # Utiliser plusieurs listes précédents comme features
                    for j in range(horizon):
                        prev_draw = all_draws[i + j]
                        # Encoder les liste précédent (par exemple, one-hot encoding)
                        for num in range(1, self.max_num + 1):
                            feature_vector.append(1 if num in prev_draw else 0)
                    
                    X.append(feature_vector)
                    y.append(all_draws[i + horizon])

            if progress_callback:
                progress_callback(50)
                
            # Convertir en DataFrame pour H2O
            feature_cols = [f'prev_{j+1}_num_{num}' for j in range(horizon) for num in range(1, self.max_num + 1)]
            X_df = pd.DataFrame(X, columns=feature_cols)
            
            # Encoder les cibles
            y_encoded = []
            for draw in y:
                target = [0] * self.max_num
                for num in draw:
                    if 1 <= num <= self.max_num:
                        target[num-1] = 1
                y_encoded.append(target)
                
            target_cols = [f'target_num_{i+1}' for i in range(self.max_num)]
            y_df = pd.DataFrame(y_encoded, columns=target_cols)
            
            # Combiner X et y
            ml_data = pd.concat([X_df, y_df], axis=1)
            
            if progress_callback:
                progress_callback(80)
                
            # Convertir en H2O frame
            h2o_data = h2o.H2OFrame(ml_data)
            
            # Sauvegarder les features pour réutilisation
            feature_file = os.path.join(self.feature_dir, f'features_{time.strftime("%Y%m%d_%H%M%S")}.csv')
            ml_data.to_csv(feature_file, index=False)
            self.log(f"Features sauvegardées dans {feature_file}")
            
            if progress_callback:
                progress_callback(100)
                
            return h2o_data, feature_cols, target_cols
            
        except Exception as e:
            self.log(f"Erreur lors de la préparation des données pour ML: {e}")
            traceback.print_exc()
            if progress_callback:
                progress_callback(0)
            return None, [], []

    def entrainer_modele(self, data, feature_cols=None, target_cols=None, 
                         max_models=None, max_runtime_secs=None, progress_callback=None):
        """Entraîne un modèle AutoML pour prédire les numéros"""
        try:
            self.log(f"DEBUG: Type de données reçu: {type(data)}")
            
            # Si c'est un tuple, on essaie de le convertir de façon appropriée
            if isinstance(data, tuple):
                self.log(f"DEBUG: Longueur du tuple: {len(data)}")
                
                # Cas spécifique: tuple de 3 éléments (h2o_frame, feature_cols, target_cols)
                if len(data) == 3 and isinstance(data[0], h2o.H2OFrame):
                    self.log("DEBUG: Tuple contenant (H2OFrame, feature_cols, target_cols) détecté")
                    h2o_df = data[0]
                    if feature_cols is None:
                        feature_cols = data[1]
                    if target_cols is None:
                        target_cols = data[2]
                    data = h2o_df
                
                # [le reste du code de gestion des tuples reste inchangé]
            
            if not self.h2o_initialized:
                self.initialiser_h2o()
            
            self.log("Entraînement du modèle AutoML...")
            
            # [le reste du code de conversion reste inchangé]
                
            if progress_callback:
                progress_callback(20)
                
            # Identifier les colonnes caractéristiques et cible
            if feature_cols is None and target_cols is None:
                # Par défaut, utiliser toutes les colonnes sauf la dernière comme caractéristiques,
                # et la dernière colonne comme cible
                feature_cols = h2o_df.columns[:-1]
                target_cols = h2o_df.columns[-1]
                
            self.log(f"Colonnes caractéristiques: {feature_cols}")
            self.log(f"Colonne(s) cible: {target_cols}")
            
            # Configurer et exécuter AutoML pour chaque colonne cible ou un modèle multivarié
            models = []
            
            # Si target_cols est une liste de plusieurs colonnes
            if isinstance(target_cols, list) and len(target_cols) > 1:
                self.log(f"Plusieurs colonnes cibles détectées ({len(target_cols)}). Entraînement d'un modèle multivarié.")
                
                # Option 1: Modèle multivarié combiné
                aml = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=1)
                
                # Pour H2O AutoML, nous devons spécifier une seule colonne 'y'
                # Nous allons créer une nouvelle colonne combinée ou nous entraînerons un modèle pour chaque colonne cible
                
                # Approche: entraîner un modèle AutoML pour chaque colonne cible
                self.log("Entraînement de modèles individuels pour chaque colonne cible")
                
                total_targets = len(target_cols)
                for i, target_col in enumerate(target_cols):
                    self.log(f"Entraînement pour la cible {i+1}/{total_targets}: {target_col}")
                    
                    if progress_callback:
                        progress_progress = 20 + (70 * i // total_targets)
                        progress_callback(progress_progress)
                        
                    aml_individual = H2OAutoML(max_models=max_models // total_targets if max_models else None, 
                                              max_runtime_secs=max_runtime_secs // total_targets if max_runtime_secs else None,
                                              seed=i+1)
                    aml_individual.train(x=feature_cols, y=target_col, training_frame=h2o_df)
                    models.append(aml_individual.leader)
                    
                # Utiliser le premier modèle comme modèle principal pour les logs et le retour
                aml.leader = models[0]
                
            else:
                # Si target_cols est une chaîne ou une liste d'une seule colonne
                if isinstance(target_cols, list) and len(target_cols) == 1:
                    target_cols = target_cols[0]
                
                self.log(f"Entraînement avec la colonne cible: {target_cols}")
                aml = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=1)
                aml.train(x=feature_cols, y=target_cols, training_frame=h2o_df)
                models.append(aml.leader)
            
            if progress_callback:
                progress_callback(90)
            
            self.log("Entraînement terminé.")
            self.log(f"Modèles entraînés: {len(models)}")
            for i, model in enumerate(models):
                self.log(f"Modèle {i+1}: {model.model_id}, Performance: {model.model_performance()}")
            
            # Sauvegarder les modèles
            model_paths = []
            for i, model in enumerate(models):
                model_path = h2o.save_model(model=model, path="./models", force=True)
                model_paths.append(model_path)
                self.log(f"Modèle {i+1} sauvegardé: {model_path}")
            
            if progress_callback:
                progress_callback(100)
                
            # Si un seul modèle, retourner ce modèle. Sinon, retourner la liste des modèles
            return models[0] if len(models) == 1 else models
            
        except Exception as e:
            self.log(f"Erreur lors de l'entraînement du modèle: {str(e)}")
            traceback.print_exc()
            raise e


    def predire_prochains_listes(self, model, data, n_prediction=5, progress_callback=None):
        """Utilise le modèle entraîné pour prédire les prochains"""
        if not self.h2o_initialized:
            self.initialiser_h2o()
            
        try:
            self.log("Prédiction des prochaines listes...")
            if progress_callback:
                progress_callback(10)
                
            if model is None:
                raise ValueError("Aucun modèle fourni pour les prédictions")
                
            # Récupérer les derniers listes pour créer les features
            all_draws = []
            for _, row in data.iterrows():
                draw = sorted([int(num) for num in row.values if not np.isnan(num)])
                all_draws.append(draw)
                
            latest_draws = all_draws[-5:]  # Prendre les 5 dernieres listes
            
            if progress_callback:
                progress_callback(30)
                
            # Créer le vecteur de features
            feature_vector = []
            
            # Utiliser les dernieres listes comme features
            for draw in latest_draws:
                for num in range(1, self.max_num + 1):
                    feature_vector.append(1 if num in draw else 0)
                    
            # Remplir avec des zéros si nécessaire
            expected_length = len(model.train_columns) - self.max_num
            while len(feature_vector) < expected_length:
                feature_vector = [0] * (self.max_num) + feature_vector
                
            feature_vector = feature_vector[-expected_length:]
            
            if progress_callback:
                progress_callback(50)
                
            # Créer un H2OFrame pour la prédiction
            feature_cols = [col for col in model.train_columns if not col.startswith("target_")]
            pred_df = pd.DataFrame([feature_vector], columns=feature_cols)
            h2o_pred = h2o.H2OFrame(pred_df)
            
            if progress_callback:
                progress_callback(70)
                
            # Faire la prédiction
            predictions = model.predict(h2o_pred)
            pred_proba = predictions.as_data_frame().iloc[0].values
            
            if progress_callback:
                progress_callback(90)
                
            # Convertir les probabilités en prédictions numériques
            pred_indices = np.argsort(pred_proba)[-n_prediction:]
            predicted_numbers = [idx + 1 for idx in pred_indices]
            
            self.log(f"Prédiction basée sur ML: {sorted(predicted_numbers)}")
            
            if progress_callback:
                progress_callback(100)
                
            return sorted(predicted_numbers)
            
        except Exception as e:
            self.log(f"Erreur lors de la prédiction: {e}")
            traceback.print_exc()
            if progress_callback:
                progress_callback(0)
            return []

    def fermer(self):
        """Arrête proprement H2O"""
        if self.h2o_initialized:
            self.log("Fermeture de H2O...")
            try:
                h2o.shutdown(prompt=False)
                self.log("H2O arrêté avec succès")
                self.h2o_initialized = False
            except:
                self.log("Erreur lors de l'arrêt de H2O")

# Pour tester le backend indépendamment
if __name__ == "__main__":
    predicteur = PredicteurListesBackend()
    predicteur.initialiser_h2o()
    # Exemple de chargement de données
    # data = predicteur.charger_donnees("exemple_listes.csv")
    # if data is not None:
    #     stats_df, predictions = predicteur.methode_statistique(data)
    #     print(f"Prédictions statistiques: {predictions}")
    predicteur.fermer()
