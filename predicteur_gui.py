import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import time
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
import h2o
import traceback
from predicteur_backend import PredicteurListesBackend

class PredicteurListesGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Prédicteur de Listes")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        self.backend = PredicteurListesBackend(log_callback=self.log_message)
        self.loaded_data = None
        self.trained_model = None
        self.stats_df = None
        
        # Variables pour suivre l'état
        self.current_task = None
        self.task_running = False
        self.thread = None
        
        self.create_menu()
        self.create_notebook()
        self.create_status_bar()
        
    def create_menu(self):
        """Crée la barre de menu de l'application"""
        menu_bar = tk.Menu(self.root)
        
        # Menu Fichier
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Charger données...", command=self.load_data_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.quit_app)
        menu_bar.add_cascade(label="Fichier", menu=file_menu)
        
        # Menu Modèles
        model_menu = tk.Menu(menu_bar, tearoff=0)
        model_menu.add_command(label="Charger un modèle...", command=self.load_model)
        model_menu.add_command(label="Sauvegarder prédictions...", command=self.save_predictions)
        menu_bar.add_cascade(label="Modèles", menu=model_menu)
        
        # Menu Aide
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="À propos", command=self.show_about)
        menu_bar.add_cascade(label="Aide", menu=help_menu)
        
        self.root.config(menu=menu_bar)
        
    def create_notebook(self):
        """Crée les onglets de l'application"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Onglet Accueil
        self.home_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.home_tab, text="Accueil")
        self.setup_home_tab()
        
        # Onglet Analyse Statistique
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Analyse Statistique")
        self.setup_stats_tab()
        
        # Onglet Machine Learning
        self.ml_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_tab, text="Machine Learning")
        self.setup_ml_tab()
        
        # Onglet Logs
        self.logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_tab, text="Logs")
        self.setup_logs_tab()
        
    def create_status_bar(self):
        """Crée la barre de statut en bas de l'application"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side='bottom', fill='x')
        
        self.status_label = ttk.Label(self.status_bar, text="Prêt", anchor='w')
        self.status_label.pack(side='left', padx=5, pady=2)
        
        self.progress = ttk.Progressbar(self.status_bar, length=200, mode='determinate')
        self.progress.pack(side='right', padx=5, pady=2)
        
    def setup_home_tab(self):
        """Configure l'onglet d'accueil"""
        # Frame pour les informations sur les données
        data_frame = ttk.LabelFrame(self.home_tab, text="Données")
        data_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(data_frame, text="Source des données:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.data_source_label = ttk.Label(data_frame, text="Aucune donnée chargée")
        self.data_source_label.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(data_frame, text="Nombre de listes:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.num_draws_label = ttk.Label(data_frame, text="0")
        self.num_draws_label.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Button(data_frame, text="Charger des données", command=self.load_data_dialog).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(data_frame, text="Afficher les données", command=self.show_data).grid(row=2, column=1, padx=5, pady=5)
        
        # Frame pour les actions principales
        action_frame = ttk.LabelFrame(self.home_tab, text="Actions")
        action_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(action_frame, text="Analyse Statistique", command=lambda: self.notebook.select(1)).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(action_frame, text="Machine Learning", command=lambda: self.notebook.select(2)).grid(row=0, column=1, padx=5, pady=5)
        
        # Frame pour les résultats
        result_frame = ttk.LabelFrame(self.home_tab, text="Derniers résultats")
        result_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(result_frame, width=40, height=10, wrap='word')
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.results_text.insert(tk.END, "Aucun résultat disponible. Veuillez charger des données et effectuer une analyse.")
        self.results_text.config(state='disabled')
        
    def setup_stats_tab(self):
        """Configure l'onglet d'analyse statistique"""
        # Frame pour les options
        options_frame = ttk.LabelFrame(self.stats_tab, text="Options")
        options_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(options_frame, text="Nombre de prédictions:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.stats_n_pred_var = tk.StringVar(value="5")
        ttk.Spinbox(options_frame, from_=1, to=20, textvariable=self.stats_n_pred_var, width=5).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Button(options_frame, text="Analyser", command=self.run_statistical_analysis).grid(row=0, column=2, padx=5, pady=5)
        
        # Frame pour les graphiques
        graph_frame = ttk.LabelFrame(self.stats_tab, text="Visualisations")
        graph_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.stats_notebook = ttk.Notebook(graph_frame)
        self.stats_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Onglet pour la fréquence des numéros
        self.freq_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.freq_tab, text="Fréquence")
        
        # Onglet pour les tendances
        self.trend_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.trend_tab, text="Tendance")
        
        # Onglet pour les prédictions
        self.pred_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.pred_tab, text="Prédictions")
        
        # Onglet pour les données brutes
        self.raw_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.raw_tab, text="Données")
        
        # Messages d'information initiale
        ttk.Label(self.freq_tab, text="Chargez des données et lancez l'analyse pour voir les graphiques.").pack(expand=True)
        ttk.Label(self.trend_tab, text="Chargez des données et lancez l'analyse pour voir les graphiques.").pack(expand=True)
        ttk.Label(self.pred_tab, text="Chargez des données et lancez l'analyse pour voir les prédictions.").pack(expand=True)
        ttk.Label(self.raw_tab, text="Chargez des données et lancez l'analyse pour voir les statistiques.").pack(expand=True)
        
    def setup_ml_tab(self):
        """Configure l'onglet de machine learning"""
        # Frame pour les options d'entraînement
        train_frame = ttk.LabelFrame(self.ml_tab, text="Entraînement")
        train_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(train_frame, text="Temps maximum (secondes):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.ml_time_var = tk.StringVar(value="300")
        ttk.Entry(train_frame, textvariable=self.ml_time_var, width=8).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(train_frame, text="Mémoire H2O (GB):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.ml_memory_var = tk.StringVar(value="4")
        ttk.Spinbox(train_frame, from_=1, to=32, textvariable=self.ml_memory_var, width=5).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Button(train_frame, text="Entraîner", command=self.train_ml_model).grid(row=0, column=2, rowspan=2, padx=5, pady=5)
        
        # Frame pour les prédictions
        pred_frame = ttk.LabelFrame(self.ml_tab, text="Prédictions")
        pred_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(pred_frame, text="Nombre de prédictions:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.ml_n_pred_var = tk.StringVar(value="5")
        ttk.Spinbox(pred_frame, from_=1, to=20, textvariable=self.ml_n_pred_var, width=5).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Button(pred_frame, text="Prédire", command=self.predict_with_ml).grid(row=0, column=2, padx=5, pady=5)
        
        # Frame pour les résultats
        result_frame = ttk.LabelFrame(self.ml_tab, text="Résultats")
        result_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.ml_results_text = scrolledtext.ScrolledText(result_frame, width=40, height=10, wrap='word')
        self.ml_results_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.ml_results_text.insert(tk.END, "Entraînez d'abord un modèle ou chargez-en un existant pour faire des prédictions.")
        self.ml_results_text.config(state='disabled')
        
    def setup_logs_tab(self):
        """Configure l'onglet des logs"""
        # Frame pour les logs
        log_frame = ttk.Frame(self.logs_tab)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Zone de texte pour les logs
        self.log_text = scrolledtext.ScrolledText(log_frame, width=40, height=20, wrap='word')
        self.log_text.pack(fill='both', expand=True)
        
        # Boutons
        button_frame = ttk.Frame(self.logs_tab)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(button_frame, text="Effacer les logs", command=self.clear_logs).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Sauvegarder les logs", command=self.save_logs).pack(side='left', padx=5)
        
    def log_message(self, message):
        """Ajoute un message au log avec horodatage"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # S'assurer que l'update est thread-safe
        self.root.after(0, lambda: self._update_log(log_entry))
        
    def _update_log(self, log_entry):
        """Met à jour la zone de log (doit être appelé depuis le thread principal)"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        
    def clear_logs(self):
        """Efface tous les logs"""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        
    def save_logs(self):
        """Sauvegarde les logs dans un fichier"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")],
            title="Sauvegarder les logs"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Sauvegarde réussie", f"Logs sauvegardés dans {file_path}")
            except Exception as e:
                messagebox.showerror("Erreur de sauvegarde", f"Impossible de sauvegarder les logs: {e}")
    
    def update_progress(self, value):
        """Met à jour la barre de progression (thread-safe)"""
        self.root.after(0, lambda: self._set_progress(value))
        
    def _set_progress(self, value):
        """Met à jour la barre de progression (doit être appelé depuis le thread principal)"""
        self.progress['value'] = value
        if value >= 100:
            self.status_label.config(text="Terminé")
        elif value <= 0:
            self.status_label.config(text="Erreur")
        else:
            self.status_label.config(text=f"En cours... {value}%")
            
    def load_data_dialog(self):
        """Ouvre une boîte de dialogue pour charger un fichier de données"""
        if self.task_running:
            messagebox.showwarning("Opération en cours", "Une tâche est déjà en cours. Veuillez attendre qu'elle se termine.")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")],
            title="Sélectionner un fichier de données"
        )
        
        if file_path:
            self.task_running = True
            self.current_task = "loading_data"
            self.status_label.config(text="Chargement des données...")
            self.progress['value'] = 0
            
            # Lancer le chargement dans un thread séparé
            self.thread = threading.Thread(target=self.load_data, args=(file_path,))
            self.thread.daemon = True
            self.thread.start()
            
    def load_data(self, file_path):
        """Charge les données à partir du fichier sélectionné"""
        try:
            self.loaded_data = self.backend.charger_donnees(file_path, progress_callback=self.update_progress)
            
            if self.loaded_data is not None:
                # Mettre à jour l'interface utilisateur
                self.root.after(0, lambda: self._update_data_info(file_path))
            else:
                self.root.after(0, lambda: messagebox.showerror("Erreur de chargement", 
                                                              "Impossible de charger les données. Vérifiez les logs pour plus d'informations."))
        except Exception as e:
            self.log_message(f"Erreur lors du chargement des données: {e}")
            traceback.print_exc()
            self.root.after(0, lambda err=e: messagebox.showerror("Erreur", f"Une erreur est survenue: {err}"))
        finally:
            self.task_running = False
            
    def _update_data_info(self, file_path):
        """Met à jour les informations sur les données chargées"""
        self.data_source_label.config(text=os.path.basename(file_path))
        self.num_draws_label.config(text=str(len(self.loaded_data)))
        self.clear_result_text()
        self.add_to_result_text(f"Données chargées depuis {os.path.basename(file_path)}\n")
        self.add_to_result_text(f"Nombre de listes: {len(self.loaded_data)}")
        messagebox.showinfo("Chargement réussi", f"{len(self.loaded_data)} listes chargés.")
        
    def show_data(self):
        """Affiche les données chargées dans une fenêtre séparée"""
        if self.loaded_data is None:
            messagebox.showinfo("Aucune donnée", "Veuillez d'abord charger des données.")
            return
            
        data_window = tk.Toplevel(self.root)
        data_window.title("Données chargées")
        data_window.geometry("800x600")
        
        # Créer un widget pour afficher les données
        frame = ttk.Frame(data_window)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Créer un treeview pour afficher le DataFrame
        tree = ttk.Treeview(frame)
        
        # Ajouter les colonnes
        tree["columns"] = list(self.loaded_data.columns)
        tree["show"] = "headings"  # Ne pas afficher la première colonne vide
        
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=50)
            
        # Ajouter les lignes
        for i, row in self.loaded_data.iterrows():
            if i < 1000:  # Limiter le nombre de lignes pour des raisons de performance
                tree.insert("", "end", values=list(row))
            else:
                break
                
        # Ajouter des barres de défilement
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Packing
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        tree.pack(side='left', fill='both', expand=True)
        
        if len(self.loaded_data) > 1000:
            ttk.Label(data_window, text=f"Affichage limité aux 1000 premières lignes sur {len(self.loaded_data)}").pack(pady=5)
            
    def run_statistical_analysis(self):
        """Exécute l'analyse statistique sur les données chargées"""
        if self.task_running:
            messagebox.showwarning("Opération en cours", "Une tâche est déjà en cours. Veuillez attendre qu'elle se termine.")
            return
            
        if self.loaded_data is None:
            messagebox.showinfo("Aucune donnée", "Veuillez d'abord charger des données.")
            return
            
        try:
            n_prediction = int(self.stats_n_pred_var.get())
            if n_prediction < 1:
                messagebox.showwarning("Valeur invalide", "Le nombre de prédictions doit être au moins 1.")
                return
        except ValueError:
            messagebox.showwarning("Valeur invalide", "Veuillez entrer un nombre valide pour le nombre de prédictions.")
            return
            
        self.task_running = True
        self.current_task = "statistical_analysis"
        self.status_label.config(text="Analyse statistique en cours...")
        self.progress['value'] = 0
        
        # Lancer l'analyse dans un thread séparé
        self.thread = threading.Thread(target=self.perform_statistical_analysis, args=(n_prediction,))
        self.thread.daemon = True
        self.thread.start()
        
    def perform_statistical_analysis(self, n_prediction):
        """Exécute l'analyse statistique dans un thread séparé"""
        try:
            stats_df, predictions = self.backend.methode_statistique(self.loaded_data, n_prediction, 
                                                                    progress_callback=self.update_progress)
            
            if stats_df is not None:
                self.stats_df = stats_df
                self.root.after(0, lambda: self._update_stats_results(stats_df, predictions))
            else:
                self.root.after(0, lambda: messagebox.showerror("Erreur d'analyse", 
                                                              "L'analyse statistique a échoué. Vérifiez les logs pour plus d'informations."))
        except Exception as e:
            self.log_message(f"Erreur lors de l'analyse statistique: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Erreur", f"Une erreur est survenue: {e}"))
        finally:
            self.task_running = False
            
    def _update_stats_results(self, stats_df, predictions):
        """Met à jour les résultats de l'analyse statistique dans l'interface"""
        # Mettre à jour le texte des résultats
        self.clear_result_text()
        self.add_to_result_text("Résultats de l'analyse statistique:\n")
        self.add_to_result_text(f"Prédiction pour le prochaine liste: {predictions}")
        
        # Mettre à jour les graphiques
        self._create_frequency_chart(stats_df)
        self._create_trend_chart(stats_df)
        self._create_prediction_chart(stats_df, predictions)
        self._display_stats_data(stats_df)
        
        messagebox.showinfo("Analyse terminée", "L'analyse statistique est terminée.")
        
    def _create_frequency_chart(self, stats_df):
        """Crée un graphique de fréquence des numéros"""
        # Effacer l'onglet
        for widget in self.freq_tab.winfo_children():
            widget.destroy()
            
        # Créer le graphique
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Tracer la fréquence
        sns.barplot(x='number', y='frequency', data=stats_df, ax=ax)
        ax.set_title('Fréquence des numéros')
        ax.set_xlabel('Numéro')
        ax.set_ylabel('Fréquence (%)')
        
        # Limiter l'affichage des xticks si trop de numéros
        if len(stats_df) > 30:
            ax.set_xticks(ax.get_xticks()[::5])
            
        fig.tight_layout()
        
        # Ajouter le graphique à l'interface
        canvas = FigureCanvasTkAgg(fig, master=self.freq_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Ajouter une barre d'outils pour le graphique
        toolbar = NavigationToolbar2Tk(canvas, self.freq_tab)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def _create_trend_chart(self, stats_df):
        """Crée un graphique de tendance des numéros"""
        # Effacer l'onglet
        for widget in self.trend_tab.winfo_children():
            widget.destroy()
            
        # Créer le graphique
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Tracer la tendance
        sns.barplot(x='number', y='trend', data=stats_df, ax=ax)
        ax.set_title('Tendance des numéros (récent vs total)')
        ax.set_xlabel('Numéro')
        ax.set_ylabel('Variation de fréquence (%)')
        
        # Ajouter une ligne horizontale à y=0
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
        
        # Limiter l'affichage des xticks si trop de numéros
        if len(stats_df) > 30:
            ax.set_xticks(ax.get_xticks()[::5])
            
        fig.tight_layout()
        
        # Ajouter le graphique à l'interface
        canvas = FigureCanvasTkAgg(fig, master=self.trend_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Ajouter une barre d'outils pour le graphique
        toolbar = NavigationToolbar2Tk(canvas, self.trend_tab)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def _create_prediction_chart(self, stats_df, predictions):
        """Crée un graphique des numéros prédits"""
        # Effacer l'onglet
        for widget in self.pred_tab.winfo_children():
            widget.destroy()
            
        # Créer le graphique
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Préparer les données pour le graphique
        pred_df = stats_df.copy()
        pred_df['is_predicted'] = pred_df['number'].isin(predictions)
        
        # Colorier les barres selon si le numéro est prédit ou non
        sns.barplot(x='number', y='frequency', data=pred_df, 
                   hue='is_predicted', palette={True: 'red', False: 'grey'}, ax=ax)
        
        ax.set_title('Numéros prédits et leur fréquence')
        ax.set_xlabel('Numéro')
        ax.set_ylabel('Fréquence (%)')
        ax.legend(title='Prédit', labels=['Non', 'Oui'])
        
        # Limiter l'affichage des xticks si trop de numéros
        if len(stats_df) > 30:
            ax.set_xticks(ax.get_xticks()[::5])
            
        fig.tight_layout()
        
        # Ajouter le graphique à l'interface
        canvas = FigureCanvasTkAgg(fig, master=self.pred_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Ajouter une barre d'outils pour le graphique
        toolbar = NavigationToolbar2Tk(canvas, self.pred_tab)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Ajouter le texte des prédictions
        pred_frame = ttk.Frame(self.pred_tab)
        pred_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(pred_frame, text="Numéros prédits:", font=('Arial', 12, 'bold')).pack(side='left', padx=5)
        ttk.Label(pred_frame, text=str(sorted(predictions)), font=('Arial', 12)).pack(side='left', padx=5)
        
    def _display_stats_data(self, stats_df):
        """Affiche les données statistiques brutes"""
        # Effacer l'onglet
        for widget in self.raw_tab.winfo_children():
            widget.destroy()
            
        # Créer un widget de texte pour afficher les données
        frame = ttk.Frame(self.raw_tab)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Créer un treeview
        tree = ttk.Treeview(frame)
        
        # Ajouter les colonnes
        tree["columns"] = list(stats_df.columns)
        tree["show"] = "headings"  # Ne pas afficher la première colonne vide
        
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=100)
            
        # Ajouter les lignes
        for i, row in stats_df.iterrows():
            tree.insert("", "end", values=list(row))
                
        # Ajouter des barres de défilement
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Packing
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        tree.pack(side='left', fill='both', expand=True)
        
    def train_ml_model(self):
        """Entraîne un modèle de machine learning sur les données chargées"""
        if self.task_running:
            messagebox.showwarning("Opération en cours", "Une tâche est déjà en cours. Veuillez attendre qu'elle se termine.")
            return
            
        if self.loaded_data is None:
            messagebox.showinfo("Aucune donnée", "Veuillez d'abord charger des données.")
            return
            
        try:
            max_time = int(self.ml_time_var.get())
            memory = f"{int(self.ml_memory_var.get())}G"
            if max_time < 10:
                messagebox.showwarning("Valeur invalide", "Le temps d'entraînement doit être d'au moins 10 secondes.")
                return
        except ValueError:
            messagebox.showwarning("Valeur invalide", "Veuillez entrer des valeurs numériques valides.")
            return
            
        self.task_running = True
        self.current_task = "ml_training"
        self.status_label.config(text="Initialisation de H2O...")
        self.progress['value'] = 0
        
        # Lancer l'entraînement dans un thread séparé
        self.thread = threading.Thread(target=self.perform_ml_training, args=(max_time, memory))
        self.thread.daemon = True
        self.thread.start()
        
    def perform_ml_training(self, max_time, memory):
        """Exécute l'entraînement du modèle ML dans un thread séparé"""
        try:
            # Initialiser H2O
            h2o_initialized = self.backend.initialiser_h2o(memory=memory, progress_callback=self.update_progress)
            if not h2o_initialized:
                self.root.after(0, lambda: messagebox.showerror("Erreur d'initialisation", 
                                                               "Impossible d'initialiser H2O. Vérifiez les logs pour plus d'informations."))
                return
                
            # Préparer les données
            self.root.after(0, lambda: self.status_label.config(text="Préparation des données..."))
            
            train_data = self.backend.preparer_donnees_ml(self.loaded_data, progress_callback=self.update_progress)
            if train_data is None:
                self.root.after(0, lambda: messagebox.showerror("Erreur de préparation", 
                                                               "Impossible de préparer les données. Vérifiez les logs pour plus d'informations."))
                return
                
            # Entraîner le modèle
            self.root.after(0, lambda: self.status_label.config(text="Entraînement du modèle..."))
            
            self.trained_model = self.backend.entrainer_modele(train_data, progress_callback=self.update_progress)
            
            if self.trained_model:
                self.root.after(0, lambda: self._update_ml_results("Modèle entraîné avec succès."))
                self.root.after(0, lambda: messagebox.showinfo("Entraînement terminé", 
                                                              "Le modèle de machine learning a été entraîné avec succès."))
            else:
                self.root.after(0, lambda: messagebox.showerror("Erreur d'entraînement", 
                                                               "L'entraînement du modèle a échoué. Vérifiez les logs pour plus d'informations."))
        except Exception as e:
            self.log_message(f"Erreur lors de l'entraînement du modèle: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Erreur", f"Une erreur est survenue: {e}"))
        finally:
            self.task_running = False
            
    def predict_with_ml(self):
        """Effectue des prédictions avec le modèle ML entraîné"""
        if self.task_running:
            messagebox.showwarning("Opération en cours", "Une tâche est déjà en cours. Veuillez attendre qu'elle se termine.")
            return
            
        if not self.backend.h2o_initialized:
            messagebox.showinfo("H2O non initialisé", "Veuillez d'abord initialiser H2O en entraînant un modèle.")
            return
            
        if self.loaded_data is None:
            messagebox.showinfo("Aucune donnée", "Veuillez d'abord charger des données.")
            return
            
        if self.trained_model is None:
            messagebox.showinfo("Aucun modèle", "Veuillez d'abord entraîner un modèle.")
            return
            
        try:
            n_prediction = int(self.ml_n_pred_var.get())
            if n_prediction < 1:
                messagebox.showwarning("Valeur invalide", "Le nombre de prédictions doit être au moins 1.")
                return
        except ValueError:
            messagebox.showwarning("Valeur invalide", "Veuillez entrer un nombre valide pour le nombre de prédictions.")
            return
            
        self.task_running = True
        self.current_task = "ml_prediction"
        self.status_label.config(text="Prédiction en cours...")
        self.progress['value'] = 0
        
        # Lancer la prédiction dans un thread séparé
        self.thread = threading.Thread(target=self.perform_ml_prediction, args=(n_prediction,))
        self.thread.daemon = True
        self.thread.start()
        
    def perform_ml_prediction(self, n_prediction):
        """Effectue des prédictions ML dans un thread séparé"""
        try:
            predictions = self.backend.predire_listes(self.trained_model, self.loaded_data, n_prediction, 
                                                     progress_callback=self.update_progress)
            
            if predictions:
                self.root.after(0, lambda: self._update_ml_predictions(predictions))
            else:
                self.root.after(0, lambda: messagebox.showerror("Erreur de prédiction", 
                                                               "La prédiction a échoué. Vérifiez les logs pour plus d'informations."))
        except Exception as e:
            self.log_message(f"Erreur lors de la prédiction: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Erreur", f"Une erreur est survenue: {e}"))
        finally:
            self.task_running = False
            
    def _update_ml_results(self, message):
        """Met à jour les résultats d'entraînement ML dans l'interface"""
        self.ml_results_text.config(state='normal')
        self.ml_results_text.delete(1.0, tk.END)
        self.ml_results_text.insert(tk.END, message)
        
        if self.trained_model:
            leaderboard = self.backend.get_leaderboard(self.trained_model)
            if leaderboard is not None:
                self.ml_results_text.insert(tk.END, "\n\nPerformance des modèles:\n")
                self.ml_results_text.insert(tk.END, str(leaderboard))
                
        self.ml_results_text.config(state='disabled')
        
    def _update_ml_predictions(self, predictions):
        """Met à jour les prédictions ML dans l'interface"""
        self.ml_results_text.config(state='normal')
        self.ml_results_text.insert(tk.END, "\n\nPrédiction pour le prochain liste:\n")
        self.ml_results_text.insert(tk.END, str(sorted(predictions)))
        self.ml_results_text.config(state='disabled')
        
        # Mettre à jour le texte des résultats
        self.clear_result_text()
        self.add_to_result_text("Résultats de la prédiction ML:\n")
        self.add_to_result_text(f"Prédiction pour le prochain liste: {sorted(predictions)}")
        
        messagebox.showinfo("Prédiction terminée", "La prédiction par machine learning est terminée.")
        
    def clear_result_text(self):
        """Effacer le texte des résultats"""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')
        
    def add_to_result_text(self, message):
        """Ajouter un message au texte des résultats"""
        self.results_text.config(state='normal')
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.config(state='disabled')
        
    def load_model(self):
        """Charge un modèle H2O existant"""
        if self.task_running:
            messagebox.showwarning("Opération en cours", "Une tâche est déjà en cours. Veuillez attendre qu'elle se termine.")
            return
            
        dir_path = filedialog.askdirectory(title="Sélectionner le répertoire du modèle")
        
        if dir_path:
            self.task_running = True
            self.current_task = "loading_model"
            self.status_label.config(text="Chargement du modèle...")
            self.progress['value'] = 0
            
            # Lancer le chargement dans un thread séparé
            self.thread = threading.Thread(target=self.perform_model_loading, args=(dir_path,))
            self.thread.daemon = True
            self.thread.start()
            
    def perform_model_loading(self, model_path):
        """Charge un modèle H2O existant dans un thread séparé"""
        try:
            # Initialiser H2O si nécessaire
            if not self.backend.h2o_initialized:
                h2o_initialized = self.backend.initialiser_h2o(progress_callback=self.update_progress)
                if not h2o_initialized:
                    self.root.after(0, lambda: messagebox.showerror("Erreur d'initialisation", 
                                                                  "Impossible d'initialiser H2O. Vérifiez les logs pour plus d'informations."))
                    return
            
            self.trained_model = self.backend.charger_modele(model_path, progress_callback=self.update_progress)
            
            if self.trained_model:
                self.root.after(0, lambda: self._update_ml_results("Modèle chargé avec succès."))
                self.root.after(0, lambda: messagebox.showinfo("Chargement terminé", 
                                                             "Le modèle de machine learning a été chargé avec succès."))
            else:
                self.root.after(0, lambda: messagebox.showerror("Erreur de chargement", 
                                                              "Le chargement du modèle a échoué. Vérifiez les logs pour plus d'informations."))
        except Exception as e:
            self.log_message(f"Erreur lors du chargement du modèle: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Erreur", f"Une erreur est survenue: {e}"))
        finally:
            self.task_running = False
            
    def save_predictions(self):
        """Sauvegarde les dernières prédictions dans un fichier"""
        # Récupérer le texte actuel des résultats
        results = self.results_text.get(1.0, tk.END)
        
        if "Prédiction pour le prochaine liste" not in results:
            messagebox.showinfo("Aucune prédiction", "Il n'y a pas de prédictions à sauvegarder.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")],
            title="Sauvegarder les prédictions"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(results)
                messagebox.showinfo("Sauvegarde réussie", f"Prédictions sauvegardées dans {file_path}")
            except Exception as e:
                messagebox.showerror("Erreur de sauvegarde", f"Impossible de sauvegarder les prédictions: {e}")
                
    def show_about(self):
        """Affiche des informations sur l'application"""
        about_text = """
        Prédicteur de Listes v4.0.8
        
        
        basée sur des méthodes statistiques et de machine learning.
        
        Utilise H2O AutoML pour l'apprentissage automatique.
        
       
        """
        
        messagebox.showinfo("À propos", about_text)
        
    def quit_app(self):
        """Ferme proprement l'application"""
        if self.task_running:
            response = messagebox.askyesno("Tâche en cours", 
                                          "Une tâche est en cours d'exécution. Voulez-vous vraiment quitter?")
            if not response:
                return
                
        # Fermer H2O
        self.backend.fermer()
        
        # Terminer le thread en cours s'il existe
        if self.thread and self.thread.is_alive():
            self.thread.join(0.1)  # Attendre un peu pour que le thread se termine proprement
            
        self.root.destroy()
        
# Point d'entrée de l'application
if __name__ == "__main__":
    # Configuration du style Ttk (thème)
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')  # Utilisez 'clam', 'alt', 'default', 'classic' selon votre préférence
    
    app = PredicteurListesGUI(root)
    
    # Intercepter la fermeture de la fenêtre
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    
    # Démarrer la boucle principale
    root.mainloop()
