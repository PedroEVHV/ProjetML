import tkinter as tk
from tkinter import filedialog, Text, Scrollbar
from PIL import Image, ImageOps
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model

# Charger le modèle Keras
cnn_model = load_model('model_CNN.h5')

# Charger les autres modèles 
model2 = joblib.load('model_Random_Forest.pkl')
model3 = joblib.load('model_SVM.pkl')

# Créer le modèle intermédiaire pour extraire les caractéristiques
layer_name = 'dense_2'  # S'assurer que le nom de la couche soit bon
intermediate_layer_model = Model(inputs=cnn_model.input,
                                 outputs=cnn_model.get_layer(layer_name).output)

image_path = None

def charger_image():
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        image_label.config(text=f"Image chargée: {image_path}")

def predict_image(model, image_path):
    img = Image.open(image_path).convert('RGB')
    
    # Redimensionner l'image pour le modèle CNN
    img = img.resize((50, 50))
    img = np.array(img).astype('float32') / 255
    img_cnn = img.reshape(1, 50, 50, 3)
    
    # Extraire les caractéristiques intermédiaires si ce n'est pas un modèle CNN
    if model != cnn_model:
        features = intermediate_layer_model.predict(img_cnn)
        features = features.flatten().reshape(1, -1)  # "Aplatir" les caractéristiques
        prediction = model.predict(features)[0]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
        else:
            probabilities = [0] * 43  # Si la probabilité n'est pas disponible
    else:
        prediction = np.argmax(model.predict(img_cnn), axis=1)[0]
        probabilities = model.predict(img_cnn)[0]
    
    return prediction, probabilities

def format_probabilities(model_name, prediction, probabilities, threshold):
    sorted_indices = np.argsort(probabilities)[::-1]
    formatted_probs = ", ".join([f"{i} ({probabilities[i]*100:.1f}%)" for i in sorted_indices if probabilities[i]*100 >= threshold])
    return f"{model_name} : Résultat {prediction} - Détails : {formatted_probs}"

def lancer():
    threshold = prob_threshold.get()
    if image_path:
        results = []
        overall_probabilities = np.zeros(43)  # 43 classes (0-42)
        selected_model_count = 0
        has_results_above_threshold = False
        predictions = []

        if model1_var.get():
            prediction, probabilities = predict_image(cnn_model, image_path)
            if any(probabilities[i] * 100 >= threshold for i in np.argsort(probabilities)[::-1]):
                overall_probabilities += probabilities
                selected_model_count += 1
                formatted_result = format_probabilities("Modèle 1", prediction, probabilities, threshold)
                results.append(formatted_result)
                predictions.append(prediction)
                has_results_above_threshold = True
            else:
                formatted_result = f"Modèle 1 : Aucune classe avec une probabilité supérieure à {threshold}%"
                results.append(formatted_result)
        if model2_var.get():
            prediction, probabilities = predict_image(model2, image_path)
            if any(probabilities[i] * 100 >= threshold for i in np.argsort(probabilities)[::-1]):
                overall_probabilities += probabilities
                selected_model_count += 1
                formatted_result = format_probabilities("Modèle 2", prediction, probabilities, threshold)
                results.append(formatted_result)
                predictions.append(prediction)
                has_results_above_threshold = True
            else:
                formatted_result = f"Modèle 2 : Aucune classe avec une probabilité supérieure à {threshold}%"
                results.append(formatted_result)
        if model3_var.get():
            prediction, probabilities = predict_image(model3, image_path)
            if any(probabilities[i] * 100 >= threshold for i in np.argsort(probabilities)[::-1]):
                overall_probabilities += probabilities
                selected_model_count += 1
                formatted_result = format_probabilities("Modèle 3", prediction, probabilities, threshold)
                results.append(formatted_result)
                predictions.append(prediction)
                has_results_above_threshold = True
            else:
                formatted_result = f"Modèle 3 : Aucune classe avec une probabilité supérieure à {threshold}%"
                results.append(formatted_result)

        if results:
            for result in results:
                result_text.insert(tk.END, result + "\n")

            if selected_model_count > 0:  # Calcul faisant parti du résultat, de la moyenne seulement si le résultat du modèle sélectionné est au dessus du seuil choisi
                if len(set(predictions)) == 1:
                    most_probable_number = predictions[0]
                    max_probability = overall_probabilities[most_probable_number] * 100 / selected_model_count
                    conclusion_message = f"Conclusion : Le numéro le plus probable sur les différents modèles est {most_probable_number} avec une probabilité moyenne de {max_probability:.1f}%.\n"
                else:
                    conclusion_message = "Conclusion : Résultats différents selon les modèles.\n"
                result_text.insert(tk.END, conclusion_message)
    else:
        result_text.insert(tk.END, "Veuillez charger une image d'abord\n")
    result_text.insert(tk.END, "-" * 40 + "\n")  

def clear_results():
    result_text.delete('1.0', tk.END)

def show_classes():
    classes_window = tk.Toplevel(root)
    classes_window.title("Classes Existantes")
    classes_window.geometry("350x400")
    classes_window.grid_columnconfigure(0, weight=1)

    # Créer un cadre pour contenir le widget Text et les barres de défilement
    main_frame = tk.Frame(classes_window)
    main_frame.grid(row=0, column=0, sticky='nsew')

    # Ajouter un widget Text pour afficher les informations des classes
    classes_text = Text(main_frame, wrap='word', width=1, height=25)
    classes_text.pack(side='left', fill='both', expand=True)

    # Ajouter un scrollbar vertical pour le Text
    scroll_y_col1 = Scrollbar(main_frame, orient='vertical', command=classes_text.yview)
    scroll_y_col1.pack(side='right', fill='y')
    classes_text.config(yscrollcommand=scroll_y_col1.set)

    # Insérer les textes dans les colonnes
    classes_text.insert(tk.END, """
    0. Limite de vitesse (20 km/h)
    1. Limite de vitesse (30 km/h)
    2. Limite de vitesse (50 km/h)
    3. Limite de vitesse (60 km/h)
    4. Limite de vitesse (70 km/h)
    5. Limite de vitesse (80 km/h)
    6. Fin de limitation de vitesse
       (80 km/h)
    7. Limite de vitesse (100 km/h)
    8. Limite de vitesse (120 km/h)
    9. Interdiction de dépasser
    10. Interdiction de dépasser pour 
        les véhicules de plus de 3,5 
        tonnes
    11. Priorité à la prochaine 
        intersection
    12. Route prioritaire
    13. Cédez le passage
    14. Arrêt
    15. Interdiction de circuler
    16. Interdiction aux véhicules 
        de plus de 3,5 tonnes
    17. Sens interdit
    18. Danger général
    19. Virage dangereux à gauche
    20. Virage dangereux à droite
    21. Double virage
    22. Chaussée déformée
    23. Route glissante    
    24. Rétrécissement de chaussée 
        par la droite                  
    25. Travaux
    26. Feux tricolores
    27. Passage pour piétons
    28. Enfants
    29. Passage pour cyclistes
    30. Risque de neige ou de verglas
    31. Passage d'animaux sauvages
    32. Fin de toutes les interdictions 
        imposées aux véhicules en 
        mouvement
    33. Tourner à droite
    34. Tourner à gauche
    35. Tout droit
    36. Tout droit ou à droite
    37. Tout droit ou à gauche
    38. Obligation de tourner à droite
    39. Obligation de tourner à gauche
    40. Sens giratoire obligatoire
    41. Fin d'interdiction de dépasser
    42. Fin d'interdiction de dépasser 
        pour les véhicules de plus
        de 3,5 tonnes              
    """)

    classes_text.config(state=tk.DISABLED)

    # Centrer la fenêtre d'informations
    center_window(classes_window)

def select_all():
    all_selected = select_all_var.get()
    model1_var.set(all_selected)
    model2_var.set(all_selected)
    model3_var.set(all_selected)

def update_prob_label(value):
    prob_label.config(text=f"Définir le seuil de probabilité minimal : {value}%")

def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')

def show_all_models_info():
    models_info = [("CNN", "Modèle 1"), ("Random_Forest", "Modèle 2"), ("SVM", "Modèle 3")]  # Liste des modèles avec leur numéro et nom

    # Créer une nouvelle fenêtre pour afficher les informations
    info_window = tk.Toplevel(root)
    info_window.title("Informations sur les Modèles")
    info_window.geometry("600x590")
    info_window.resizable(False, False) 

    # Créer un cadre pour contenir le widget Text et les barres de défilement
    info_frame = tk.Frame(info_window)
    info_frame.pack(padx=10, pady=10, fill='both', expand=True)

    # Ajouter un widget Text pour afficher les informations
    info_text = Text(info_frame, wrap='none', width=70, height=30)
    info_text.pack(side='left', fill='both', expand=True)

    # Ajouter un scrollbar vertical pour le Text
    scroll_y = Scrollbar(info_frame, command=info_text.yview)
    scroll_y.pack(side='right', fill='y')

    # Configurer les scrollbars pour agir sur le widget Text
    info_text.config(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    for model_name, model_num in models_info:
        try:
            # Charger le rapport de classification et la matrice de confusion
            report = joblib.load(f'report_{model_name}.pkl')
            cm = joblib.load(f'cm_{model_name}.pkl')

            # Afficher le titre du modèle
            info_text.insert(tk.END, f"=== {model_num} ({model_name}) : ===\n\n")

            # Afficher le rapport de classification
            info_text.insert(tk.END, f"Classification report for the {model_name} :\n{report}\n\n")

            # Afficher la matrice de confusion
            info_text.insert(tk.END, f"Confusion Matrix for the {model_name} :\n{cm}\n\n")

        except FileNotFoundError:
            info_text.insert(tk.END, f"Les données pour {model_num} ({model_name}) n'ont pas été trouvées.\n\n")

    # Rendre le Text en lecture seule
    info_text.config(state=tk.DISABLED)

    # Centrer la fenêtre d'informations
    center_window(info_window)

# Créer la fenêtre principale
root = tk.Tk()
root.title("Interface de Reconnaissance d'Image")
root.geometry("600x590")
root.resizable(False, False)  

# Charger une image
charger_label = tk.Label(root, text="Charger votre image", font=('Helvetica', 10, 'bold'))
charger_label.pack(pady=10)

charger_button = tk.Button(root, text="Charger", command=charger_image)
charger_button.pack(pady=10)

image_label = tk.Label(root, text="")
image_label.pack(pady=5)

# Choisir le modèle
model_label = tk.Label(root, text="Choisir votre modèle", font=('Helvetica', 10, 'bold'))
model_label.pack(pady=10)

# Bouton pour afficher les informations
info_button = tk.Button(root, text="Information", command=show_all_models_info)
info_button.pack(pady=10)

model_frame = tk.Frame(root)
model_frame.pack(pady=10)

model1_var = tk.BooleanVar()
model1_checkbutton = tk.Checkbutton(model_frame, text="Modèle 1", variable=model1_var)
model1_checkbutton.grid(row=0, column=0, padx=5)

model2_var = tk.BooleanVar()
model2_checkbutton = tk.Checkbutton(model_frame, text="Modèle 2", variable=model2_var)
model2_checkbutton.grid(row=0, column=1, padx=5)

model3_var = tk.BooleanVar()
model3_checkbutton = tk.Checkbutton(model_frame, text="Modèle 3", variable=model3_var)
model3_checkbutton.grid(row=0, column=2, padx=5)

select_all_var = tk.BooleanVar()
select_all_checkbutton = tk.Checkbutton(model_frame, text="Tout sélectionner", variable=select_all_var, command=select_all)
select_all_checkbutton.grid(row=0, column=3, padx=5)

# Seuil de probabilité
prob_label = tk.Label(root, text="Définir le seuil de probabilité : 50%", font=('Helvetica', 10, 'bold'))
prob_label.pack(pady=5)

prob_threshold = tk.Scale(root, from_=0, to=100, orient='horizontal', length=400, command=update_prob_label)
prob_threshold.set(50)
prob_threshold.pack(pady=5)

# Bouton pour lancer
lancer_button = tk.Button(root, text="Lancer", command=lancer)
lancer_button.pack(pady=10)

# Ligne de séparation et bouton effacer
separator_frame = tk.Frame(root)
separator_frame.pack(fill='x', pady=10)

separator_label = tk.Label(separator_frame, text="Résultats", font=('Helvetica', 10, 'bold'))
separator_label.pack(anchor='center')

clear_button = tk.Button(separator_frame, text="Effacer", command=clear_results)
clear_button.pack(side='right', padx=15)  

# Bouton pour afficher les classes
classes_button = tk.Button(separator_frame, text="Afficher les classes", command=show_classes)
classes_button.pack(side='right', padx=5)

# Résultats avec barre de défilement horizontale
result_frame = tk.Frame(root)
result_frame.pack(pady=10, padx=10, fill='both', expand=True)

scroll_x = Scrollbar(result_frame, orient='horizontal')
scroll_x.pack(side='bottom', fill='x')

result_text = Text(result_frame, wrap='none', width=65, height=15, xscrollcommand=scroll_x.set)
result_text.pack(side='left', fill='both', expand=True)

scroll_x.config(command=result_text.xview)

scroll_y = Scrollbar(result_frame, orient='vertical', command=result_text.yview)
scroll_y.pack(side='right', fill='y')

result_text.config(yscrollcommand=scroll_y.set)

# Centrer la fenêtre au démarrage
root.update_idletasks()
width = root.winfo_width()
height = root.winfo_height()
x = (root.winfo_screenwidth() // 2) - (width // 2)
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f'{width}x{height}+{x}+{y}')

root.mainloop()
