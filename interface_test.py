import tkinter as tk
from tkinter import filedialog, Text, Scrollbar
from PIL import Image, ImageOps
import numpy as np
import joblib

# Définir en plus seuil + Conclusion calculée 

# Charger les modèles
model1 = joblib.load('model_test_1.pkl')
model2 = joblib.load('model_test_2.pkl')
model3 = joblib.load('model_test_3.pkl')

image_path = None

def charger_image():
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        image_label.config(text=f"Image chargée: {image_path}")

def predict_image(model, image_path):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((8, 8))
    img = np.array(img).astype('float32') / 16
    img = img.reshape(1, -1)
    prediction = model.predict(img)[0]
    probabilities = model.predict_proba(img)[0]
    return prediction, probabilities

def format_probabilities(model_name, prediction, probabilities, threshold):
    sorted_indices = np.argsort(probabilities)[::-1]
    formatted_probs = ", ".join([f"{i} ({probabilities[i]*100:.1f}%)" for i in sorted_indices if probabilities[i]*100 >= threshold])
    return f"{model_name} : Résultat {prediction} - Détails : {formatted_probs}"

def lancer():
    threshold = prob_threshold.get()
    if image_path:
        results = []
        overall_probabilities = np.zeros(10)  # Assuming 10 classes (0-9)
        selected_model_count = 0
        has_results_above_threshold = False

        if model1_var.get():
            prediction, probabilities = predict_image(model1, image_path)
            overall_probabilities += probabilities
            selected_model_count += 1
            formatted_result = format_probabilities("Modèle 1", prediction, probabilities, threshold)
            if any(probabilities[i] * 100 >= threshold for i in np.argsort(probabilities)[::-1]):
                results.append(formatted_result)
                has_results_above_threshold = True
        if model2_var.get():
            prediction, probabilities = predict_image(model2, image_path)
            overall_probabilities += probabilities
            selected_model_count += 1
            formatted_result = format_probabilities("Modèle 2", prediction, probabilities, threshold)
            if any(probabilities[i] * 100 >= threshold for i in np.argsort(probabilities)[::-1]):
                results.append(formatted_result)
                has_results_above_threshold = True
        if model3_var.get():
            prediction, probabilities = predict_image(model3, image_path)
            overall_probabilities += probabilities
            selected_model_count += 1
            formatted_result = format_probabilities("Modèle 3", prediction, probabilities, threshold)
            if any(probabilities[i] * 100 >= threshold for i in np.argsort(probabilities)[::-1]):
                results.append(formatted_result)
                has_results_above_threshold = True

        if not results:
            result_text.insert(tk.END, f"Aucun résultat à une probabilité supérieure au seuil défini ({threshold}%)\n")
        else:
            for result in results:
                result_text.insert(tk.END, result + "\n")

            if has_results_above_threshold:
                average_probabilities = overall_probabilities / selected_model_count
                most_probable_number = np.argmax(average_probabilities)
                max_probability = average_probabilities[most_probable_number] * 100
                conclusion_message = f"Conclusion : Le numéro le plus probable sur les différents modèles est {most_probable_number} avec une probabilité moyenne de {max_probability:.1f}%.\n"
                result_text.insert(tk.END, conclusion_message)
    else:
        result_text.insert(tk.END, "Veuillez charger une image d'abord\n")
    result_text.insert(tk.END, "-" * 40 + "\n")  # Ajouter une ligne séparatrice après le lancement

def clear_results():
    result_text.delete('1.0', tk.END)

def select_all():
    all_selected = select_all_var.get()
    model1_var.set(all_selected)
    model2_var.set(all_selected)
    model3_var.set(all_selected)

def update_prob_label(value):
    prob_label.config(text=f"Définir le seuil de probabilité : {value}%")

# Centrer la fenêtre sur l'écran
def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    vroot_height = window.winfo_vrootheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    # Ajuster pour la barre des tâches si elle est en bas
    if vroot_height > screen_height:
        y -= (vroot_height - screen_height)

    window.geometry(f'{width}x{height}+{x}+{y}')

# Créer la fenêtre principale
root = tk.Tk()
root.title("Interface de Reconnaissance d'Image")
root.geometry("600x590")
root.resizable(False, False)  # Désactiver le redimensionnement

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
clear_button.pack(side='right', padx=15)  # Adjust padding to move the button slightly left

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
center_window(root)

root.mainloop()
