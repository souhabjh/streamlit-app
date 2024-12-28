import pickle
from sklearn.ensemble import RandomForestClassifier

# Exemple de données et entraînement
X = [[50, 60], [70, 80], [90, 40], [30, 20]]  # Variables explicatives
y = ["Moyenne", "Élevée", "Faible", "Faible"]  # Classes cibles

model = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='gini')
model.fit(X, y)

# Sauvegarde du modèle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
