# -*- coding: utf-8 -*-
"""
https://colab.research.google.com/drive/1wy4wVS2NMtXgS_1_nx6zzItfNLt12oYf
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('agaricus-lepiota.data', header=None)
columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]
data.columns = columns
data.head()
print(data.isin(['?']).sum())

"""
Une seul varible contient les valeur manquante qui represente presque 30% du data
Apres traitment des valeurs manquantes
"""
data.replace('?', np.nan, inplace=True)
most_frequent = data['stalk-root'].mode()[0]
data['stalk-root'] = data['stalk-root'].fillna(most_frequent)
print("apres traitment des valeurs manquantes ")
print(data.isnull().values.any())



#Countplot
"""
diagramme en barres qui montre frequences (comptes) pour chaque valeur d'une variable categorique.
"""
n_cols = 5
n_rows = (len(columns) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()
for i, c in enumerate(columns):
    sns.countplot(x=c, data=data, ax=axes[i])
    axes[i].set_title(c)
    axes[i].tick_params(axis='x', rotation=45)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()
"""distribution (frequence) des classes e->edible (comestible) p -> poisonous (toxique)"""
sns.countplot(x='class', data=data)
plt.title("Distribution des classes (comestible ou toxique)")
plt.show()
"""les classes sont equilibres"""



#mosaic
"""
mosaic plot  utile pour visualiser la relation entre deux variables categoriques. ici class et odor
Les variables ou les rectangles sont fortement desequilibres (ex: une odeur qui est presque toujours toxique) sont probablement tres utiles.
"""
from statsmodels.graphics.mosaicplot import mosaic
features = [col for col in columns if col != 'class']
n_cols = 4
n_rows = (len(features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()
for i, feature in enumerate(features):
    mosaic(data, [feature, 'class'], ax=axes[i])
    axes[i].set_title(f"{feature} vs class", fontsize=10)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()


#Encodage
"""
Encoder les variables catégoriques en nombres
"""
from sklearn.preprocessing import LabelEncoder
X = data.drop('class', axis=1)
y = data['class']
X_encoded = X.apply(LabelEncoder().fit_transform)
y_encoded = LabelEncoder().fit_transform(y)
"""Quand on utilise LabelEncoder, il assigne des entiers par ordre alphabetique. Donc :
'edible' -> 0
'poisonous'-> 1
d'ou :
Classe positive (1) = 'poisonous' = toxique
Classe negative (0) = 'edible' = comestible"""



#chi2
from sklearn.feature_selection import chi2
chi_scores, p_values = chi2(X_encoded, y_encoded)
chi2_results = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi_scores,
    'p-value': p_values
}).sort_values(by='Chi2 Score', ascending=False)
print(chi2_results)

"""Chi2 Score eleve = forte dependance avec class = utile pour le modele
p-value faible (ex: < 0.05) = resultat statistiquement significatif
"""
colonnes_a_supprimer = ['cap-color', 'veil-color', 'gill-attachment', 'veil-type','odor']
X_encoded = X_encoded.drop(columns=colonnes_a_supprimer)



#KNN
"""Chercher Meilleur K"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)
"""Nous allons choisir le k qui minimise les erreurs ou un champignon non comestible est predit comme comestible, car ce type d'erreur est dangereux.
1ere methode:
"""
from sklearn.metrics import confusion_matrix
faux_negatifs = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fn = cm[1, 0]
    faux_negatifs.append(fn)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), faux_negatifs, marker='o', color='red')
plt.title("Faux negatifs vs k (nombre de voisins)")
plt.xlabel("k (n_neighbors)")
plt.ylabel("Nombre de faux negatifs")
plt.grid(True)
plt.show()
best_k = faux_negatifs.index(min(faux_negatifs)) + 1
print(f"Meilleur k pour minimiser les faux negatifs = {best_k} (avec {min(faux_negatifs)} faux negatifs)")

"""2eme methode:"""
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, confusion_matrix
def false_negative_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fn = cm[1, 0]
    positives = cm[1].sum()
    return fn / positives if positives > 0 else 0
scorer = make_scorer(false_negative_rate, greater_is_better=False)
param_grid = {'n_neighbors': list(range(1, 21))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring=scorer)
grid.fit(X_train, y_train)
print("Meilleur k pour minimiser les faux negatifs :", grid.best_params_)

"""Metric"""
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
best_k = grid.best_params_['n_neighbors']
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
cv_results = {}
for metric in metrics:
    knn_best = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    cv_scores = cross_val_score(knn_best, X_train, y_train, cv=5)
    cv_results[metric] = {
        'scores': cv_scores,
        'mean_score': cv_scores.mean()
    }
for metric, result in cv_results.items():
    print(f"Metric: {metric}")
    print(f"Scores cross-validation: {result['scores']}")
    print(f"Moyenne de validation: {result['mean_score']}\n")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

metrics = ['cosine', 'jaccard']
cv_results = {}

for metric in metrics:
    print(f"Test de la metrique : {metric}")
    try:
        knn_best = make_pipeline(
            FunctionTransformer(to_dense),
            KNeighborsClassifier(n_neighbors=best_k, metric=metric)
        )
        cv_scores = cross_val_score(knn_best, X_train, y_train, cv=5)
        cv_results[metric] = {
            'scores': cv_scores,
            'mean_score': cv_scores.mean()
        }
    except Exception as e:
        print(f" Erreur avec la metrique {metric} : {e}\n")

for metric, result in cv_results.items():
    print(f"Metric: {metric}")
    print(f"Scores cross-validation: {result['scores']}")
    print(f"Moyenne de validation: {result['mean_score']}\n")

"""Cross validation
Entraine et valide le modele plusieurs fois en changeant le jeu de validation a chaque fois
-> estimer la performance generale du modele sur de nouvelles donnees. Cela permet d'eviter de tirer des conclusions trop hatives a partir d'un seul entrainement et de tests sur un seul split
"""
best_k = grid.best_params_['n_neighbors']
knn_best = KNeighborsClassifier(n_neighbors=best_k,metric='manhattan')
cv_scores = cross_val_score(knn_best, X_train, y_train, cv=5)
print("Scores cross-validation :", cv_scores)
print("Moyenne (validation) :", cv_scores.mean())

"""Entrainement"""
knn_best.fit(X_train, y_train)
y_train_pred = knn_best.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy d'entraînement : ", train_accuracy)

"""Test"""
y_test_pred_knn = knn_best.predict(X_test)
test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
print("Accuracy finale sur test set knn : ", test_accuracy_knn)
print("KNN - Rapport de classification :")
print(classification_report(y_test, y_test_pred_knn))



#Decision Tree Classifier
"""Pre-elagage
On cherche les meilleurs parametres comme max_depth, min_samples_split et min_samples_leaf
limite a l'avance la croissance de l'arbre pour eviter qu'il ne devienne trop complexe
max_depth : profondeur maximale de l'arbre
min_samples_split : nombre minimum d'echantillons pour faire un split
min_samples_leaf : nombre minimum d'echantillons dans une feuille
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
"""
Entraine un arbre pour chaque combinaison
Evalue les performances (accuracy moyenne) sur les folds
Selectionne la meilleure combinaison"""
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Meilleurs parametres (pre-elagage) :", grid.best_params_)

"""Post-elagage
simplifie l'arbre apres construction, consiste a reduire la taille de l'arbre apres qu'il a ete completement construit, pour eviter ce sur-apprentissage.
ccp_alpha est le parametre de complexite utilise pour decider combien de branches supprimer dans l'arbre lors du post-pruning.
On genere la courbe d'elagage en entrainant l'arbre pour plusieurs valeurs de ccp_alpha
"""
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # On exclut le dernier alpha (arbre vide)
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

"""Tracer la courbe de validation pour ccp_alpha"""

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
train_scores = [clf.score(X_train, y_train) for clf in clfs]
val_scores = [np.mean(cross_val_score(clf, X_train, y_train, cv=5)) for clf in clfs]
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='Train accuracy')
plt.plot(ccp_alphas, val_scores, marker='o', label='Validation accuracy')
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Post-pruning via ccp_alpha")
plt.grid()
plt.show()

"""Choisir le meilleur ccp_alpha"""
best_alpha = ccp_alphas[np.argmax(val_scores)]
print("Meilleur ccp_alpha :", best_alpha)

"""Modele final"""
final_model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha, **grid.best_params_)
final_model.fit(X_train, y_train)
"""-> L'arbre complet est deja le meilleur selon la validation croisee."""
"""Evaluation"""
from sklearn.metrics import accuracy_score
y_pred = final_model.predict(X_test)
print("Accuracy finale sur test :", accuracy_score(y_test, y_pred))
print("Decision Tree - Rapport de classification :")
print(classification_report(y_test, y_pred))
cv_tree = cross_val_score(final_model, X_train, y_train, cv=5)
print("Decision Tree - Moyenne validation croisée :", cv_tree.mean())
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
y_prob = final_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.grid()
plt.show()



#Comparaison
"""
Les deux modeles (KNN et Decision Tree) ont donne les memes resultats parfaits.
Meme Accuracy finale sur test : 1.0
Meme Rapport de classification
Meme Moyenne validation croisee : 1.0"""



#Pipeline KNN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
X = data.drop('class', axis=1)
y = data['class']
le = LabelEncoder()
y = le.fit_transform(y)
colonnes_a_supprimer = ['cap-color', 'veil-color', 'gill-attachment', 'veil-type','odor']
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.columns)
"""Pipeline de pretraitement"""
preprocessing = Pipeline([
    ('drop_cols', DropColumns(columns=colonnes_a_supprimer)),
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder())
])
"""Pipeline complet avec KNN"""
pipeline_knn = Pipeline([
    ('preprocessing', preprocessing),
    ('knn', KNeighborsClassifier(n_neighbors=1, metric='manhattan'))
])
"""Train/test split"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""Entraînement"""
pipeline_knn.fit(X_train, y_train)
"""Prédiction"""
y_pred_knn = pipeline_knn.predict(X_test)



#Pipline Arbre
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
"""Separer X et y"""
X = data.drop('class', axis=1)
y = data['class']
le = LabelEncoder()
y = le.fit_transform(y)
"""Colonnes a supprimer"""
colonnes_a_supprimer = ['cap-color', 'veil-color', 'gill-attachment', 'veil-type','odor']
"""Creation d'un transformeur personnalise pour supprimer des colonnes"""
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.columns)
"""Pipeline de pretraitement"""
preprocessing = Pipeline([
    ('drop_cols', DropColumns(columns=colonnes_a_supprimer)),
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder())
])
"""Pipeline complet avec Decision Tree"""
pipeline_dt = Pipeline([
    ('preprocessing', preprocessing),
    ('tree', DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=2,
        ccp_alpha=0.0,
        random_state=42
    ))
])
"""Train/test split"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""Entrainement"""
pipeline_dt.fit(X_train, y_train)
"""Prediction"""
y_pred_dt = pipeline_dt.predict(X_test)



#Evaluation apres Pipline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def evaluate_model(name, y_test, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion : {name}')
    plt.xlabel('Predit')
    plt.ylabel('Reel')
    plt.show()
"""evaluation des deux modeles"""
evaluate_model("KNN", y_test, y_pred_knn)
evaluate_model("Decision Tree", y_test, y_pred_dt)