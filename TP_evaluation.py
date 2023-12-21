import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
#on necupere les infos

with open('modele_svm_wsex.pkl', 'rb') as file:
    model_SVM = pickle.load(file)
with open('transformer2_wsex.pkl', 'rb') as file:
    transformer2=pickle.load(file)
with open('modele_randomForest_wsex.pkl', 'rb') as file:
    model_RandomForest = pickle.load(file)
with open('modele_AdaBoostClassifier_wsex.pkl', 'rb') as file:
    model_AdaBoost = pickle.load(file)
with open('modele_GradienBoostingClassifier_wsex.pkl', 'rb') as file:
    model_GradientBoost = pickle.load(file)

features= pd.read_csv("./acsincome_ca_features.csv")
labels= pd.read_csv("./acsincome_ca_labels.csv")
X_all=features.values[:]
y_all=labels.values[:]
X_all, y_all = shuffle(X_all, y_all, random_state=1)

num_samples = int(len(X_all)*0.1)
X, y = X_all[:num_samples], y_all[:num_samples]
percentage_train=int(0.8*len(X))
X_train,y_train=X[:percentage_train],y[:percentage_train].ravel()
X_test,y_test=X[percentage_train:],y[percentage_train:].ravel()
# #calculez les corrélations entre chacune des features et le label à  partir des données d’entrainement
# X_all=features.values[:]
# y_all=labels.values[:]
# X_all, y_all = shuffle(X_all, y_all, random_state=1)

# num_samples = int(len(X_all)*0.1)
# X, y = X_all[:num_samples], y_all[:num_samples]
# percentage_train=int(0.8*len(X))
# X_train,y_train=X[:percentage_train],y[:percentage_train].ravel()
# X_test,y_test=X[percentage_train:],y[percentage_train:].ravel()

# df_train = pd.DataFrame(X_train, columns=features.columns)  
# df_train['label'] = y_train  # Ajouter la colonne de l'étiquette
# correlations_train = df_train.corr()['label'].drop('label')  # On exclut la corrélation avec elle-même
# print("correlations entre features et label dataset==>",correlations_train)
# df_test = pd.DataFrame(X_test, columns=features.columns)  
# df_test['label_SVM'] = model_SVM.predict(transformer2.transform(X_test)) # Ajouter la colonne de l'étiquette
# df_test['label_randomForest']=model_RandomForest.predict(transformer2.transform(X_test))
# df_test['label_AdaBoostClassifier']=model_AdaBoost.predict(transformer2.transform(X_test))
# df_test['label_GradientBoost']=model_GradientBoost.predict(transformer2.transform(X_test))

# correlations_train_SVM = df_test.corr()['label_SVM'].drop(['label_SVM','label_randomForest','label_GradientBoost','label_AdaBoostClassifier'])  # On exclut la corrélation avec elle-même
# correlations_train_randomFor = df_test.corr()['label_randomForest'].drop(['label_SVM','label_randomForest','label_GradientBoost','label_AdaBoostClassifier'])  # On exclut la corrélation avec elle-même
# correlations_train_AdaBoost = df_test.corr()['label_AdaBoostClassifier'].drop(['label_SVM','label_randomForest','label_GradientBoost','label_AdaBoostClassifier'])  # On exclut la corrélation avec elle-même
# correlations_train_GradientBoost = df_test.corr()['label_GradientBoost'].drop(['label_SVM','label_randomForest','label_GradientBoost','label_AdaBoostClassifier'])  # On exclut la corrélation avec elle-même
# print("correlations entre features et label SVM dataset==>",correlations_train_SVM)
# print("correlations entre features et label randomFor dataset==>",correlations_train_randomFor)
# print("correlations entre features et label AdaBoost dataset==>",correlations_train_AdaBoost)
# print("correlations entre features et label GradientBoost dataset==>",correlations_train_GradientBoost)





###evaluation de l'importance des features
#SVM

# result1 = permutation_importance(model_SVM, transformer2.transform(X_test).toarray(), y_test, n_repeats=10,
#                                  random_state=0)
# importances_mean_wsex1=result1.importances_mean
# print(f'Pour SVM, \n importances_mean_wsex={importances_mean_wsex1}')

# #RandomForest
# result2 = permutation_importance(model_RandomForest, transformer2.transform(X_test).toarray(), y_test, n_repeats=10,
#                                  random_state=0)
# importances_mean_wsex2=result2.importances_mean
# print(f'Pour RandomForest, \n importances_mean_={importances_mean_wsex2}')

# #AdaBoost
# result3 = permutation_importance(model_AdaBoost, transformer2.transform(X_test).toarray(), y_test, n_repeats=10,
#                                  random_state=0)
# importances_mean_wsex3=result3.importances_mean
# print(f'Pour AdaBoosting, \n importances_mean_={importances_mean_wsex3}')

# #GradientBoosting
# result4 = permutation_importance(model_GradientBoost, transformer2.transform(X_test).toarray(), y_test, n_repeats=10,
#                                  random_state=0)
# importances_mean_wsex4=result4.importances_mean
# print(f'Pour GradientBoosting, \n importances_mean_={importances_mean_wsex4}')


##Matrice de confusion pour le sex 
gender_column = X_test[:,8]
# Définir les indices pour les deux groupes
hommes_indices = [i for i in range(len(gender_column)) if int(gender_column[i])==1]
femmes_indices = [i for i in range(len(gender_column)) if int(gender_column[i])==2]
y_test_hommes = y_test[hommes_indices]
y_test_femmes=y_test[femmes_indices]
# Sélectionnez les prédictions et les étiquettes réelles pour cette caractéristique
y_pred_hommes=model_SVM.predict(transformer2.transform(X_test[hommes_indices,:]))
y_pred_femmes=model_SVM.predict(transformer2.transform(X_test[femmes_indices,:]))
# conf_matrix_hommes = confusion_matrix(y_test_hommes, y_pred_hommes)
# conf_matrix_femmes = confusion_matrix(y_test_femmes, y_pred_femmes)
# print("Matrice de Confusion SVM -->hommes==>",conf_matrix_hommes)
# print("Matrice de Confusion SVM -->femmes==>",conf_matrix_femmes)
# # Afficher la matrice de confusion avec seaborn
# sns.heatmap(conf_matrix_hommes, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.xlabel("Prédictions")
# plt.ylabel("Réelles")
# plt.title("Matrice de Confusion SVM pour les hommes")
# plt.show()
# sns.heatmap(conf_matrix_femmes, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Matrice de Confusion SVM pour les femmes")
# plt.show()

#Calcul de metriques
# Calcul de l'exactitude pour chaque groupe démographique
y_train_pred=model_SVM.predict(transformer2.transform(X_train))
y_test_pred=model_SVM.predict(transformer2.transform(X_test))
accuracy_train_hommes = accuracy_score(y_train[hommes_indices], y_train_pred[hommes_indices])
accuracy_train_femmes = accuracy_score(y_train[femmes_indices], y_train_pred[femmes_indices])

accuracy_test_hommes = accuracy_score(y_test[hommes_indices], y_test_pred[hommes_indices])
accuracy_test_femmes = accuracy_score(y_test[femmes_indices], y_test_pred[femmes_indices])

# Calcul de la disparité d'impact
impact_disparity_train = abs(accuracy_train_hommes - accuracy_train_femmes)
impact_disparity_test = abs(accuracy_test_hommes - accuracy_test_femmes)

print("Disparité d'impact avec SVM(entraînement) :", impact_disparity_train)
print("Disparité d'impact avec SVM(test) :", impact_disparity_test)





y_pred_hommes=model_RandomForest.predict(transformer2.transform(X_test[hommes_indices,:]))
y_pred_femmes=model_RandomForest.predict(transformer2.transform(X_test[femmes_indices,:]))
# conf_matrix_hommes = confusion_matrix(y_test_hommes, y_pred_hommes)
# conf_matrix_femmes = confusion_matrix(y_test_femmes, y_pred_femmes)
# print("Matrice de Confusion RandomForest -->hommes==>",conf_matrix_hommes)
# print("Matrice de Confusion RandomForest -->femmes==>",conf_matrix_femmes)
# # Afficher la matrice de confusion avec seaborn
# sns.heatmap(conf_matrix_hommes, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.xlabel("Prédictions")
# plt.ylabel("Réelles")
# plt.title("Matrice de Confusion RandomForest pour les hommes")
# plt.show()
# sns.heatmap(conf_matrix_femmes, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Matrice de Confusion RandomForest pour les femmes")
# plt.show()

#Calcul de metriques
# Calcul de l'exactitude pour chaque groupe démographique
y_train_pred=model_RandomForest.predict(transformer2.transform(X_train))
y_test_pred=model_RandomForest.predict(transformer2.transform(X_test))
accuracy_train_hommes = accuracy_score(y_train[hommes_indices], y_train_pred[hommes_indices])
accuracy_train_femmes = accuracy_score(y_train[femmes_indices], y_train_pred[femmes_indices])

accuracy_test_hommes = accuracy_score(y_test[hommes_indices], y_test_pred[hommes_indices])
accuracy_test_femmes = accuracy_score(y_test[femmes_indices], y_test_pred[femmes_indices])

# Calcul de la disparité d'impact
impact_disparity_train = abs(accuracy_train_hommes - accuracy_train_femmes)
impact_disparity_test = abs(accuracy_test_hommes - accuracy_test_femmes)

print("Disparité d'impact avec RandomForest(entraînement) :", impact_disparity_train)
print("Disparité d'impact avec RandomForest(test) :", impact_disparity_test)




y_pred_hommes=model_AdaBoost.predict(transformer2.transform(X_test[hommes_indices,:]))
y_pred_femmes=model_AdaBoost.predict(transformer2.transform(X_test[femmes_indices,:]))
conf_matrix_hommes = confusion_matrix(y_test_hommes, y_pred_hommes)
conf_matrix_femmes = confusion_matrix(y_test_femmes, y_pred_femmes)
# print("Matrice de Confusion AdaBoost -->hommes==>",conf_matrix_hommes)
# print("Matrice de Confusion AdaBoost -->femmes==>",conf_matrix_femmes)
# # Afficher la matrice de confusion avec seaborn
# sns.heatmap(conf_matrix_hommes, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.xlabel("Prédictions")
# plt.ylabel("Réelles")
# plt.title("Matrice de Confusion AdaBoost pour les hommes")
# plt.show()
# sns.heatmap(conf_matrix_femmes, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Matrice de Confusion AdaBoost pour les femmes")
# plt.show()

#Calcul de metriques
# Calcul de l'exactitude pour chaque groupe démographique
y_train_pred=model_AdaBoost.predict(transformer2.transform(X_train))
y_test_pred=model_AdaBoost.predict(transformer2.transform(X_test))
accuracy_train_hommes = accuracy_score(y_train[hommes_indices], y_train_pred[hommes_indices])
accuracy_train_femmes = accuracy_score(y_train[femmes_indices], y_train_pred[femmes_indices])

accuracy_test_hommes = accuracy_score(y_test[hommes_indices], y_test_pred[hommes_indices])
accuracy_test_femmes = accuracy_score(y_test[femmes_indices], y_test_pred[femmes_indices])

# Calcul de la disparité d'impact
impact_disparity_train = abs(accuracy_train_hommes - accuracy_train_femmes)
impact_disparity_test = abs(accuracy_test_hommes - accuracy_test_femmes)

print("Disparité d'impact avec AdaBoost(entraînement) :", impact_disparity_train)
print("Disparité d'impact avec AdaBoost(test) :", impact_disparity_test)




y_pred_hommes=model_GradientBoost.predict(transformer2.transform(X_test[hommes_indices,:]))
y_pred_femmes=model_GradientBoost.predict(transformer2.transform(X_test[femmes_indices,:]))
conf_matrix_hommes = confusion_matrix(y_test_hommes, y_pred_hommes)
conf_matrix_femmes = confusion_matrix(y_test_femmes, y_pred_femmes)
# print("Matrice de Confusion GradientBoost -->hommes==>",conf_matrix_hommes)
# print("Matrice de Confusion GradientBoost -->femmes==>",conf_matrix_femmes)
# # Afficher la matrice de confusion avec seaborn
# sns.heatmap(conf_matrix_hommes, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.xlabel("Prédictions")
# plt.ylabel("Réelles")
# plt.title("Matrice de Confusion GradientBoost pour les hommes")
# plt.show()
# sns.heatmap(conf_matrix_femmes, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Matrice de Confusion GradientBoost pour les femmes")
# plt.show()
#Calcul de metriques
# Calcul de l'exactitude pour chaque groupe démographique
y_train_pred=model_GradientBoost.predict(transformer2.transform(X_train))
y_test_pred=model_GradientBoost.predict(transformer2.transform(X_test))
accuracy_train_hommes = accuracy_score(y_train[hommes_indices], y_train_pred[hommes_indices])
accuracy_train_femmes = accuracy_score(y_train[femmes_indices], y_train_pred[femmes_indices])

accuracy_test_hommes = accuracy_score(y_test[hommes_indices], y_test_pred[hommes_indices])
accuracy_test_femmes = accuracy_score(y_test[femmes_indices], y_test_pred[femmes_indices])

# Calcul de la disparité d'impact
impact_disparity_train = abs(accuracy_train_hommes - accuracy_train_femmes)
impact_disparity_test = abs(accuracy_test_hommes - accuracy_test_femmes)

print("Disparité d'impact avec GradientBoost(entraînement) :", impact_disparity_train)
print("Disparité d'impact avec GradientBoost(test) :", impact_disparity_test)


##Matrice de confusion pour le RAC1P 




