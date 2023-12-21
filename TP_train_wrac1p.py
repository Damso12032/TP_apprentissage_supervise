from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import pickle
import sys
from IPython.display import display
from sklearn.preprocessing import OneHotEncoder #pour le recodage des variables catégorielles 
from sklearn.preprocessing import MinMaxScaler # pour effectuer la normalisations min-max
from sklearn.compose import make_column_transformer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

with open('output_tp.txt', 'w') as f:
    # Rediriger la sortie standard vers le fichier
    sys.stdout = f

    ###Lecture et choix d'une part du dataset
    features= pd.read_csv("./acsincome_ca_features.csv")
    labels= pd.read_csv("./acsincome_ca_labels.csv")
    features = features.drop(columns=['RAC1P'])
    print("feature SEX enlevé du dataset")
    X_all=features.values[:]
    y_all=labels.values[:]
    X_all, y_all = shuffle(X_all, y_all, random_state=1)
    # only use the first N samples to limit training time
    num_samples = int(len(X_all)*1)
    X, y = X_all[:num_samples], y_all[:num_samples]
    print("Lecture et choix d'une part du dataset done")
    ###Separation en train et test set 
    percentage_train=int(0.8*len(X))
    X_train,y_train=X[:percentage_train],y[:percentage_train].ravel()
    X_test,y_test=X[percentage_train:],y[percentage_train:].ravel()
    print("Separation en train et test set done")
    ### Normalisation du train et test set
    numeric_features = [0,7]
    print(numeric_features)
    categorical_features = [1,2,3,4,5,6,8,9]  # You need to specify your categorical feature indices here
    transformer2 = make_column_transformer(
        (MinMaxScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    )
    X_train_transformed = transformer2.fit_transform(X_train)
    # Sauvegarde des ensembles d'entrainement
    with open('X_train.pkl', 'wb') as file:
        pickle.dump(X_train_transformed, file)
    with open('y_train.pkl', 'wb') as file:
        pickle.dump(y_train, file)    
    print("Normalisation du train set done")
    #chargement du transformer2
    # Enregistrement du transformateur
    with open('transformer2.pkl', 'wb') as file:
        pickle.dump(transformer2, file)
    print("transformer enregistré dans transformer2.pkl")
    X_test_transformed = transformer2.transform(X_test)
    # Sauvegarde des ensembles de test
    with open('X_test.pkl', 'wb') as file:
        pickle.dump(X_test_transformed, file)
    with open('y_test.pkl', 'wb') as file:
        pickle.dump(y_test, file)
    print("Ensemble de test enregistré dans X_test.pkl et y_test.pkl")
    print("Normalisation du test set done")
    k=5
    ########Modele SVM
    ##Mise en place du gridSearchCV avc SVM
    param_grid={'kernel':['linear','poly','rbf','sigmoid']}
    grid=GridSearchCV(svm.SVC(),param_grid,cv=k)
    grid.fit(X_train_transformed,y_train)
    print("Recherche meilleurs params de SVM avec gridSearch done")
    #recuperer les meilleurs scores avec SVM
    best_params=grid.best_params_
    best_score=grid.best_score_
    print(f'Pour SVM, best_params={best_params} and bestscore={best_score}')

    #score du meilleur modele sur l'ensemble de test
    model1=grid.best_estimator_
    test_score1=model1.score(X_test_transformed,y_test)
    accuracy_score1=accuracy_score(y_test,model1.predict(X_test_transformed))
    classification_score1=classification_report(y_test,model1.predict(X_test_transformed))
    print(f'Pour SVM, \n test_score={test_score1}, \n accuracy_core={accuracy_score1}, \n classification_report={classification_score1}')

    #Appliquation du k-cross validation sur le model
    score=cross_val_score(model1,X_train_transformed,y_train,cv=k,scoring='accuracy').mean()
    print("scores du meilleur modele SVM obtenu=",score)
    
    #Calcul de la matrice de confusion
    matrice_confusion=confusion_matrix(y_test,model1.predict(X_test_transformed))
    print("matrice de confusion du meilleur modele de SVM obtenu:",matrice_confusion)

    # Enregistrez le meilleur modèle dans un fichier
    with open('modele_svm.pkl', 'wb') as file:
        pickle.dump(model1, file)
    print("enregistrement du meilleur modele SVM done")

    ########Modele RandomForest
    ##Mise en place du gridSearchCV avc RandomForest
    param_grid2={'criterion':['entropy','gini','log_loss'],'max_depth':np.arange(6,11),'min_samples_leaf':np.arange(1,4)}
    grid2=GridSearchCV(RandomForestClassifier(),param_grid2,cv=k)
    grid2.fit(X_train_transformed,y_train)
    print("Recherche meilleurs params de RandomForest avec gridSearch done")
    #recuperer les meilleurs scores avec RandomForest
    best_params2=grid2.best_params_
    best_score2=grid2.best_score_
    print(f'Pour RandomForest, best_params={best_params2} and bestscore={best_score2}')
    
    #score du meilleur modele sur l'ensemble de test
    model2=grid2.best_estimator_
    test_score2=model2.score(X_test_transformed,y_test)
    accuracy_score2=accuracy_score(y_test,model2.predict(X_test_transformed))
    classification_score2=classification_report(y_test,model2.predict(X_test_transformed))
    print(f'Pour RandomForest, \n test_score={test_score2}, \n accuracy_core={accuracy_score2}, \n classification_report={classification_score2}')
            
    #Appliquation du k-cross validation sur le model
    score=cross_val_score(model2,X_train_transformed,y_train,cv=k,scoring='accuracy').mean()
    print("scores du meilleur modele de RandomForest obtenu=",score)
    
    #Calcul de la matrice de confusion
    matrice_confusion=confusion_matrix(y_test,model2.predict(X_test_transformed))
    print("matrice de confusion du meilleur modele de RandomForest obtenu:",matrice_confusion)

    # Enregistrez le meilleur modèle dans un fichier
    with open('modele_randomForest.pkl', 'wb') as file:
        pickle.dump(model2, file)
    print("enregistrement du meilleur modele RandomForest done")

    ########Modele AdaBoostClassifier
    ##Mise en place du gridSearchCV avc AdaBoostClassifier
    param_grid3={'n_estimators':np.arange(25,250,25),'algorithm':['SAMME', 'SAMME.R']}
    grid3=GridSearchCV(AdaBoostClassifier(),param_grid3,cv=k)
    grid3.fit(X_train_transformed,y_train)
    print("Recherche meilleurs params de AdaBoostClassifier avec gridSearch done")
    ##recuperer les meilleurs scores avec AdaBoostClassifier
    best_params3=grid3.best_params_
    best_score3=grid3.best_score_
    print(f'Pour AdaBoostClassifier, best_params={best_params3} and bestscore={best_score3}')

    #score du meilleur modele sur l'ensemble de test
    model3=grid3.best_estimator_
    test_score3=model3.score(X_test_transformed,y_test)
    accuracy_score3=accuracy_score(y_test,model3.predict(X_test_transformed))
    classification_score3=classification_report(y_test,model3.predict(X_test_transformed))
    print(f'Pour AdaBoost, \n test_score={test_score3}, \n accuracy_core={accuracy_score3}, \n classification_report={classification_score3}')
        
    #Appliquation du k-cross validation sur le model
    score=cross_val_score(model3,X_train_transformed,y_train,cv=k,scoring='accuracy').mean()
    print("scores du meilleur modele de AdaBoostClassifier obtenu=",score)
    
    #Calcul de la matrice de confusion
    matrice_confusion=confusion_matrix(y_test,model3.predict(X_test_transformed))
    print("matrice de confusion du meilleur modele de AdaBoostClassifier obtenu:",matrice_confusion)

    # Enregistrez le meilleur modèle dans un fichier
    with open('modele_AdaBoostClassifier.pkl', 'wb') as file:
        pickle.dump(model3, file)
    print("enregistrement du meilleur modele AdaBoostClassifier done")
    
    ########Modele GradienBoostingClassifier
    ##Mise en place du gridSearchCV avc GradienBoostingClassifier
    param_grid4={'loss':['log_loss','exponential'],'criterion':['friedman_mse', 'squared_error'],'n_estimators':np.arange(25,250,25)}
    grid4=GridSearchCV(GradientBoostingClassifier(),param_grid4,cv=k)
    grid4.fit(X_train_transformed,y_train)
    print("Recherche meilleurs params de GradienBoostingClassifier avec gridSearch done")
    #recuperer les meilleurs scores avec GradienBoostingClassifier
    best_params4=grid4.best_params_
    best_score4=grid4.best_score_
    print(f'Pour GradienBoostingClassifier, best_params={best_params4} and bestscore={best_score4}')
    
    #score de chaque meilleur modele sur l'ensemble de test
    model4=grid4.best_estimator_
    test_score4=model4.score(X_test_transformed,y_test)
    accuracy_score4=accuracy_score(y_test,model4.predict(X_test_transformed))
    classification_score4=classification_report(y_test,model4.predict(X_test_transformed))
    print(f'Pour GradienBoostingClassifier, \n test_score={test_score4}, \n accuracy_core={accuracy_score4}, \n classification_report={classification_score4}')
    
    #Appliquation du k-cross validation sur le model
    score=cross_val_score(model4,X_train_transformed,y_train,cv=k,scoring='accuracy').mean()
    print("scores du meilleur modele de GradienBoostingClassifier obtenu=",score)
    
    #Calcul de la matrice de confusion
    matrice_confusion=confusion_matrix(y_test,model4.predict(X_test_transformed))
    print("matrice de confusion du meilleur modele de GradienBoostingClassifier obtenu:",matrice_confusion)

    # Enregistrez le meilleur modèle dans un fichier
    with open('modele_GradienBoostingClassifier.pkl', 'wb') as file:
        pickle.dump(model4, file)  
    print("enregistrement du meilleur modele GradienBoostingClassifier done")
    
    
    sys.stdout = sys.__stdout__
