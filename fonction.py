import numpy as np
import pandas as pd

def test_length_median(data, df_billets, var_1, var_2) : 
    """ 
    Fonction permettant de prédire la véracité d'un billet en calculant la distance entre la mediane des vrais et faux billets.
    """
    
    df_billets_false = df_billets.loc[df_billets['is_genuine']==False]
    df_billets_true = df_billets.loc[df_billets['is_genuine']==True]

    
    data['median_true_length'] = df_billets_true[var_1].median()
    data['median_false_length'] = df_billets_false[var_1].median()
    data['median_true_margin_low'] = df_billets_true[var_2].median()
    data['median_false_margin_low'] = df_billets_false[var_2].median()
    
    #Calcul de la distance entre la valeur à tester et la mediane des faux billets
    data['result_false_length'] = abs(data['median_false_length'] - data[var_1])
    data['result_false_margin_low'] = abs(data['median_false_margin_low'] - data[var_2])
    
    #Calcul de la distance entre la valeur à tester et la mediane des vrais billets
    data['result_true_length'] = abs(data['median_true_length'] - data[var_1])
    data['result_true_margin_low'] = abs(data['median_true_margin_low'] - data[var_2])
    
    #Résultat en fonction de la longueur du billet
    data['is_genuine_1'] = np.where(data['result_false_length'] > data['result_true_length'], True, False)
    #Résultat en fonction de la marge inferieure du billet
    data['is_genuine_2'] = np.where(data['result_false_margin_low'] > data['result_true_margin_low'], True, False)
        
    
    return(data[['diagonal','height_left','height_right','margin_low','margin_up','length','id', 'is_genuine_1','is_genuine_2']].set_index('id'))




def test_billets_reg_log(test_file, model_reg_log) : 
    """
    Fonction permettant de prédire la véracité d'un billet à l'aide d'une regression logistique.
    """
    data = pd.read_csv(test_file)
    
    
    # def X
    X_test = data.iloc[:, :-1]
    
    # result of the model
    reg_log_pr = model_reg_log.predict(X_test)
    reg_log_proba = model_reg_log.predict_proba(X_test)[:,1]
    
    
    data["predict_proba"]=reg_log_proba
    data["is_genuine"]=reg_log_pr
    
    return(data[['diagonal','height_left','height_right','margin_low','margin_up','length','id', 'predict_proba', 'is_genuine']].set_index('id'))



def test_billets_kmeans(test_file, kmeans) : 
    
    """
    Fonction permettant de prédire la véracité d'un billet à l'aide de kmeans.
    """
    
    data = pd.read_csv(test_file)
    
    # def X
    X_test = data.iloc[:, :-1]
    
    # result of the model
    kmeans_pr = kmeans.predict(X_test)
    
    data["is_genuine"]=kmeans_pr
    data['is_genuine'] = data['is_genuine'].replace(to_replace=[0, 1], value =[True, False])
    
    return(data[['diagonal','height_left','height_right','margin_low','margin_up','length','id', 'is_genuine']].set_index('id'))







