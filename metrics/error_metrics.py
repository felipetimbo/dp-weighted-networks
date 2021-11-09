import numpy as np
import pandas as pd

from metrics import egocentric_metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def calculate(error_metric, y_true, y_pred, k=10):
    if error_metric == 'mae':
        error = mae(y_true, y_pred)
    elif error_metric == 'mre':
        error = mre(y_true, y_pred)
    elif error_metric == 'l2':
        error = l2(y_true, y_pred)
    elif error_metric == 'mape':
        error = mape(y_true, y_pred)
    elif error_metric == 'mse':
        error = rmse(y_true, y_pred)
    elif error_metric == 'rmse':
        error = rmse(y_true, y_pred)
    elif error_metric == 'absolute':
        error = absolute(y_true, y_pred)
    elif error_metric == 'op':
        error = op(y_true, y_pred, k)
    else:
        error = None

    return error  

def mape(y_true, y_pred):  
    mask_not_zeros = y_true > 0       
    error = np.mean( np.abs ( (y_true[mask_not_zeros] - y_pred[mask_not_zeros]) / y_true[mask_not_zeros]))*100
    # msgs.log_msg('MAPE = %f' % error )
    return error

def mae(y_true, y_pred): 
    if len(y_true) > 0:         
        error = np.mean( np.abs ( (y_true - y_pred) ))
    else:
        error = 0
    # msgs.log_msg('MAPE = %f' % error )
    return error

def mre(y_true, y_pred): 
    mask_not_zeros = y_true > 0       
    error = np.mean( np.abs ( (y_pred[mask_not_zeros] - y_true[mask_not_zeros] )) / y_true[mask_not_zeros] )
    # msgs.log_msg('MRE = %f' % error )
    return error

def l2(y_true, y_pred):          
    error = np.sum(np.power((y_true - y_pred),2))
    return error

def mse(y_true, y_pred):
    if len(y_true) > 0:
        error = mean_squared_error( np.array(y_true), np.array(y_pred) )
    # msgs.log_msg('mse = %f' % error )
    else:
        error = 0
    return error

def rmse(y_true, y_pred):
    ms_error = mse(y_true, y_pred)
    error = np.sqrt(ms_error)
    # msgs.log_msg('rmse = %f' % error )
    return error

def absolute(y_true, y_pred):
    error = mean_absolute_error( np.array(y_true), np.array(y_pred) )
    # msgs.log_msg('absolute error = %f' % error )
    return error

# overlapping percentage
def op(y_true, y_pred, k):
    topk_true = np.argsort(-y_true)[:k]
    topk_pred = np.argsort(-y_pred)[:k]
    error = len(set(topk_true) & set(topk_pred)) / k
    # msgs.log_msg('overlapping percentage = %f' % error )
    return error

def recall(y_true_pos, y_pred_pos):
    error = len(set(y_true_pos).intersection(set(y_pred_pos))) / len(set(y_true_pos))
    # error = len(set(topk_true) & set(topk_pred))
    # msgs.log_msg('overlapping percentage = %f' % error )
    return error

def jaccard_distance(y_true, y_pred, num_zeros):
    num_zeros_to_be_added_in_y_pred = num_zeros + len(y_true) - len(y_pred)

    x = np.nonzero(y_true==0)[0]  # y_true[y_true==0]  # np.nonzero(y_true==0)[0]
    y = np.nonzero(y_pred==0)[0] # y_pred[y_pred==0] # 
    intersection_cardinality = len(set(x).intersection(set(y))) + num_zeros_to_be_added_in_y_pred
    union_cardinality = len(set(x).union(set(y))) + num_zeros_to_be_added_in_y_pred
    distance = intersection_cardinality / float(union_cardinality)
    return distance

def jaccard_distance_non_zeros(y_true, y_pred):
    x = np.nonzero(y_true!=0)[0]  # y_true[y_true==0]  # np.nonzero(y_true==0)[0]
    y = np.nonzero(y_pred!=0)[0] # y_pred[y_pred==0] # 
    intersection_cardinality = len(set(x).intersection(set(y)))
    union_cardinality = len(set(x).union(set(y)))
    distance = intersection_cardinality / float(union_cardinality)
    return distance

def calculate_error_edges_w_ego( error_metric, edges_true_arr, edges_pred_arr ):

    errors = []
    
    for i in range (len(edges_true_arr)):

        edges_true = edges_true_arr[i]
        edges_pred = edges_pred_arr[i]

        df_true = pd.DataFrame(edges_true, columns=['s', 'd', 'ewt'])
        df_pred = pd.DataFrame(edges_pred, columns=['s', 'd', 'ewp'])
        
        if error_metric == 'mre':
            df_merge_left = pd.merge(df_true, df_pred, how="left", on=["s","d"])
            y_true = df_merge_left['ewt'].to_numpy()
            y_pred = df_merge_left['ewp'].fillna(0).to_numpy()
        
        else:
            df_merge_right = pd.merge(df_true, df_pred, how="right", on=["s","d"])
            y_true = df_merge_right['ewt'].fillna(0).to_numpy()
            y_pred = df_merge_right['ewp'].to_numpy()

        errors.append(calculate(error_metric, y_true, y_pred))

    return np.mean(errors)


def calculate_error_edges_w( error_metric, edges_true, edges_pred ):
    df_true = pd.DataFrame(edges_true, columns=['s', 'd', 'ewt'])
    df_pred = pd.DataFrame(edges_pred, columns=['s', 'd', 'ewp'])
     
    if error_metric == 'mre':
        df_merge_left = pd.merge(df_true, df_pred, how="left", on=["s","d"])
        y_true = df_merge_left['ewt'].to_numpy()
        y_pred = df_merge_left['ewp'].fillna(0).to_numpy()
        error = calculate(error_metric, y_true, y_pred)
    
    else:
        df_merge_right = pd.merge(df_true, df_pred, how="right", on=["s","d"])
        y_true = df_merge_right['ewt'].fillna(0).to_numpy()
        y_pred = df_merge_right['ewp'].to_numpy()
        error = calculate(error_metric, y_true, y_pred)

    return error

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

    # num_zeros_to_be_added = np.pad(edges_w_pred, (0, len(all_edges_w_true) - len(edges_w_pred) ), 'constant')

    # source_equals_mask = df_concat['st'] == df_concat['sp']
    # destination_equals_mask = df_concat['dt'] == df_concat['dp']
    # same_edges = source_equals_mask & destination_equals_mask

# def mape_laplace_mec(optins_v, optins_v_perturbed):
#     error = np.mean(np.abs((optins_v - optins_v_perturbed) / optins_v)) 
#     # error = mean_absolute_error( np.array(optins_v), np.array(optins_v_perturbed) )
#     # msgs.log_msg('MAPE laplace = ' + str(error) )
#     return error


# Mean absolute error

# def absolute_error_levels_mec(G, g, metric):
#     y_true = egocentric_metrics.calculate(G, metric)
#     y_pred = egocentric_metrics.calculate(g, metric)

#     error = mean_absolute_error(y_true, y_pred)

#     msgs.log_msg('absolute error ' + metric + ' = ' + str(error) )
#     return error

# Relative mean square error

# def rmse_levels_mec(G, g, metric):

#     predictions = egocentric_metrics.calculate(g, metric)
#     targets = egocentric_metrics.calculate(G, metric)

#     error = calculate_error(predictions, targets) 

#     msgs.log_msg('rmse proposed mechanism ' + metric + ' = ' + str(error) )
#     return error

