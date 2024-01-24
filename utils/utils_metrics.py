"""Define metrics"""
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

def earth_mover_distance(points_gt, points_pred):
    d = distance.cdist(points_gt, points_pred)
    
    assignment = linear_sum_assignment(d)
    emd = d[assignment].sum() / len(d)
    
    return emd
