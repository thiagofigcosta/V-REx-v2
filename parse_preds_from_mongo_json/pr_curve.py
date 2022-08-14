import json
import re
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import roc_auc_score,auc
from matplotlib import pyplot as plt 
import numpy as np



def smooth(y, box_pts,keep_borders=True):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    if keep_borders and len(y_smooth) >=4:
        y_smooth=list(y_smooth)
        return y[:2]+y_smooth[2:-2]+y[-2:]
    else:
        return y_smooth 

graphs_to_build=[
    {'v_rex_preds_file': 'improving_vrtbep-vrex_preds.json', 'baseline': 'improving-vrtbep', 'baseline_pr_file': 'pr_curve_improving-vrtbep.json', 'baseline_roc_file':'roc_curve_improving-vrtbep.json', 'roc_auc': 0.91, 'pr_auc': None, 'smooth': 2},
    {'v_rex_preds_file': 'epss_cvss-vrex_preds.json', 'baseline': 'epss', 'baseline_pr_file': 'pr_curve_epss.json', 'baseline_roc_file':'roc_curve_epss.json', 'roc_auc': 0.87, 'pr_auc': 0.332, 'smooth': 2},
    {'v_rex_preds_file': 'fastembed-vrex_preds.json', 'baseline': 'fastembed-nvd', 'baseline_pr_file': 'pr_curve_fastembed-nvd.json', 'baseline_roc_file':'roc_curve_fastembed-nvd.json', 'roc_auc': 0.832, 'pr_auc': None, 'smooth': 2},
    {'v_rex_preds_file': 'fastembed-vrex_preds.json', 'baseline': 'fastembed-securityfocus', 'baseline_pr_file': 'pr_curve_fastembed-securityfocus.json', 'baseline_roc_file':'roc_curve_fastembed-securityfocus.json', 'roc_auc': 0.912, 'pr_auc': None, 'smooth': 2},
]
ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for graph_to_build in graphs_to_build:
    v_rex_data=json.load(open(graph_to_build['v_rex_preds_file']))
    label = []
    pred = []
    for entry in v_rex_data:
        try:
            re_result = re.search(r"(CVE-[0-9]{4}-[0-9]*): Label: ([0-9]*) \| Predicted Exploit: ([0-9]*) \| Conficende ([0-9\.]*)\% \| Prediction Match: (True|False)", entry, re.IGNORECASE).groups()
            label.append(int(re_result[1]))
            pred.append(float(re_result[3])/100.0)
        except:
            print(f'Could not parse: `{entry}`')
            
            
    baseline_roc_auc = graph_to_build['roc_auc']
    baseline_pr_auc = graph_to_build['pr_auc']
    
    baseline_pr_curve=json.load(open(graph_to_build['baseline_pr_file']))
    baseline_roc_curve=json.load(open(graph_to_build['baseline_roc_file']))
            
    roc_auc = roc_auc_score(label, pred)
    false_pos, true_pos, thresholds = roc_curve(label, pred)
    title = f'V-REx ROC Curve (AUC: {round(roc_auc,3)})'
    print(title)
    print(f'{graph_to_build["baseline"]} ROC AUC: {baseline_roc_auc}')
    plt.plot(false_pos, true_pos, linestyle='--', label='V-REx')
    plt.plot(baseline_roc_curve['x_coords'], smooth(baseline_roc_curve['y_coords'],graph_to_build['smooth']), label='baseline') # graph_to_build['baseline']
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.legend()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(title+' vs '+graph_to_build["baseline"])
    plt.show()
    print()
    
    precision, recall, thresholds = precision_recall_curve(label, pred)
    pr_auc = auc(recall,precision)


    title = f'V-REx PR Curve (AUC: {round(pr_auc,3)})'
    print(title)
    print(f'{graph_to_build["baseline"]} PR AUC: {baseline_pr_auc}')
    plt.plot(recall,precision, linestyle='--', label='V-REx')
    plt.plot(baseline_pr_curve['x_coords'], smooth(baseline_pr_curve['y_coords'],graph_to_build['smooth']), label='baseline') # graph_to_build['baseline']
    plt.title(title)
    plt.xlabel('Recall (Coverage)')
    plt.ylabel('Precision (Efficiency)')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.legend()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(title+' vs '+graph_to_build["baseline"])
    plt.show()
    print()
    print()
