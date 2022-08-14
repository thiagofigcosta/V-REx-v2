import json
import re
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import roc_auc_score,auc
from matplotlib import pyplot as plt 

filename = 'epss_cvss.json'

raw_data=json.load(open(filename))

label = []
pred = []
for entry in raw_data:
    try:
        re_result = re.search(r"(CVE-[0-9]{4}-[0-9]*): Label: ([0-9]*) \| Predicted Exploit: ([0-9]*) \| Conficende ([0-9\.]*)\% \| Prediction Match: (True|False)", entry, re.IGNORECASE).groups()
        label.append(int(re_result[1]))
        pred.append(float(re_result[3])/100.0)
    except:
        print(f'Could not parse: `{entry}`')

roc_auc = roc_auc_score(label, pred)
true_pos, false_pos, thresholds = roc_curve(label, pred)

ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

print(f'ROC AUC: {roc_auc}')
plt.plot(precision, recall, linestyle='--', label='V-REx')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xticks(ticks)
plt.yticks(ticks)
plt.legend()
plt.show()

precision, recall, thresholds = precision_recall_curve(label, pred)
pr_auc = auc(recall,precision)


print(f'PR AUC: {pr_auc}')
plt.plot(recall,precision, linestyle='--', label='V-REx')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xticks(ticks)
plt.yticks(ticks)
plt.legend()
plt.show()