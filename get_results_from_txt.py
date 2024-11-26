import numpy as np
from pathlib import Path
import json
import scipy.signal as signal

depend = 'val'
seeds=[0,10,20,30,40]
path = Path('your result path')
name='your name'

metric= ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
main_metric= 'cohen_kappa'

# metric=  ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy",'f1','precision','recall']
# main_metric= 'roc_auc'

result_dict, val_result_dict = {}, {}
for i in metric:
    result_dict[i] = []
    val_result_dict[i] = []

for seed in seeds:
    best_epoch = 0
    best_dict, dict_temp, val_dict = {}, {}, {}
    for i in metric:
        best_dict[i] = 0.0
        val_dict[i] = 0.0
    r_path = path/name/f'seed_{seed}'/'model'/'log_0.txt'
    file = open(r_path,'r')

    for lineid, line in enumerate(file.readlines()):
        line = line.strip() 
        its = line.split(', ')
        for it in its:
            k = it.split(': ')[0]
            v = it.split(': ')[1]
            if 'unused_code' not in k: 
                if '}' in  v:
                    v = v.replace('}','')           
                v = float(v)
            dict_temp[k] = v
        # print(dict_temp)

        if dict_temp['"'+depend+'_'+main_metric+'"'] >= val_dict[main_metric]:
            if abs(best_epoch -dict_temp['{"epoch"'])>=50:
                break
            else:
                best_epoch = dict_temp['{"epoch"']
                print(dict_temp['{"epoch"'], dict_temp['"'+depend+'_'+main_metric+'"'], best_dict[main_metric])
                for k in best_dict.keys():
                    best_dict[k] = dict_temp['"'+'test_'+k+'"']
                    val_dict[k] = dict_temp['"'+'val_'+k+'"']
    

    file.close()
    
    for i in metric:
        result_dict[i].append(best_dict[i])
        val_result_dict[i].append(val_dict[i])

# print(val_result_dict)
        
for i in metric:
    print('test name: {} index:{}, {:.2f}/{:.2f}'.format(name, i, np.mean(result_dict[i])*100,np.std(result_dict[i])*100))
for i in metric:
    print('val name: {} index:{}, {:.2f}/{:.2f}'.format(name, i, np.mean(val_result_dict[i])*100,np.std(val_result_dict[i])*100))

ttt = [round(i*100,2) for i in result_dict[main_metric]]
print(f'name {name} test main_metric {main_metric}, results {ttt}')

ttt = [round(i*100,2) for i in val_result_dict[main_metric]]
print(f'name {name} val main_metric {main_metric}, results {ttt}')

