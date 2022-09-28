from collections import Counter
import torch
import pandas as pd
import os
import sys
import re
from tqdm import tqdm
import csv
from datetime import datetime
import random



def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')



def check_data(base_path):
    try:
        assert os.path.isfile(base_path+'data/CHARTEVENTS.csv')
    except:
        if os.path.isfile(base_path+'data/CHARTEVENTS.csv.gz'):
            print('\n'+'*'*41)
            print("Error: Unzip the CHARTEVENTS.csv.gz data")
            print('*'*41+'\n')
        else:
            print('\n'+'*'*36)
            print("Error: Check the CHARTEVENTS data")
            print('*'*36+'\n')
        sys.exit()

    

def dict4icuPatient(path, min_los=1, max_los=2):
    ICU_patients = {}
    ICU_stay = pd.read_csv(path)
    ICU_stay_df = pd.DataFrame(ICU_stay)

    ICU_stay_ID = ICU_stay_df['ICUSTAY_ID'].tolist()
    ICU_subject_ID = ICU_stay_df['SUBJECT_ID'].tolist()
    ICU_hadm_ID = ICU_stay_df['HADM_ID'].tolist()
    ICU_stay_LOS = ICU_stay_df['LOS'].round(4).tolist()
    ICU_in = ICU_stay_df['INTIME'].tolist()
    ICU_out = ICU_stay_df['OUTTIME'].tolist()

    # for sanity check
    assert len(ICU_stay_ID) == len(ICU_subject_ID) == len(ICU_hadm_ID) == len(ICU_stay_LOS) == len(ICU_in) == len(ICU_out)

    # filtering the patients by LOS
    for icu_stay_id, icu_hadm_id, icu_subject_id, icu_los, icu_in, icu_out in tqdm(zip(ICU_stay_ID, ICU_hadm_ID, ICU_subject_ID, ICU_stay_LOS, ICU_in, ICU_out), desc='Preprocess the ICU data..'):
        if min_los <= icu_los <= max_los:
            # for sanity check (out_time has to be larger than in_time)
            assert int(''.join(re.findall("\d+", icu_out))) - int(''.join(re.findall("\d+", icu_in))) > 0

            tmp = {}
            tmp['HADM_ID'] = icu_hadm_id; tmp['SUBJECT_ID'] = icu_subject_id; tmp['LOS'] = icu_los
            tmp['ICU_IN'] = icu_in
            tmp['ICU_OUT'] = icu_out
            tmp['X_DATA'] = {}
            ICU_patients[icu_stay_id] = tmp

    return ICU_patients
    


def dict4admissions(path):
    admin_patients = {}
    admission = pd.read_csv(path)
    admission_df = pd.DataFrame(admission)

    admin_hadm_ID = admission_df['HADM_ID'].tolist()
    admin_subject_ID = admission_df['SUBJECT_ID'].tolist()
    admin_death = admission_df['DEATHTIME'].tolist()
    admin_ethnicity = admission_df['ETHNICITY'].tolist()
    admin_type = admission_df['ADMISSION_TYPE'].tolist()
    admin_diagnosis = admission_df['DIAGNOSIS'].tolist()

    # for sanity check
    assert len(admin_hadm_ID) == len(admin_subject_ID) == len(admin_death) == len(admin_ethnicity)
    
    for admin_hadm_id, admin_subject_id, death, ethnicity, type, diagnosis in tqdm(zip(admin_hadm_ID, admin_subject_ID, admin_death, admin_ethnicity, admin_type, admin_diagnosis), desc='Preprocess the ADMISSIONS data..'):
        tmp = {}
        tmp['SUBJECT_ID'] = admin_subject_id
        tmp['DEATHTIME'] = death
        tmp['ETHNICITY'] = ethnicity
        tmp['ADMISSION_TYPE'] = type
        tmp['DIAGNOSIS'] = diagnosis
        admin_patients[admin_hadm_id] = tmp

    return admin_patients



def labeling(ICU_patients, admin_patients):
    icu_keys =  ICU_patients.keys()
    admin_keys = admin_patients.keys()

    for icu_key in tqdm(icu_keys, desc='labeling..'):
        hadm_id = ICU_patients[icu_key]['HADM_ID']
        if hadm_id in admin_keys:
            # for sanity check
            assert admin_patients[hadm_id]['SUBJECT_ID'] == ICU_patients[icu_key]['SUBJECT_ID']
            try:
                death_time = int(''.join(re.findall("\d+", admin_patients[hadm_id]['DEATHTIME'])))
                in_time = int(''.join(re.findall("\d+", ICU_patients[icu_key]['ICU_IN'])))
                out_time = int(''.join(re.findall("\d+", ICU_patients[icu_key]['ICU_OUT'])))

                # Dead case in ICU
                if (in_time - death_time <= 0) and (out_time - death_time >= 0):
                    ICU_patients[icu_key]['LABEL'] = 1

                # Dead case not in ICU
                else:
                    ICU_patients[icu_key]['LABEL'] = 0

            # Not dead case
            except:
                ICU_patients[icu_key]['LABEL'] = 0

            ICU_patients[icu_key]['ETHNICITY'] = admin_patients[hadm_id]['ETHNICITY']
            ICU_patients[icu_key]['ADMISSION_TYPE'] = admin_patients[hadm_id]['ADMISSION_TYPE']
            ICU_patients[icu_key]['DIAGNOSIS'] = admin_patients[hadm_id]['DIAGNOSIS']

        else:
            print('ICU hadm id does not exist in admission table')
            raise AssertionError

    return ICU_patients



def make_x_data(path, ICU_patients, max_chartevent_time=3, max_seq=100):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for idx, lines in tqdm(enumerate(reader), desc='Reading Chartevents..'):
            # for finding null icu_id
            try:
                icu_id = int(lines[3])
            except:
                continue

            # check the icu_id is about our target patients
            if (idx > 0) and (icu_id in ICU_patients.keys()):
                # calculate chart event time whether it is whithin in max chartevent time
                chart_time_obj = datetime.strptime(lines[5], '%Y-%m-%d %H:%M:%S')
                ICU_in_obj = datetime.strptime(ICU_patients[icu_id]['ICU_IN'], '%Y-%m-%d %H:%M:%S')
                during = ((chart_time_obj - ICU_in_obj).days*24*3600 + (chart_time_obj - ICU_in_obj).seconds)/3600
                
                # chartevetn time has to be between icu_in_time and icu_out_time
                chart_time = int(''.join(re.findall("\d+", lines[5])))
                icu_in_time = int(''.join(re.findall("\d+", ICU_patients[icu_id]['ICU_IN'])))
                icu_out_time = int(''.join(re.findall("\d+", ICU_patients[icu_id]['ICU_OUT'])))

                if (0 <= during <= max_chartevent_time) and (chart_time <= icu_out_time):
                    # for sanity check
                    assert icu_in_time <= chart_time <= icu_out_time

                    try:
                        ICU_patients[icu_id]['X_DATA'][chart_time]['ITEMID'].append(int(lines[4]))
                        try:
                            ICU_patients[icu_id]['X_DATA'][chart_time]['VALUENUM'].append(float(lines[9]))
                        except ValueError:
                            ICU_patients[icu_id]['X_DATA'][chart_time]['VALUENUM'].append('NotNum')

                    except KeyError:
                        item_id = []
                        value_num = []
                        item_id.append(int(lines[4]))
                        try:
                            value_num.append(float(lines[9]))
                        except:
                            value_num.append(0)
                        
                        tmp = {'ITEMID': item_id, 'VALUENUM': value_num}
                        ICU_patients[icu_id]['X_DATA'][chart_time] = tmp

                    if len(ICU_patients[icu_id]['X_DATA']) > max_seq:
                        min_chart_time = min(ICU_patients[icu_id]['X_DATA'].keys())
                        del ICU_patients[icu_id]['X_DATA'][min_chart_time]
    
    return ICU_patients



def compensate_x_data(ICU_patients):
    random.seed(999)
    zero_key, one_key = [], []

    for icu_key in ICU_patients.keys():
        if ICU_patients[icu_key]['LABEL'] == 0 and len(ICU_patients[icu_key]['X_DATA']) != 0:
            zero_key.append(icu_key)
        elif ICU_patients[icu_key]['LABEL'] == 1 and  len(ICU_patients[icu_key]['X_DATA']) != 0:
            one_key.append(icu_key)

    for icu_key in ICU_patients.keys():    
        if len(ICU_patients[icu_key]['X_DATA']) == 0 and ICU_patients[icu_key]['LABEL'] == 0:
            choice_key = random.choice(zero_key)
            ICU_patients[icu_key]['X_DATA'] = ICU_patients[choice_key]['X_DATA']
            ICU_patients[icu_key]['ETHNICITY'] = ICU_patients[choice_key]['ETHNICITY']
            ICU_patients[icu_key]['ADMISSION_TYPE'] = ICU_patients[choice_key]['ADMISSION_TYPE']
            ICU_patients[icu_key]['DIAGNOSIS'] = ICU_patients[choice_key]['DIAGNOSIS']
        elif len(ICU_patients[icu_key]['X_DATA']) == 0 and ICU_patients[icu_key]['LABEL'] == 1:
            choice_key = random.choice(one_key)
            ICU_patients[icu_key]['X_DATA'] = ICU_patients[choice_key]['X_DATA']
            ICU_patients[icu_key]['ETHNICITY'] = ICU_patients[choice_key]['ETHNICITY']
            ICU_patients[icu_key]['ADMISSION_TYPE'] = ICU_patients[choice_key]['ADMISSION_TYPE']
            ICU_patients[icu_key]['DIAGNOSIS'] = ICU_patients[choice_key]['DIAGNOSIS']

        for chartTime in ICU_patients[icu_key]['X_DATA'].keys():
            assert len(ICU_patients[icu_key]['X_DATA'][chartTime]['ITEMID']) == len(ICU_patients[icu_key]['X_DATA'][chartTime]['VALUENUM'])
        
        assert 0 < len(ICU_patients[icu_key]['X_DATA']) <= 100
    
    return ICU_patients



def divide_dataset(ICU_patients):
    train_icu, test_icu = {}, {}

    for icu_key in ICU_patients.keys():
        if icu_key%10==8 or icu_key%10==9:
            test_icu[icu_key] = ICU_patients[icu_key]
        else:
            train_icu[icu_key] = ICU_patients[icu_key]
            
    assert len(ICU_patients) == len(train_icu) + len(test_icu)

    return train_icu, test_icu



def make_feature_dict(ICU_patients, diag_topk, itemid_topk, *keys):
    dict4baseInfo = {'[UNK]': 0}
    dict4itemid = {}
    is_x, is_diag = False, False

    for key in keys:
        if key == 'ITEMID':
            is_x = True
            continue
        if key == 'DIAGNOSIS':
            is_diag = True
            continue

        tmp = set()
        for val in ICU_patients.values():
            tmp.add(val[key])

        for val in sorted(list(tmp)):
            dict4baseInfo[val] = len(dict4baseInfo)
    
    if is_diag:
        diag_freq = Counter()
        for val in ICU_patients.values():
            diag_freq[val['DIAGNOSIS']] += 1
        diag_keys = sorted(list(dict(diag_freq.most_common(diag_topk)).keys()))
        for val in diag_keys:
            dict4baseInfo[val] = len(dict4baseInfo)
        
    if is_x:
        topk_itemid = Counter()
        for val in ICU_patients.values():
            for x_data in val['X_DATA'].values():
                topk_itemid.update(x_data['ITEMID'])
        topk_itemid = sorted(list(dict(topk_itemid.most_common(itemid_topk)).keys()))
        for val in topk_itemid:
            dict4itemid[val] = len(dict4itemid)

    return dict4baseInfo, dict4itemid



def make_feature(data, dict4baseInfo, dict4itemid):
    fin_data = {}
    for key in data.keys():
        value, tmp = data[key], {}
        tmp['label'] = value['LABEL']
        tmp['baseInfo'] = torch.LongTensor([dict4baseInfo[value[k]] if value[k] in dict4baseInfo else dict4baseInfo['[UNK]']  for k in ['ETHNICITY', 'ADMISSION_TYPE', 'DIAGNOSIS']])
        
        if len(dict4itemid) != 0:
            charttime_tmp = []
            for charttime in value['X_DATA'].keys():
                charttime_value = value['X_DATA'][charttime]
                chart_x = [0] * len(dict4itemid)
                for id, num in zip(charttime_value['ITEMID'], charttime_value['VALUENUM']):
                    try:
                        if num == 'NotNum':
                            chart_x[dict4itemid[id]] += 1
                        else:
                            if chart_x[dict4itemid[id]] == 0:
                                chart_x[dict4itemid[id]] = num
                    except KeyError:
                        continue
                charttime_tmp.append(chart_x)
            
            tmp['itemid'] = torch.FloatTensor(charttime_tmp)
        fin_data[key] = tmp
    return fin_data