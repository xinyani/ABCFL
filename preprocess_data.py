import pandas as pd
import json
import numpy as np
import random

X_KEY = ['flight_number',	'aircraft_type_code',	'origin_air_temperature',	'origin_wind_direction',	'origin_wind_speed',
         'origin_visibility',
         'origin_cloud_coverage_lvl_1',	'origin_cloud_coverage_lvl_2',	'origin_cloud_coverage_lvl_3',	'origin_cloud_height_lvl_1',	'LAA']

X_INVALID_KEY = ['flight_number',	'aircraft_type_code',]

X_need_mapping_key = ['origin_cloud_coverage_lvl_1', 'origin_cloud_coverage_lvl_2', 'origin_cloud_coverage_lvl_3']

X_LABEL_KEY = ['LAA']

def process_csv_Data(src_file, dst_file, train_ratio=0.9):
    data = pd.read_csv(src_file, encoding='utf-8')
    columns = (data.columns.values).tolist()
    
    final_data = {}
    all_data = []

    labels = []
    for column in X_LABEL_KEY:
        labels.append(data[column].tolist())
    labels = np.array(labels).T   # n * 8

    for column in columns:
        if column in X_INVALID_KEY or column in X_LABEL_KEY:
            continue
        tmp = (data[column].values).tolist()   # 取出某一列特征
        # map str to number
        if column in X_need_mapping_key:
            mp = {}
            idx = -1
            for i in range(len(tmp)):
                if tmp[i] not in mp:
                    idx += 1
                    mp[tmp[i]] = idx
            final_data[column + '_map'] = mp
            tmp = [mp[i] for i in tmp]
        if not isinstance(tmp, list):
            tmp = tmp.tolist()
        all_data.append(tmp)
    
    all_data = np.array(all_data).astype(np.float32).T    # n * m
    final_data['all_data'] = []
    assert all_data.shape[0] == labels.shape[0]

    for i in range(all_data.shape[0]):
        label = int(labels[i])
        val = all_data[i]
        final_data['all_data'].append({'data': val.tolist(), 'label': label})

    # for i in range(all_data.shape[0]):
    #     label = 1 if sum(labels[i, :]) >= 1 else 0
    #     val = all_data[i]
    #     final_data['all_data'].append({'data': val.tolist(), 'label': label})
    
    random.shuffle(final_data['all_data'])
    train_len = int(len(final_data['all_data']) * train_ratio)
    print("train length: {}, val length: {}".format(train_len, len(final_data['all_data']) - train_len))
    final_data['train_data'] = final_data['all_data'][:train_len]
    final_data['val_data'] = final_data['all_data'][train_len:]
    json.dump(final_data, open(dst_file, 'w'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    process_csv_Data('../data/yanwu.csv','../data/yanwu_processed.json')