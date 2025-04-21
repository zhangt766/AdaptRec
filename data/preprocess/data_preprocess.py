import pandas as pd
import numpy as np
from numpy.lib.arraypad import pad
data = pd.read_csv('./u.data', sep='\t')
data.columns = ['session_id', 'item_id', 'rating', 'time']
data['session_id'] = data['session_id'] - 1
data['item_id'] = data['item_id'] - 1
groups = data.groupby('session_id')
ids = data.session_id.unique()
data_grouped = pd.DataFrame()
for id in ids:
    group = groups.get_group(id)
    group = group.sort_values('time')
    data_grouped = pd.concat([data_grouped, group], axis=0)
data_grouped
l3_session = data_grouped
total_ids = l3_session.session_id.unique()
groups = l3_session.groupby('session_id')
id_time = dict()
for id in total_ids:
    group = groups.get_group(id)
    begin_time = min(group['time'])
    id_time[id] = begin_time
time_ids = sorted(id_time.items(), key = lambda kv:(kv[1], kv[0]))
sorted_ids = [i[0] for i in time_ids]
fractions = np.array([0.8, 0.1, 0.1])
train_ids, val_ids, test_ids = np.array_split(sorted_ids, (fractions[:-1].cumsum() * len(total_ids)).astype(int))
train_sessions = l3_session[l3_session['session_id'].isin(train_ids)]
val_sessions = l3_session[l3_session['session_id'].isin(val_ids)]
test_sessions = l3_session[l3_session['session_id'].isin(test_ids)]
train_sessions.to_pickle('./train_sessions.df')
val_sessions.to_pickle('./val_sessions.df')
test_sessions.to_pickle('./test_sessions.df')
def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist
length = 10
pad_item = 1682
train_sessions = pd.read_pickle('./train_sessions.df')
groups = train_sessions.groupby('session_id')
ids = train_sessions.session_id.unique()
state, len_state, action, is_read, next_state, len_next_state, is_done = [], [], [], [], [],[],[]
for id in ids:
    group = groups.get_group(id)
    history = []
    for index, row in group.iterrows():
        s = list(history)
        len_state.append(length if len(s) >= length else 1 if len(s) == 0 else len(s))
        s = pad_history(s, length, pad_item)
        a = row['item_id']
        is_r = row['time']
        state.append(s)
        action.append(a)
        is_read.append(is_r)
        history.append(row['item_id'])
        is_done.append(False)
    is_done[-1] = True
dic = {'seq':state,'len_seq':len_state,'next':action}
replay_buffer=pd.DataFrame(data=dic)
replay_buffer.to_pickle('./train_data.df')
dic = {'seq_size':[length],'item_num':[pad_item]}
data_statis = pd.DataFrame(data=dic)
data_statis.to_pickle('./data_statis.df')
def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist
state, len_state, action = [], [], []
ids = val_sessions.session_id.unique()
groups = val_sessions.groupby('session_id')
for id in ids:
    group = groups.get_group(id)
    history = []
    for index, row in group.iterrows():
        s = list(history)
        len_state.append(length if len(s) >= length else 1 if len(s) == 0 else len(s))
        s = pad_history(s, length, pad_item)
        a = row['item_id']
        is_r = row['time']
        state.append(s)
        action.append(a)
        history.append(row['item_id'])
val_dic = {'seq':state,'len_seq':len_state,'next':action}
val_buffer=pd.DataFrame(data=val_dic)
val_buffer.to_pickle('./val_data.df')
def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist
test_sessions = pd.concat([val_sessions, test_sessions], axis=0)
state, len_state, action = [], [], []
ids = test_sessions.session_id.unique()
groups = test_sessions.groupby('session_id')
for id in ids:
    group = groups.get_group(id)
    history = []
    for index, row in group.iterrows():
        s = list(history)
        len_state.append(length if len(s) >= length else 1 if len(s) == 0 else len(s))
        s = pad_history(s, length, pad_item)
        a = row['item_id']
        is_r = row['time']
        state.append(s)
        action.append(a)
        history.append(row['item_id'])
test_dic = {'seq':state,'len_seq':len_state,'next':action}
test_buffer=pd.DataFrame(data=test_dic)
test_buffer.to_pickle('./test_data.df')
def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist
state, len_state, action = [], [], []
ids = test_sessions.session_id.unique()
groups = test_sessions.groupby('session_id')
for id in ids:
    group = groups.get_group(id)
    history = []
    for index, row in group.iterrows():
        history.append(row['item_id'])
    state.append(pad_history(history[:-1], length, pad_item))
    len_state.append(length if len(history[:-1]) >= length else 1 if len(history[:-1]) == 0 else len(history[:-1]))
    action.append(history[-1])
test_dic = {'seq':state,'len_seq':len_state,'next':action}
test_buffer=pd.DataFrame(data=test_dic)
test_buffer.to_pickle('./Test_data.df')