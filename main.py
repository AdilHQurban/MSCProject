import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import sklearn
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

file_path_train_NSLKDD = 'C:/Users/DELL/PycharmProjects/MScProject/NSL-KDD2/KDDTrain+.txt'
file_path_test_NSLKDD = 'C:/Users/DELL/PycharmProjects/MScProject/NSL-KDD2/KDDTest+.txt'

NSLKDD_train = pd.read_csv(file_path_train_NSLKDD)
NSLKDD_test = pd.read_csv(file_path_test_NSLKDD)

columns = (['duration'
    , 'protocol_type'
    , 'service'
    , 'flag'
    , 'src_bytes'
    , 'dst_bytes'
    , 'land'
    , 'wrong_fragment'
    , 'urgent'
    , 'hot'
    , 'num_failed_logins'
    , 'logged_in'
    , 'num_compromised'
    , 'root_shell'
    , 'su_attempted'
    , 'num_root'
    , 'num_file_creations'
    , 'num_shells'
    , 'num_access_files'
    , 'num_outbound_cmds'
    , 'is_host_login'
    , 'is_guest_login'
    , 'count'
    , 'srv_count'
    , 'serror_rate'
    , 'srv_serror_rate'
    , 'rerror_rate'
    , 'srv_rerror_rate'
    , 'same_srv_rate'
    , 'diff_srv_rate'
    , 'srv_diff_host_rate'
    , 'dst_host_count'
    , 'dst_host_srv_count'
    , 'dst_host_same_srv_rate'
    , 'dst_host_diff_srv_rate'
    , 'dst_host_same_src_port_rate'
    , 'dst_host_srv_diff_host_rate'
    , 'dst_host_serror_rate'
    , 'dst_host_srv_serror_rate'
    , 'dst_host_rerror_rate'
    , 'dst_host_srv_rerror_rate'
    , 'attack'
    , 'level'])

NSLKDD_train.columns = columns
NSLKDD_test.columns = columns


is_attack = NSLKDD_train.attack.map(lambda a: 0 if a == 'normal' else 1)
test_attack = NSLKDD_test.attack.map(lambda a: 0 if a == 'normal' else 1)

NSLKDD_train['attack_flag'] = is_attack
NSLKDD_test['attack_flag'] = test_attack

dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']

probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']

privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']

access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

attack_labels = ['Normal','DOS','Probe','Privilege','Access']


def map_attack(attack):
    if attack in dos_attacks:
        attack_type = 1
    elif attack in probe_attacks:
        attack_type = 2
    elif attack in privilege_attacks:
        attack_type = 3
    elif attack in access_attacks:
        attack_type = 4
    else:
        attack_type = 0

    return attack_type

attack_map = NSLKDD_train.attack.apply(map_attack)
NSLKDD_train['attack_map'] = attack_map

test_attack_map = NSLKDD_test.attack.apply(map_attack)
NSLKDD_test['attack_map'] = test_attack_map

attack_vs_protocol = pd.crosstab(NSLKDD_train.attack, NSLKDD_train.protocol_type)


def bake_pies(data_list, labels):
    list_length = len(data_list)


    color_list = sns.color_palette()
    color_cycle = itertools.cycle(color_list)
    cdict = {}


    fig, axs = plt.subplots(1, list_length, figsize=(18, 10), tight_layout=False)
    plt.subplots_adjust(wspace=1 / list_length)


    for count, data_set in enumerate(data_list):


        for num, value in enumerate(np.unique(data_set.index)):
            if value not in cdict:
                cdict[value] = next(color_cycle)


        wedges, texts = axs[count].pie(data_set,
                                       colors=[cdict[v] for v in data_set.index])


        axs[count].legend(wedges, data_set.index,
                          title="Flags",
                          loc="center left",
                          bbox_to_anchor=(1, 0, 0.5, 1))

        axs[count].set_title(labels[count])

    return axs

features_to_encode = ['protocol_type', 'service', 'flag']
encoded = pd.get_dummies(NSLKDD_train[features_to_encode])
test_encoded_base = pd.get_dummies(NSLKDD_test[features_to_encode])

test_index = np.arange(len(NSLKDD_test.index))
column_diffs = list(set(encoded.columns.values)-set(test_encoded_base.columns.values))

diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)

column_order = encoded.columns.to_list()

test_encoded_temp = test_encoded_base.join(diff_df)

test_final = test_encoded_temp[column_order].fillna(0)


numeric_features = ['duration', 'src_bytes', 'dst_bytes']

to_fit = encoded.join(NSLKDD_train[numeric_features])
test_set = test_final.join(NSLKDD_test[numeric_features])



k = 9
kmeans = KMeans(n_clusters=k, random_state=45)
to_fit['cluster'] = kmeans.fit_predict(to_fit)


y_train = NSLKDD_train['attack_flag']


random_forest_params = {
    'n_estimators': 100,  # Number of trees in the forest
    'max_depth': 10,  # Maximum depth of the tree
    'min_samples_split': 3,  # Minimum number of samples required to split an internal node
    'min_samples_leaf': 2,  # Minimum number of samples required to be at a leaf node
    'random_state': 45  # Random seed for reproducibility
}

clf = RandomForestClassifier(**random_forest_params)
clf.fit(to_fit, y_train)

test_set['cluster'] = kmeans.predict(test_set)


y_pred = clf.predict(test_set)


test_labels = NSLKDD_test['attack_flag']
accuracy = accuracy_score(test_labels, y_pred)
classification_rep = classification_report(test_labels, y_pred)

y_pred = clf.predict(test_set)

# Evaluate the performance
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)

conf_matrix = confusion_matrix(test_labels, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print("Confusion Matrix:\n", conf_matrix)
