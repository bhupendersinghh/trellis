import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans,MeanShift,DBSCAN
from sklearn import preprocessing ,  neighbors,svm
import pandas as pd
import sklearn
import time
from sklearn.model_selection import train_test_split
from pandas import ExcelWriter
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from IPython.display import display
df = pd.read_csv("filteringdata1month.csv")

del(df['Unnamed: 0'])

df['url_split']=df.url.str.split('/').str.get(0)
df['url_split']=df.url_split.str.split('?').str.get(0)

df1= df[~((df['object_type']==2.0) & (df['url_split'].isin(['event-detail','discussions-about', 'announcement-detail', 'document-reader', 'discussions-reply', 'status']))) ]
        
              

gp = df1.groupby('user_id').aggregate(np.count_nonzero)
tf = gp[gp.created_time > 9].reset_index()
tf=tf['user_id'].tolist()
df2=df1[(df1['user_id'].isin(tf))]
#print(len(df1.user_id.unique()))



gp = df1.groupby('user_id').aggregate(np.count_nonzero)
tf = gp[gp.created_time < 10].reset_index()
tf=tf['user_id'].tolist()
df3=df1[(df1['user_id'].isin(tf))]
##df3.to_csv('unuseddata.csv')

def get_url_type(url_sub_type):
   url_types = {}
   
   url_types['discussion'] = [
       'discussions-about', 'group-discussions', 'discussions', 'discussions-reply',
       'createDiscussions'
   ]
   url_types['group'] = [
       'group-home', 'myGroups', 'group-about', 'group-news', 'pendingGroups', 'group-related',
       'discoverGroups', 'group-create-request', 'group-analytics', 'createGroupStep1',
       'createGroupStep2', 'createGroupStep3', 'createGroupStep4', 'group-external-drive',
       'group-template'
   ]
   url_types['announcement'] = [
       'announcement-detail', 'group-announcement', 'announcement-reply', 'edit-announcement'
   ]
   url_types['document'] = [
       'document-reader', 'group-library', 'document-detail', 'library', 'reviewDocs','document-uploads'
   ]
   url_types['profile_view'] = ['profile', 'group-admin-members']
   url_types['event'] = ['event-detail', 'group-events', 'events', 'event-entry']
   url_types['notification'] = ['notification', 'content-notification', 'group-notification','main-home', 'site-home']
   url_types['profile_edit'] = ['profile-edit', 'email-notification', 'manageAccount']
   url_types['registration'] = ['betaRegistration']
   url_types['search'] = ['search']
   url_types['status'] = ['status', 'create-status', 'status-reply']
   url_types['connection'] = ['connections', 'pendingConnections']
 
    
   for url_type, sub_types in url_types.items():
       if url_sub_type in sub_types:
           return url_type

pd.options.mode.chained_assignment = None


df2['url_type'] = df2['url_split'].apply(get_url_type)
df2 = df2[~pd.isna(df2['url_type'])]



print(df2['url_type'].value_counts())


user_df_columns = [
    'discussion_urls', 'group_urls', 'announcement_urls', 'document_urls',
    'profile_view_urls', 'event_urls', 'notification_urls', 'profile_edit_urls', 'registration_urls',
    'search_urls', 'status_urls', 'connection_urls', 
]

user_df_dict = {}
for index, row in df2.iterrows():
    if row['user_id'] not in user_df_dict:
        user_df_dict[row['user_id']] = [0]*len(user_df_columns)
    
    column_name = row['url_type'] + '_urls'
    idx = user_df_columns.index(column_name)
    user_df_dict[row['user_id']][idx] += 1
    
user_df = pd.DataFrame.from_dict(user_df_dict, orient='index', columns=user_df_columns)


user_scaled_df = user_df.apply(lambda x: round(x/x.sum(), 2), axis=1)


print(user_scaled_df.to_string())

clustering_columns = [
    'discussion_urls', 'group_urls', 'document_urls', 'profile_view_urls', 'notification_urls',
    'profile_edit_urls'
]

user_scaled_df_2 = user_scaled_df[clustering_columns]

pca = PCA(n_components=2)
transformed = pd.DataFrame(pca.fit_transform(user_scaled_df_2[clustering_columns]))
plt.scatter(transformed[0], transformed[1])
plt.show()


distortions = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, n_init=50, max_iter=2000, random_state=100).fit(user_scaled_df_2)
    cluster_labels = kmeans.fit_predict(user_scaled_df_2)
    silhouette_avg = round(silhouette_score(user_scaled_df_2, cluster_labels), 3)
##    print('%s: %s' % (k, silhouette_avg))
    distortions.append(sum(np.min(cdist(user_scaled_df_2, kmeans.cluster_centers_, 'euclidean'), axis=1)) / user_scaled_df_2.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, n_init=50, max_iter=2000, random_state=100).fit(user_scaled_df_2)
labels = kmeans.labels_
user_scaled_df_2['label'] = labels
plt.figure(figsize=(12, 9))
plt.scatter(transformed[0], transformed[1], c=labels)

plt.show()


##for c in clustering_columns:
##    print(c)
##    for l in sorted(user_scaled_df_2['label'].unique()):
##        filtered_df = user_scaled_df_2[user_scaled_df_2['label'] == l]
##        print('Cluster %s: %s' % (l+1, filtered_df[c].mean()))


for l in sorted(user_scaled_df_2['label'].unique()):
    print('Cluster %s' % str(l+1))
    filtered_df = user_scaled_df_2[user_scaled_df_2['label'] == l]
    
    cluster_url_percentages = []
    for c in clustering_columns:
        cluster_url_percentages.append((c, filtered_df[c].mean()))
    
    for c, percentage in sorted(cluster_url_percentages, key=lambda x: x[1], reverse=True):
        print(c)
        print(round(percentage*100, 2),'%')
    
    print('\n')



