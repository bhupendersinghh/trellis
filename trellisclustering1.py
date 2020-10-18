import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans,MeanShift,DBSCAN
from sklearn import preprocessing ,  neighbors,svm
import pandas as pd
import sklearn
from kmodes import kmodes
import time
from sklearn.model_selection import train_test_split
                                                                                                  

df = pd.read_json("trellisjune.json", lines = True)

df=df[['user_id','created_timestamp','object_type','url']]


df['page_type'] = df.apply(lambda _: '', axis=1)



df.fillna("NaN", inplace=True)




df=df.sort_values(['user_id','created_timestamp'])





df=df.values

df=df.astype(object) 

s1='library'
s2='site-home'
s3='status-reply'
s4='terms-conditions'
s5='verify'
s6='search?'
s7='profile-edit'
s8='reviewDocs'
s9='pendingGroup'
s10='notification'
s11='myGroups'
s12='manageAccount'
s13='main-home'
s14='group-news'
s15='group-related'
s16='group-notif'
s17='group-home'
s18='group-events'
s19='group-discuss'
s20='group-create'
s21='group-announ'
s22='group-analy'
s23='group-admin'
s24='group-about'
s25='events'
s26='event-entry'
s27='email notif'
s28='edit announ'
s29='discussion'
s30='document-detail'
s31='discoverGroup'
s32='createDiscus'
s33='create-stat'
s34='content-notif'
s35='connection'
s36='betaReg'
s37='about'
s38='pendingConnec'
s39='createGroup'
s40='group-external'
s41='group-template'
s42='createGroup'
s43='editannouncement'

for i in range(0,46450):
    if df[i][2]==2:
        df[i][4]="group page"
        
    if df[i][2]==0:
        df[i][4]="main home"
        
    if df[i][2]==1:
        df[i][4]="profile page"
        
    if df[i][2]==3:
        df[i][4]="discussion page"
        
    if df[i][2]==4:
        df[i][4]="status"
        
    if df[i][2]==5:
        df[i][4]="event page"
        
    if df[i][2]==6:
        df[i][4]="library"
        
    if df[i][2]==7:
        df[i][4]="document page"
        
    if df[i][2]==14:
        df[i][4]="document detail"

    if df[i][2]==9:
        df[i][4]="announcement"
        
    if df[i][2]=="NaN":
        if s1 in df[i][3]: 
           df[i][4]="library"    
        if s2 in df[i][3]: 
           df[i][4]="site home"
        if s3 in df[i][3]: 
           df[i][4]="status reply"
        if s4 in df[i][3]: 
           df[i][4]="terms conditions"
        if s5 in df[i][3]: 
           df[i][4]="verify"
        if s6 in df[i][3]: 
           df[i][4]="search"
        if s7 in df[i][3]: 
           df[i][4]="profile edit"
        if s8 in df[i][3]: 
           df[i][4]="review-docs"
        if s9 in df[i][3]: 
           df[i][4]="pending groups"
        if s10 in df[i][3]: 
           df[i][4]="notification"
        if s11 in df[i][3]: 
           df[i][4]="mygroups"
        if s12 in df[i][3]: 
           df[i][4]="manageaccount"
        if s13 in df[i][3]: 
           df[i][4]="main home"
        if s14 in df[i][3]: 
           df[i][4]="group news"
        if s15 in df[i][3]: 
           df[i][4]="group related"
        if s16 in df[i][3]: 
           df[i][4]="group notification"
        if s17 in df[i][3]: 
           df[i][4]="group home"
        if s18 in df[i][3]: 
           df[i][4]="group events"
        if s19 in df[i][3]: 
           df[i][4]="group discussion"
        if s20 in df[i][3]: 
           df[i][4]="group create"
        if s21 in df[i][3]: 
           df[i][4]="group announce"
        if s22 in df[i][3]: 
           df[i][4]="group analytic"
        if s23 in df[i][3]: 
           df[i][4]="group admin"
        if s24 in df[i][3]: 
           df[i][4]="group about"
        if s25 in df[i][3]: 
           df[i][4]="events"
        if s26 in df[i][3]: 
           df[i][4]="event entry"
        if s27 in df[i][3]: 
           df[i][4]="email notif"
        if s28 in df[i][3]: 
           df[i][4]="edit announcement"
        if s29 in df[i][3]: 
           df[i][4]="discussion"
        if s30 in df[i][3]: 
           df[i][4]="document detail"
        if s31 in df[i][3]: 
           df[i][4]="discover groups"
        if s32 in df[i][3]: 
           df[i][4]="create discussion"
        if s33 in df[i][3]: 
           df[i][4]="create status"
        if s34 in df[i][3]: 
           df[i][4]="content_notif"
        if s35 in df[i][3]: 
           df[i][4]="connection"
        if s36 in df[i][3]: 
           df[i][4]="beta registration"
        if s37 in df[i][3]: 
           df[i][4]="about page"
        if s38 in df[i][3]: 
           df[i][4]="pending connection"
        if s39 in df[i][3]: 
           df[i][4]="create group"
        if s40 in df[i][3]: 
           df[i][4]="group-external"
        if s41 in df[i][3]: 
           df[i][4]="group-template"
        if s42 in df[i][3]: 
           df[i][4]="announcement"
        if s43 in df[i][3]:
           df[i][4]="edit announcement" 
        
    
        
    df[i][1]=time.ctime(df[i][1])

    
    


df= pd.DataFrame(df, columns=['user_id','created_time','object_type','url','page_type'])




    
df['url'] = df['url'].map(lambda x: str(x)[32:])




df=df.values

df=df.astype(object)


for i in range(0,46450):
    if df[i][3]=='':
        df[i][3]="document upload"
      


df= pd.DataFrame(df, columns=['user_id','created_time','object_type','url','page_type'])



print (df.to_string())

df.to_csv('filteringdata1month.csv')


