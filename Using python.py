import matplotlib.pyplot as plt
import numpy as np

x=np.logspace(-1,1,40)
y1=x**2
y2=x**1.5
plt.figure()
plt.plot(x,y1,'bo-',linewidth=2,markersize=4,label='first')
plt.plot(x,y2,'gs-',linewidth=2,markersize=6,label='second')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-2,12,-5,105])
plt.legend(loc='upper left')

x=np.random.normal(size=1000)
plt.hist(x)
plt.hist(x,normed=True)
plt.hist(x,normed=True,bins=np.linspace(-5,5,21))


plt.figure()
plt.subplot(221)
plt.hist(x,bins=30)
plt.subplot(222)
plt.hist(x,bins=30,normed=True)
plt.subplot(223)
plt.hist(x,bins=30,cumulative=30)
plt.subplot(224)
plt.hist(x,bins=30,normed=True,histtype='step')


import random
random.choice([1,2,3,4,5,6])
random.choice(range(1,7))
random.choice([range(1,7)])

ys=[]
for rep in range(100):
    y=0
    for k in range(10):
        x=random.choice(range(1,7))
        y+=x
    ys.append(y)
plt.hist(ys)
plt.hist(ys,normed=True)

import time
start_time=time.clock()
x=np.random.randint(1,7,(10000,10))
y=np.sum(x,axis=1)
plt.hist(y)
end_time=time.clock()
print(end_time-start_time)

#random walk
delta_x=np.random.normal(0,1,(2,100))
x=np.cumsum(delta_x,axis=1)
plt.plot(x[0],x[1],'ro-')

x_0=np.array([[0],[0]])
delta_x=np.random.normal(0,1,(2,100))
X=np.concatenate((x_0,np.cumsum(delta_x,axis=1)),axis=1)
plt.plot(X[0],X[1],'ro-')


plt.figure()
plt.subplot(121)
plt.plot(x[0],x[1],'ro-')
plt.subplot(122)
plt.plot(x[0],x[1],'ro-',linewidth=0.5,markersize=3)



test="This is my test. we're keeping this text short to keep things manageable "

def count_words(text):
    text=text.lower()
    skips=[',','.',';',':',"'"]
    for ch in skips:
        text.replace(ch,'')
    word_counts={}
    for word in text.split(' '):
        if word in word_counts:
            word_counts[word]+=1
        else:
            word_counts[word]=1
    return word_counts

from collections import Counter
def count_words_fast(text):
    skips=[',','.',';',':',"'"]
    for ch in skips:
        text.replace(ch,'')
    word_counts=Counter(text.split(' '))
    return word_counts

def read_book(title_path):
    with open(title_path,'r',encoding='utf8') as current_file:
        text=current_file.read()
        text=text.replace('\n','').replace('r','')
    return text

def word_stats(word_counts):
    num_unique=len(word_counts)
    counts=word_counts.values()
    return(num_unique,counts)



text=read_book('./Books/English/shakespeare/Romeo and Juliet.txt')
len(text)
ind=text.find("What's in a name?")
sample_text=text[ind:ind+1000]

word_counts=count_words_fast(text)
(num_unique,counts)=word_stats(word_counts)
print(num_unique,sum(counts))


import os
book_dir="./Books"
import pandas as pd

stats=pd.DataFrame(columns=('language','author','title','length','unique'))
title_num=1
for language in os.listdir(book_dir):
    for author in os.listdir(book_dir+'/'+language):
        for title in os.listdir(book_dir+'/'+language+'/'+author):
            inputfile=book_dir+'/'+language+'/'+author+'/'+title
            print(inputfile)
            text=read_book(inputfile)
            (num_unique,counts)=word_stats(count_words_fast(text))
            stats.loc[title_num]=language,author.capitalize(),title.replace('.txt',''),sum(counts),num_unique
            title_num+=1


table=pd.DataFrame(columns=('name','age'))
table.loc[1]='james',32
table.loc[2]='tom',12
import matplotlib.pyplot as plt
plt.plot(stats.length,stats.unique,'bo')
plt.loglog(stats.length,stats.unique,'bo')

plt.figure(figsize=(10,10))
subset=stats[stats.language=='English']
plt.loglog(subset.length,subset.unique,'o',label='English',color='crimson')
plt.loglog(subset.length,subset.unique,'o',label="English",color="crimson")
subset=stats[stats.language=="French"]
plt.loglog(subset.length,subset.unique,'o',label="French",color="forestgreen")
subset=stats[stats.language=="German"]
plt.loglog(subset.length,subset.unique,'o',label="German",color="orange")
subset=stats[stats.language=="Portuguese"]
plt.loglog(subset.length,subset.unique,'o',label="Portuguese",color="blueviolet")

plt.legend()
plt.xlabel('Book length')
plt.ylabel('Number of unique')

import numpy as np
def distance(p1,p2):
    return np.sqrt(np.sum(np.power(p2-p1,2)))
p1=np.array([1,1])
p2=np.array([4,5])
distance(p1,p2)



import random
def majority_vote(votes):
    vote_counts={}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote]+=1
        else:
            vote_counts[vote]=1
    winners=[]
    max_counts=max(vote_counts.values())
    for vote,count in vote_counts.items():
        if count==max_counts:
            winners.append(vote)
    return random.choice(winners)

votes=[1,2,3,1,2,3,1,2,3,3,3,3]
vote_counts=majority_vote(votes)
max_counts=max(vote_counts.values())

import scipy.stats as ss
def majority_vote_short(votes):
    mode,count=ss.mstats.mode(votes)
    return mode

points=np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
p = np.array([2.5,2])

def find_nearest_neighbors(p,points,k=5):
    distances=np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i]=distance(p,points[i])
    ind=np.argsort(distances)
    return ind[:k]

import matplotlib.pyplot as plt
plt.plot(points[:,0],points[:,1],'ro')
plt.plot(p[0],p[1],'bo')
plt.axis=[0.5,3.5,0.5,3.5]

def knn_predict(p,points,outcomes,k=5):
    ind=find_nearest_neighbors(p,points,k)
    return majority_vote(outcomes[ind])


outcomes=np.array([0,0,0,0,1,1,1,1,1])
  
def generate_synth_data(n=50):
    points=np.concatenate((ss.norm(0,1).rvs((n,2)),ss.norm(1,1).rvs((n,2))),axis=0)
    outcomes=np.concatenate((np.repeat(0,n),np.repeat(1,n)))
    return (points,outcomes)
    
n=1000
(points,outcomes)=generate_synth_data(n)
plt.figure()
plt.plot(points[:n,0],points[:n,1],'ro',markersize=2)
plt.plot(points[n:,0],points[n:,1],'bo',markersize=2)


def make_prediction_grid(predictors,outcomes,limits,h,k):
    (x_min,x_max,y_min,y_max)=limits
    xs=np.arange(x_min,x_max,h)
    ys=np.arange(y_min,y_max,h)
    xx,yy=np.meshgrid(xs,ys)
    prediction_grid=np.zeros(xx.shape,dtype=int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p=np.array([x,y])
            prediction_grid[j,i]=knn_predict(p,predictors,outcomes,k)
    return (xx,yy,prediction_grid)

(predictors,outcomes)=generate_synth_data(1000)
k=5;limits=(-3,4,-3,4);h=0.1
(xx,yy,prediction_grid)=make_prediction_grid(predictors,outcomes,limits,h,k)

def plot_prediction_grid(xx,yy,prediction_grid):
    from matplotlib.colors import ListedColormap
    background_colormap=ListedColormap(['hotpink','lightskyblue','yellowgreen'])
    observation_colormap=ListedColormap(['red','blue','green'])
    plt.figure(figsize=(10,10))
    plt.pcolormesh(xx,yy,prediction_grid,cmap=background_colormap,alpha=0.5)
    plt.scatter(predictors[:,0],predictors[:,1],c=outcomes,cmap=observation_colormap,s=50)
    plt.xlabel('Variable1');plt.ylabel('Variable2')
    plt.xticks(());plt.yticks(())
    plt.xlim(np.min(xx),np.max(xx))
    plt.ylim(np.min(yy),np.max(yy))

plot_prediction_grid(xx,yy,prediction_grid)
    



from sklearn import datasets
iris=datasets.load_iris()
predictors=iris.data[:,0:2]
outcomes=iris.target
plt.plot(predictors[outcomes==0][:,0],predictors[outcomes==0][:,1],'ro')
plt.plot(predictors[outcomes==1][:,0],predictors[outcomes==1][:,1],'go')
plt.plot(predictors[outcomes==2][:,0],predictors[outcomes==2][:,1],'bo')

k=5;limits=(4,8,1.5,4.5);h=0.1
(xx,yy,prediction_grid)=make_prediction_grid(predictors,outcomes,limits,h,k)
plot_prediction_grid(xx,yy,prediction_grid)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors,outcomes)
sk_predictions=knn.predict(predictors)
plt.plot(sk_predictions,'ro',markersize=1)

my_predictions=np.array([knn_predict(p,predictors,outcomes,5) for p in predictors])

np.mean(sk_predictions==my_predictions)
np.mean(outcomes==my_predictions)


data={'name':['tim','pan','jim','san'],
      'age':[23,12,23,55],
      'zip':['123','432','243','234','1234']}


import pandas as pd
whisky=pd.read_csv('whiskies.txt')
whisky['region']=pd.read_csv('regions.txt')
whisky.head()
whisky.columns
flavors=whisky.iloc[:,3:14]

import matplotlib.pyplot as plt
corr_flavors=pd.DataFrame(flavors)
plt.pcolor(corr_flavors)
plt.colorbar()

corr_whisky=pd.DataFrame.corr(flavors.transpose())
plt.pcolor(corr_whisky)
plt.colorbar()

from sklearn.cluster.bicluster import SpectralCoclustering
model=SpectralCoclustering(n_clusters=6,random_state=0)
model.fit(corr_whisky)

model.rows_
np.sum(model.rows_,axis=0)
np.sum(model.rows_,axis=1)
model.row_labels_

whisky['Group']=pd.Series(model.row_labels_,index=whisky.index)
whisky=whisky.loc[np.argsort(model.row_labels_)]
whisky=whisky.reset_index(drop=True)

correlations=pd.DataFrame.corr(whisky.iloc[:,3:14].transpose())
correlations=np.array(correlations)

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title('Original')
plt.subplot(122)
plt.pcolor(correlations)
plt.title('Rearranged')
plt.axis('tight')
plt.colorbar()



data=pd.Series([1,2,3,4])
data=data.loc[[3,0,1,2]]
data=data.reset_index(drop=True)

birddata=pd.read_csv('bird_tracking.csv')
birddata.info()
birddata.head()

import matplotlib.pyplot as plt
import numpy as np
ix=birddata.bird_name=='Eric'
x,y=birddata.longitude[ix],birddata.latitude[ix]
plt.plot(x,y,'.')

bird_names=pd.unique(birddata.bird_name)
for bird_name in bird_names:
    ix=birddata.bird_name==bird_name
    x,y=birddata.longitude[ix],birddata.latitude[ix]
    plt.plot(x,y,'.',label=bird_name)
plt.xlabel('longitude');plt.ylabel('latitude')
plt.legend(loc='upper left')

plt.figure(figsize=(8,4))
ix=birddata.bird_name=='Eric'
speed=birddata.speed_2d[ix]
ind=np.isnan(speed)
plt.hist(speed[~ind],bins=np.linspace(0,30,20),normed=True)
plt.xlabel('2d speed m/s');plt.ylabel('frequency')

birddata.speed_2d.plot(kind='hist',range=[0,30])
plt.xlabel('2d speed m/s')



import datetime
datetime.datetime.today()
date_str=birddata.date_time[0]
datetime.datetime.strptime(date_str[:-3],"%Y-%m-%d %H:%M:%S")

timestamps=[]
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime\
    (birddata.date_time.iloc[k][:-3],"%Y-%m-%d %H:%M:%S"))

birddata['timestamp']=pd.Series(timestamps,index=birddata.index)



data=birddata[birddata.bird_name=='Eric']
times=data.timestamp
elapsed_time=[time-times[0] for time in times]
elapsed_days=np.array(elapsed_time)/datetime.timedelta(days=1)
next_day=1
daily_mean_speed=[]
inds=[]
for (i,t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day+=1
        inds=[]
plt.figure(figsize=(8,6))
plt.plot(daily_mean_speed)
plt.xlabel('day');plt.ylabel('mean speed m/s')

    

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj=ccrs.Mercator()

plt.figure(figsize=(10,10))
ax=plt.axes(projection=proj)
ax.set_extent((-25,20,52,10))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS,linestyle=':')

bird_names=pd.unique(birddata.bird_name)
for name in bird_names:
    ix=birddata['bird_name']==name
    x,y=birddata.longitude[ix],birddata.latitude[ix]
    ax.plot(x,y,transform=ccrs.Geodetic(),label=name)
            
























