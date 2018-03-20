
import pandas as pd
titanic=pd.read_csv('train.csv')
titanic.head()
titanic.info()
titanic.describe()

titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())
titanic['Sex'].unique()
titanic.Sex[titanic['Sex']=='male']=0
titanic.Sex[titanic['Sex']=='female']=1

titanic['Embarked'].unique()
titanic['Embarked']=titanic['Embarked'].fillna('S')
titanic.Embarked[titanic['Embarked']=='S']=0
titanic.Embarked[titanic['Embarked']=='C']=1
titanic.Embarked[titanic['Embarked']=='Q']=2




#__________________LinearRegression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

predictors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
alg=LinearRegression()
kf=KFold(3,random_state=1)

predictions=[]
for train,test in kf.split(titanic):
    train_predictors=(titanic[predictors].iloc[train,:])
    train_target=titanic['Survived'].iloc[train]
    alg.fit(train_predictors,train_target)
    test_predictions=alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

import numpy as np
predictions=np.concatenate(predictions,axis=0)
import matplotlib.pyplot as plt
plt.plot(predictions,'ro',markersize=2)
np.mean(predictions)
predictions[predictions>0.6]=1
predictions[predictions<=0.6]=0
accuracy=len(predictions[predictions==titanic['Survived']])/len(predictions)

#0.8148

#__________________LogisticRegression


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
algo=LogisticRegression(random_state=1)
import matplotlib.pyplot as plt
score_p=[]
for i in range(3,10):
    scores=cross_val_score(algo,titanic[predictors],titanic['Survived'],cv=i)
    score_p.append(np.mean(scores))
print(score_p)
plt.plot(score_p)
np.max(score_p)

#0.7969

scores=cross_val_score(algo,titanic[predictors],titanic['Survived'],cv=8)


#____________prediction


titanic_test=pd.read_csv('test.csv')

titanic_test['Age']=titanic_test['Age'].fillna(titanic['Age'].median())
titanic_test['Fare']=titanic_test['Fare'].fillna(titanic_test['Fare'].median())

titanic_test.Sex[titanic_test['Sex']=='male']=0
titanic_test.Sex[titanic_test['Sex']=='female']=1
titanic_test['Embarked']=titanic_test['Embarked'].fillna('S')
titanic_test.Embarked[titanic_test['Embarked']=='S']=0
titanic_test.Embarked[titanic_test['Embarked']=='C']=1
titanic_test.Embarked[titanic_test['Embarked']=='Q']=2

predictions=alg.predict(titanic_test[predictors])
submission=pd.DataFrame({"PassengerId":titanic_test['PassengerId'],'Survived':predictions})
submission.to_csv('kaggle_titanic.csv',index=False)




#______________________RandomForestClassifier

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
Ram=RandomForestClassifier(random_state=1,n_estimators=12,min_samples_split=2,min_samples_leaf=1)
kf=cross_validation.KFold(titanic.shape[0],n_folds=8,random_state=1)
scores=cross_validation.cross_val_score(Ram,titanic[predictors],titanic['Survived'],cv=kf)
print(scores.mean())

#0.8104


Ram2=RandomForestClassifier(random_state=1,n_estimators=40,min_samples_split=4,min_samples_leaf=2)
scores=cross_validation.cross_val_score(Ram2,titanic[predictors],titanic['Survived'],cv=kf)
print(scores.mean())

#0.8340

forplot=[]
for i in range(10,60):
    RamP=RandomForestClassifier(random_state=1,n_estimators=i,min_samples_split=4,min_samples_leaf=2)
    scores=cross_validation.cross_val_score(RamP,titanic[predictors],titanic['Survived'],cv=kf)
    print(scores.mean())
    forplot.append(scores.mean())
plt.plot(forplot,'ro-',markersize=2)

#50,   2,3,4,5   1,2

Ram=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=2)
#0.8396


titanic['NameLength']=titanic['Name'].apply(lambda x : len(x))

import re
def get_title(name):
    title_search=re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ''

titles=titanic['Name'].apply(get_title)
print(pd.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
                 "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, 
                 "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10,
                 "Countess": 10, "Jonkheer": 10, "Sir": 9,
                 "Capt": 7, "Ms": 2,"Dona": 4}
for k,v in title_mapping.items():
    titles[titles==k]=v
print(pd.value_counts(titles))
titanic['Titles']=titles







#---------Feature Selection



import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif

predictors=["Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Titles"]
selector=SelectKBest(f_classif,k=5)
selector.fit(titanic[predictors],titanic['Survived'])
scores=-np.log10(selector.pvalues_)

plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation='vertical')


predictors=['Pclass','Sex','Fare','Titles']

from sklearn.feature_selection import SelectKBest,f_classif
selector=SelectKBest(f_classif,k=4)
selector.fit(titanic[predictors],titanic['Survived'])

scores=-np.log10(selector.pvalues_)

plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation='vertical')




#___________Model Selectioin
#___________Ensemble Generation

predictors=["Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Titles"]

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
kf=cross_validation.KFold(titanic.shape[0],n_folds=8,random_state=1)
alg=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=4)
scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=kf)
scores.mean()

#0.8406

#___GradientBoosingClassifier and LogisticRegression

predictors=["Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Titles"]

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
algo=[[GradientBoostingClassifier(random_state=1,n_estimators=50,max_depth=3),predictors],[LogisticRegression(random_state=1),predictors]]
kf=cross_validation.KFold(titanic.shape[0],n_folds=8,random_state=1)

predictions=[]
for train,test in kf:
    train_target=titanic['Survived'].iloc[train]
    full_test_predictions=[]
    for alg,predictors in algo:
        alg.fit(titanic[predictors].iloc[train,:],train_target)
        test_predictions=alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions=(full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions<=0.6]=0
    test_predictions[test_predictions>0.6]=1
    predictions.append(test_predictions)

predictions=np.concatenate(predictions,axis=0)
accuracy=len(predictions[predictions==titanic['Survived']])/len(predictions)
print(accuracy)
#0.8170



#_____________GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
kf=cross_validation.KFold(titanic.shape[0],n_folds=8,random_state=1)
gra=GradientBoostingClassifier(random_state=1,n_estimators=50,max_depth=3)
gra.fit(titanic[predictors],titanic['Survived'])
scores=cross_validation.cross_val_score(gra,titanic[predictors],titanic['Survived'],cv=kf)
scores.mean()

#0.8283




#_____________finetuning

score_p=[]
depth_p=[]
for i in range(1,6):
    for j in range(10,60):
        gra=GradientBoostingClassifier(random_state=1,n_estimators=j,max_depth=i)
        scores=cross_validation.cross_val_score(gra,titanic[predictors],titanic['Survived'],cv=kf)
        depth_p.append(scores.mean())
    score_p.append(depth_p)
    depth_p=[]
    print(i)
    

for z in range(5):
    y=score_p[z]
    plt.plot(y,markersize=2,label=z)
plt.legend(loc='upper left')

# 47 2 one loop

# 37 5 wo loop 


gra=GradientBoostingClassifier(random_state=1,n_estimators=37,max_depth=5)
scores=cross_validation.cross_val_score(gra,titanic[predictors],titanic['Survived'],cv=kf)
scores.mean()

# 0.8328 one loop
# 0.8362 two loop




score_R=[]
depth_R=[]
for i in range(2,7):
    for j in range(10,60):
        alg=RandomForestClassifier(random_state=1,n_estimators=j,min_samples_split=i,min_samples_leaf=4)
        scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=kf)
        depth_R.append(scores.mean())
    score_R.append(depth_R)
    depth_R=[]
    print(i)

for x in range(5):
    y=score_R[x]
    plt.plot(y,markersize=2,label=x)
plt.legend(loc='upper left')

alg=RandomForestClassifier(random_state=1,n_estimators=43,min_samples_split=5,min_samples_leaf=4)
scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=kf)
scores.mean()

#0.8429


titanic_test=pd.read_csv('test.csv')

titanic_test['Age']=titanic_test['Age'].fillna(titanic['Age'].median())
titanic_test['Fare']=titanic_test['Fare'].fillna(titanic_test['Fare'].median())

titanic_test.Sex[titanic_test['Sex']=='male']=0
titanic_test.Sex[titanic_test['Sex']=='female']=1
titanic_test['Embarked']=titanic_test['Embarked'].fillna('S')
titanic_test.Embarked[titanic_test['Embarked']=='S']=0
titanic_test.Embarked[titanic_test['Embarked']=='C']=1
titanic_test.Embarked[titanic_test['Embarked']=='Q']=2

import re
def get_title(name):
    title_search=re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ''

titles_test=titanic_test['Name'].apply(get_title)
print(pd.value_counts(titles_test))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
                 "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, 
                 "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10,
                 "Countess": 10, "Jonkheer": 10, "Sir": 9,
                 "Capt": 7, "Ms": 2,"Dona": 4}
for k,v in title_mapping.items():
    titles_test[titles_test==k]=v
print(pd.value_counts(titles_test))
titanic_test['Titles']=titles_test

alg.fit(titanic[predictors],titanic['Survived'])
predictions=alg.predict(titanic_test[predictors])
submission=pd.DataFrame({"PassengerId":titanic_test['PassengerId'],'Survived':predictions})
submission.to_csv('kaggle_titanic_tonight.csv',index=False)























ensemble=[[GradientBoostingClassifier(random_state=1,n_estimators=50,max_depth=3),predictors],[alg,predictors]]
kf=cross_validation.KFold(titanic.shape[0],n_folds=8,random_state=1)

predictions=[]
for train,test in kf:
    train_target=titanic['Survived'].iloc[train]
    full_test_predictions=[]
    for alg,predictors in ensemble:
        alg.fit(titanic[predictors].iloc[train,:],train_target)
        test_predictions=alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions=(full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions<=0.6]=0
    test_predictions[test_predictions>0.6]=1
    predictions.append(test_predictions)

predictions=np.concatenate(predictions,axis=0)
accuracy=len(predictions[predictions==titanic['Survived']])/len(predictions)
print(accuracy)

#0.8226























