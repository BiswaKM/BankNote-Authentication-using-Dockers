# Basics
import pandas as pd
import numpy as np
# Visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler,binarize
# Model Selection
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
# Metrics
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,roc_curve,accuracy_score
# Feature Selection
from sklearn.feature_selection import SelectKBest,chi2

df=pd.read_csv('C:/Users/biswa/Desktop/Data Science project_Krish Naik/Fake currency Classifier/BankNote_Authentication.csv')

def summary(data):
    summary_df={'count':data.shape[0],
        'Na values':data.isnull().sum(),
        '% Na':(data.isna().sum()/data.shape[0])*100,
        'Unique':data.nunique(),
        'D Types':data.dtypes,
        'min':data.min(),
        '25%':data.quantile(0.25),
        '50%':data.quantile(0.50),
        'mean':data.mean(),
        '75%':data.quantile(0.75),
        'max':data.max()
        }
    return pd.DataFrame(summary_df)
print(summary(df))

df.hist(figsize=(12,8))

col_names=df.drop('class',axis=1).columns.tolist()
plt.figure(figsize=(10,3))
i=0
for col in col_names:
    plt.subplot(1,4,i+1)
    plt.grid(True,alpha=0.5)
    sns.kdeplot(df[col][df['class']==0],label='Fake Note')
    sns.kdeplot(df[col][df['class']==1],label='Real Note')
    plt.title('Class vs ' + col)
    plt.tight_layout()
    i+=1
plt.show()

# Spliting of Data sets
x=df.iloc[:,0:3]
y=df[['class']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=7)

# Models
models=[]
models.append(('LR',LogisticRegression()))
models.append(('DT',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('SVM',SVC()))
models.append(('RF',RandomForestClassifier()))
models.append(('ADA',AdaBoostClassifier()))

def model_selection(x_train,y_train):
    acc_result=[]
    auc_result=[]
    names=[]
    col=['Model','ROC AUC Mean','ROC AUC Std','ACC Mean','ACC Std']
    result=pd.DataFrame(columns=col)
    i=0
    for name,model in models:
        kfold=KFold(n_splits=10,random_state=7)
        cv_acc_result=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
        cv_auc_result=cross_val_score(model,x_train,y_train,cv=kfold,scoring='roc_auc')
        acc_result.append(cv_acc_result)
        auc_result.append(cv_auc_result)
        names.append(name)
        result.loc[i]=[name,
                       cv_auc_result.mean(),
                       cv_auc_result.std(),
                       cv_acc_result.mean(),
                       cv_acc_result.std()]
        result=result.sort_values('ROC AUC Mean',ascending=False)
        i+=1
    acc_result=pd.DataFrame(acc_result).T
    auc_result=pd.DataFrame(auc_result).T
    acc_result.columns=names
    auc_result.columns=names
    acc_result.plot(kind='box')
    auc_result.plot(kind='box')
    plt.show()
    return(result)
        
model_selection(x_train,y_train)

# Model Validation



def model_validation(model,x_test,y_test,thr=0.5): 
    y_pred_prob=model.predict_proba(x_test)[:,1]
    y_pred=binarize(y_pred_prob.reshape(1,-1),thr)[0]
    cnf_matrix=confusion_matrix(y_test,y_pred)
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    sns.heatmap(cnf_matrix,annot=True,fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    
    fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
    plt.subplot(1,2,2)
    sns.lineplot(fpr,tpr)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.show()
    
    print('Classification Report :')
    print('===='*20)
    print(classification_report(y_test,y_pred))
    score=tpr-fpr
    opt_threshold=sorted(zip(score,threshold))[-1][1]
    print('===='*20)
    print('Area under the curve',roc_auc_score(y_test,y_pred))
    print('Accuracy',accuracy_score(y_test,y_pred))
    print('Optimal Threshold',opt_threshold)
    print('='*20)
    
classifier=KNeighborsClassifier()
param_grid={
            'leaf_size':[4,6,8,9,12],
            'n_neighbors':[2,5,7,9,11],
            'p':[1,2]}

grid=GridSearchCV(KNeighborsClassifier(),param_grid=param_grid)
grid.fit(x_train,y_train)
grid.best_estimator_
final_model=grid.best_estimator_

model_validation(final_model,x_test,y_test)

# Creating a pickle file using serialization
import pickle    
pickle_out=open('final_model.pkl','wb')
pickle.dump(final_model,pickle_out)
pickle_out.close()





