# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 00:54:02 2020

@author: User
"""
import pandas as pd
import numpy as np
from scipy import interp
from itertools import combinations
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_validate  
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
import warnings
import pickle

# In[]
warnings.filterwarnings("ignore")
randstateseed=45639820
randstateseed=76840034
randstateseed=34048502
randstateseed=27487045
randstateseed=77478393
randstateseed=11838145
randstateseed=83097023
randstateseed=11838145
randstateseed=40930938
randstateseed=83197023
randstateseed=88489545 # used


#datset file
datafilename='CSdata.tab'
#Step for comparison of algorithms
STEP1_flag=False
#Step for training and tes using best algorithm
STEP2_flag=True
#Features for Stage1 and Stage2
Level_1_medical_test=["bc","bacth","1mgDSTc","mc","ufc","types"]
Level_2_medical_test=["bc","bacth","1mgDSTc","2mgDSTc","8mgDSTc","mc","ufc","adrMass","pitMass","types"]
Diagnostic_Level_dict={"Stage1":Level_1_medical_test,"Stage2":Level_2_medical_test}
ufc_normal_upper=  403 #this is the user set value for this feature
cv_train_size=0.7 # ratio of train split in crossalidation
cv_test_size=0.3 #ratio of test  split in crossalidation
train_size=0.7 # ratio of original data for train split
test_size_final=0.3 # ratio of original data for final (never-seen) test split
crossval_kfold=5 #number of cross validation folds

#CONFIGURATIONS
pd.set_option('display.expand_frame_repr', False)
show_figures=True
cross_validaiton_flag=False
gen_feature_importances_flag=True
confusion_matrix_flag=True
classification_report_flag=True
learning_curves_flag=True
auc_roc_with_cv_flag=True
auc_roc_flag=True
pr_curve_flag=True
binary_classification_metrics_flag=True


# In[]

def impute_withmedian_log_transform(X_data,ufc_normal_upper=  ufc_normal_upper):
    # DATA IMPUTATION AND TRANSFORM 
    col_medians=X_data.median()
    for id in col_medians.index.values:
        X_data.fillna(value=col_medians[id],axis=0,inplace=True)     
    # if 'ufc' in X_data.index.values:
    #     X_data['ufc']=X_data['ufc']/ufc_normal_upper                
    #take log
    X_data=X_data.apply(np.log10,axis=0) 
    return X_data,col_medians     
    # In[]
def load_data(datafile,delim='\t'):
    dfo=pd.read_csv(datafile,delimiter=delim) #load tab seperated data   
    df=dfo.copy()
    feat_labels=pd.Series(['age','gender','bc','bacth','1mgDSTc','2mgDSTc','8mgDSTc','mc','ufc','adrMass','pitMass','types'], name='Features')   
    #rename feature columns
    df.columns=feat_labels.values   
    nameof_class=df['types'].unique()
    nof_class=nameof_class.size 
        #assign numbers to classes    
    for i,n in enumerate(subtypes):
        df.loc[df['types']==n,'types']=i   
    return (df,nof_class,nameof_class,feat_labels)  

    # In[]
def split_data(df,train_size=0.8,test_size_final=0.2, first_shuffle=True,sampling_randomseed=randstateseed):      
    #GENERATE STRATIFIED TRAIN AND FINAL TEST SAMPLES  
    if first_shuffle:
        #shuffle data rows 
        from sklearn.utils import shuffle
        df = shuffle(shuffle(df)) 
    X=df.drop(columns=df.columns.values[-1])
    y=df[df.columns.values[-1]]
    X = X.astype(np.float64)
    #stratified train test split
    X_train, X_test_final, y_train, y_test_final = train_test_split(X, y,train_size=train_size,test_size=test_size_final,
                                                        random_state=randstateseed,shuffle=True, stratify=y)
    return X_train, X_test_final, y_train, y_test_final
    
  # In[]
def preprocess_data(X_train, X_test_final, y_train, y_test_final,imputation='impute_withmedian_log_transform',ufc_normal_upper=  ufc_normal_upper):      
    #Two-Sample Kolmogorov-Smirnov Test
    #Compute the Kolmogorov-Smirnov statistic on 2 samples.
    #This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
    dfks=KolmogorovSmirnovTest(X_train,X_test_final)
    print("Train and Test data Splitting. Result of Kolmogorov Smirnov Test")
    print(dfks)
    #Measure Skwness of the Data
    sk_before=MeasureSkewness(X_train)   
    if imputation=='impute_withmedian_log_transform':
        X_train,X_train_medians=impute_withmedian_log_transform(X_train,ufc_normal_upper=  ufc_normal_upper) 
        X_test_final,X_test_final_medians=impute_withmedian_log_transform(X_test_final,ufc_normal_upper=  ufc_normal_upper) 
    #Measure Skwness of the Data after preprocessing
    sk_after=MeasureSkewness(X_train) 
    print("Skewness of Train Data:\nFeatures: {}\nBefore preprocessing:\n{}\nAfter preprocessing: \n{}".format(X_train.columns.values,sk_before,sk_after))    
    dfs_train=pd.concat([X_train,y_train],axis=1)
    dfs_test_final=pd.concat([X_test_final,y_test_final],axis=1)
    
    return (X_train,y_train,X_test_final,y_test_final,dfs_train,dfs_test_final)

# In[]
def gen_feature_importances(classifier,key,model_title,X,y_test_final, y_pred_final,feature_index,show_figures=True):    
    #MDI. Mean Decrease in ımpurty    
    # significant features
    importan=classifier.feature_importances_
    impor=pd.Series(data=classifier.feature_importances_,index=feature_index)
    names=feature_index.values
    if show_figures:
        plt.figure(figsize=(8,6))
        #Feature importances with forests of trees
        # Barplot: Add bars
        plt.bar(range(X.shape[1]), importan*100)
        # Add feature names as x-axis labels
        plt.xticks(range(X.shape[1]), names, rotation=20, fontsize = 8)
        # Create plot axis labels and title
        plt.xlabel("Features")
        plt.ylabel("Importance Ratio %")
        plt.grid(axis='y')
        plt.title('Relative Feature Importances: {} \n( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',model_title))
        figtitle='Relative Feature Importances {} ( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',model_title)
        figtitle="./figures/"+figtitle+".png"
        
        # Show plot
        plt.show()
        #plt.savefig(figtitle,dpi=1000)
        #plt.close()
    print("Relative Feature Importances")
    print(impor)
    return (impor)
# In[]
def gen_confusion_matrix(classifier, key,y_true, y_pred, model_title=None,
                          normalize=False, show_figures=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
        # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes=[class_target_codes[c] for c in key]   
    if show_figures:
        import matplotlib.pyplot as plt
        cmap=plt.cm.Blues
        title='Confusion Matrix: {} \n( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',model_title)
       
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        

        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')  
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")   
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        figtitle='Confusion Matrix {} ( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',model_title)
        figtitle="./figures/"+figtitle+".png"
        
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        #fig.text(0.01, 0.98, figannotation, weight="bold", fontsize="16",horizontalalignment='left', verticalalignment='center')
        plt.tight_layout()
        plt.show()
        plt.savefig(figtitle,dpi=1000)
        #plt.close()
    return cm

def calculate_binary_classification_metrics(classifier, key,y_true, y_pred):
    """
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)   
    # Overall accuracy
    #ACC = (TP+TN)/(TP+FP+FN+TN)
    """
    TPR=None
    TNR=None
    PPV=None
    NPV=None
    FPR=None
    FNR=None
    FDR=None
    F1=None
    ACC=None   
    if len(classifier.classes_)!=2: #binary
        print('calculate_binary_classification_metrics(): Not binary class')
        return F1,ACC,TPR,TNR,PPV,NPV,FPR,FNR,FDR    
    CM=confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)    
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    #F1 acore    
    F1=2*TPR*TNR/(TPR+TNR)    
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)       
    return F1,ACC,TPR,TNR,PPV,NPV,FPR,FNR,FDR
def calculate_multiclass_classifications_metrics(y_true, y_pred,key, target_names):
    class_id = [s for s in key]
    y_true=y_true.values
    TP = []
    FP = []
    TN = []
    FN = []
    TPR= []
    TNR= []
    PPV= []
    F1= []
    for index ,_id in enumerate(class_id):
        TP.append(0)
        FP.append(0)
        TN.append(0)
        FN.append(0)
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i] == _id:
                TP[index] += 1
            if y_pred[i] == _id and y_true[i] != y_pred[i]:
                FP[index] += 1
            if y_true[i] == y_pred[i] != _id:
                TN[index] += 1
            if y_true[i] == _id and y_true[i] != y_pred[i]:
                FN[index] += 1
                    # Sensitivity, hit rate, recall, or true positive rate
        TPR.append(TP[index]/(TP[index]+FN[index]))        
        # Specificity or true negative rate
        TNR.append(TN[index]/(TN[index]+FP[index])) 
        # Precision or positive predictive value
        PPV.append(TP[index]/(TP[index]+FP[index]))
        #F1 acore
        F1.append(2*TPR[index]*TNR[index]/(TPR[index]+TNR[index]))        
    dfe=pd.DataFrame({})    
    index=['TPR','TNR','PPV','F1','TP','FP','TN','FN']
    dfe['class name']=pd.Series(target_names,name='class name')
    dfe['TPR']=pd.Series(TPR, name='TPR')
    dfe['TNR']=pd.Series(TNR, name='TNR')
    dfe['PPV']=pd.Series(PPV, name='PPV')
    dfe['F1']=pd.Series(F1, name='F1')
    dfe['TP']=pd.Series(TP, name='TP')
    dfe['FP']=pd.Series(FP, name='FP')
    dfe['TN']=pd.Series(TN, name='TN')
    dfe['FN']=pd.Series(FN, name='FN')
    dfe=dfe.set_index('class name')
    dfe=dfe.round(3)
    return dfe
def KolmogorovSmirnovTest(X1,X2,pvaluethreshold=0.05):    
    pvaluelist=[]
    sstlist=[]
    dfe=pd.DataFrame({})
    for c in X1.columns.values:
        sst,pvalue=stats.ks_2samp(X1[c], X2[c])
        pvaluelist.append(pvalue)
        sstlist.append(sst)
    s0=pd.Series(X1.columns.values,name='Feature')
    s1=pd.Series(sstlist,name='KS-stat')
    s2=pd.Series(pvaluelist,name='p-value')
    dfe['Feature']=s0
    dfe['KS-stat']=s1
    dfe['p-value']=s2
    dfe=dfe.set_index('Feature')
    return dfe
def MeasureSkewness(X1):  
    dfe=pd.DataFrame({})
    sklist=[]
    for c in X1.columns.values:
        XFeat=X1[c].dropna()
        sk=stats.skew(XFeat,axis=0)
        sklist.append(sk)        
    s0=pd.Series(X1.columns.values,name='Feature')
    s1=pd.Series(sklist,name='Skewness')
    dfe['Feature']=s0
    dfe['Skewness']=s1
    dfe=dfe.set_index('Feature')    
    return dfe
# In[]
def gen_learning_curve(estimator, key,title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.2, 1.0, 10),scoring='accuracy',plotshow=True):
    """
    Generate a  plot of learning curve.

    """      
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    if show_figures:
        
        fig=plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.set_title('Learning Curve: {} \n( {} )'.format(' vs '.join([class_target_codes[s]  for s in reversed(key)]) if len(estimator.classes_)==2 else 'All Types',title))
        figtitle='Learning Curve {} ( {} )'.format(' vs '.join([class_target_codes[s]  for s in reversed(key)]) if len(estimator.classes_)==2 else 'All Types',title)
        figtitle="./figures/"+figtitle+".png"
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score(F1)")
        plt.grid()  
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        ax.plot(train_sizes, test_scores_mean, '^-', color="g",
                 label="Cross-validation score")  
        ax.legend(loc="best")
        ax.legend(loc="lower right")
        
        #fig.text(0.01, 0.98, figannotation, weight="bold", fontsize="16",horizontalalignment='left', verticalalignment='center')
        
        fig.tight_layout()

        plt.show()      
        #plt.show()
        #plt.savefig(figtitle,dpi=1000)

        #plt.close()
    return train_scores_mean,train_scores_std,test_scores_mean,test_scores_std
# In[]
def gen_auc_roc_with_cv(classifier,key,X,y,title,cv,plotshow=True):
    """
    This function calculates and plots the ROC curve of cross validation with AUC scores.

    """  
    if len(classifier.classes_)!=2: #binary
        print('gen_auc_roc_with_cv(): Not binary class')
        return None,None
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    X=X.values
    y=y.values
    fig=plt.figure(figsize=(8,6))
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        markers=['1','2','3','4','+','1','2','3','4','+']
        plt.plot(fpr, tpr, lw=1, alpha=0.6,marker=markers[i],
                 label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)  
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, marker='.',color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)  
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')  
    if plotshow:
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('1-Specificity (FPR)')
        plt.ylabel('Sensitivity (TPR, Recall)')
        
        plt.title('ROC Curve: {} \n( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',title))
        figtitle='ROC Curve {} ( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',title)
        figtitle="./figures/"+figtitle+".png"
        plt.legend(loc="lower right")
        #fig.text(0.01, 0.98, figannotation, weight="bold", fontsize="16",horizontalalignment='left', verticalalignment='center')
        fig.tight_layout()
        plt.show()
        #plt.savefig(figtitle,dpi=1000)
        #plt.close()
    return mean_auc,std_auc
# In[]
def gen_pr_curve(classifier,key,X,y,title,plotshow=True):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    from sklearn.metrics import average_precision_score
    from matplotlib import pyplot as plt
    model=classifier  
    if len(classifier.classes_)!=2: #binary
        print('gen_pr_curve(): Not binary class')
        return None,None,None  
    probs=model.predict_proba(X)
    probs=probs[:,1]
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y.values, probs)
    # predict class values
    y_pred = model.predict(X)
    # calculate F1 score
    f1 = f1_score(y.values, y_pred)
    # calculate precision-recall AUC
    pr_auc = auc(recall, precision)
    # calculate average precision score
    ap = average_precision_score(y, y_pred)
    if plotshow:    
        fig=plt.figure(figsize=(8,6))    
        plt.xlabel('Recall (Sensitivity,TPR)')
        plt.ylabel('Precision (PPV)')
        plt.title('PR Curve: {} \n( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',title))
        figtitle='PR Curve {} ( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',title)
        figtitle="./figures/"+figtitle+".png"
        # plot the precision-recall curve for the model
        plt.plot(recall, precision, marker='.',color='b',label='AUC= %0.3f)' % (pr_auc))
        # plot no skill      
        plt.plot([0, 1], [0.1, 0.1], linestyle='--',color='r', label=r'Chance')
        plt.legend(loc="center left")
        #fig.text(0.01, 0.98, figannotation, weight="bold", fontsize="16",horizontalalignment='left', verticalalignment='center')
        fig.tight_layout()
        # show the plot
        plt.show()
        #plt.savefig(figtitle,dpi=1000)
        #plt.close()
    print('PR Curve Scores: pr_auc=%.3f' % (pr_auc))
    return f1,pr_auc, ap
# In[]
def gen_roc_curve(classifier,key,X,y,title,plotshow=True):
    from sklearn.metrics import auc
    from matplotlib import pyplot as plt
    model=classifier  
    if len(classifier.classes_)!=2: #binary
        print('gen_pr_curve(): Not binary class')
        return None,None,None  
    probs=model.predict_proba(X)
    probs=probs[:,1]
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y.values, probs)
    # calculate roc AUC
    roc_auc = auc(fpr, tpr)
    if plotshow:    
        fig=plt.figure(figsize=(8,6))    
        plt.xlabel('1-Specificity (FPR)')
        plt.ylabel('Sensitivity (TPR, Recall)')
        plt.title('ROC Curve: {} \n( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',title))
        figtitle='ROC Curve {} ( {} )'.format(' vs '.join([class_target_codes[s] for s in reversed(key)]) if len(classifier.classes_)==2 else 'All Types',title)
        
        figtitle="./figures/"+figtitle+".png"
        # plot the precision-recall curve for the model
        plt.plot(fpr, tpr, marker='.',color='b',label='AUC= %0.3f)' % (roc_auc))
        # plot no skill        
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)    
        plt.legend(loc="lower right")
        #fig.text(0.01, 0.98, figannotation, weight="bold", fontsize="16",horizontalalignment='left', verticalalignment='center')
        fig.tight_layout()
        # show the plot
        plt.show()
        #plt.savefig(figtitle,dpi=1000)
        #plt.close()
    print('ROC Curve AUC: ROC_AUC=%.3f' % (roc_auc))
    return roc_auc
    # In[]:    
def crossvalidation(classifier,X,y,kfold=5,trainsize=0.8,testsize=0.2,sampling_randomseed=None):  
    #sss=StratifiedShuffleSplit(n_splits=kfold,train_size=trainsize,test_size=testsize,random_state=sampling_randomseed)    
    sss=StratifiedKFold(n_splits=kfold,random_state=sampling_randomseed)
    scores = cross_validate(classifier, X, y, cv=sss, scoring=['f1_macro','accuracy','precision_macro', 'recall_macro'],return_train_score=True,return_estimator=True)
    del scores['estimator']  
    score_df=pd.DataFrame(scores)
    score_df=score_df.round(3)
    #change column order
    score_df=score_df.reindex(['test_f1_macro','train_f1_macro','test_accuracy','train_accuracy','test_precision_macro', 'test_recall_macro','train_precision_macro',  'train_recall_macro','fit_time', 'score_time' ],axis=1)      
    print('cross-validation:\n', score_df)      
    #The mean score and the 95% confidence interval of the score estimate are hence given by:
    print("cross-validation mean test_f1_macro: %0.3f (+/- %0.3f)" % (score_df['test_f1_macro'].mean(), score_df['test_f1_macro'].std() * 2))    
    print("cross-validation mean train_f1_macro: %0.3f (+/- %0.3f)" % (score_df['train_f1_macro'].mean(), score_df['train_f1_macro'].std() * 2))     
    cv_test_mean_f1=score_df['test_f1_macro'].mean()
    cv_test_std95_f1=score_df['test_f1_macro'].std() * 2
    cv_test_mean_accuracy=score_df['test_accuracy'].mean()
    cv_test_std95_accuracy=score_df['test_accuracy'].std() * 2
    return (cv_test_mean_f1, cv_test_std95_f1,cv_test_mean_accuracy,cv_test_std95_accuracy)
# In[]
def create_datasubsets(X_train,y_train,X_test_final,y_test_final,classes):  
    ##CREATE all-in datasets for multiclass classification
    comb_all=tuple(classes)
    df_train_allin_dict={}
    df_test_final_allin_dict={}
    dd=X_train.copy()
    dd[y_train.name]=y_train
    dd_test_final=X_test_final.copy()
    dd_test_final[y_test_final.name]=y_test_final  
    df_train_allin_dict[comb_all]=dd
    df_test_final_allin_dict[comb_all]=dd_test_final  
    #CREATE one to one datasets for binary classification
    df_train_one2one_dict={}
    df_test_final_one2one_dict={}
    comb_one = combinations(classes, 2)
    comb_one=list(comb_one) 
    for c in comb_one:
        #print(c)
        dd=X_train.copy()
        dd[y_train.name]=y_train
        dd_test_final=X_test_final.copy()
        dd_test_final[y_test_final.name]=y_test_final      
        mask=(dd['types']==c[0]) | (dd['types']==c[1])
        dd_tr=dd[mask].copy()
        dd_tr.loc[dd_tr['types']==c[0],'types']=0
        dd_tr.loc[dd_tr['types']==c[1],'types']=1      
        df_train_one2one_dict[c]=dd_tr      
        mask=(dd_test_final['types']==c[0]) | (dd_test_final['types']==c[1])
        dd_tt=dd_test_final[mask].copy() 
        dd_tt.loc[dd_tt['types']==c[0],'types']=0
        dd_tt.loc[dd_tt['types']==c[1],'types']=1
        df_test_final_one2one_dict[c]=dd_tt    
    #CREATE one to rest datasets for binary classification
    df_train_one2rest_dict={}
    df_test_final_one2rest_dict={}   
    comb_rest=[]
    for i,c in enumerate(classes):       
        cp=classes.copy()
        cp.remove(c)
        cp=tuple(cp)
        comb_rest.append((cp,c))
    del comb_rest[0]       
    for c in comb_rest:
        #print(c)
        dd=X_train.copy()
        dd[y_train.name]=y_train
        dd_test_final=X_test_final.copy()
        dd_test_final[y_test_final.name]=y_test_final        
        dd.loc[dd['types']!=c[1],'types']=-1
        dd.loc[dd['types']==c[1],'types']=1
        dd.loc[dd['types']==-1,'types']=0                
        dd_test_final.loc[dd_test_final['types']!=c[1],'types']=-1
        dd_test_final.loc[dd_test_final['types']==c[1],'types']=1
        dd_test_final.loc[dd_test_final['types']==-1,'types']=0        
        df_test_final_one2rest_dict[c]=dd_test_final
        df_train_one2rest_dict[c]=dd
    #CREATE one to rest REVERSE datasets for binary classification
    df_train_one2restREV_dict={}
    df_test_final_one2restREV_dict={} 
    comb_restREV_=[]
    comb_restREV=[]
    for i,c in enumerate(classes):     
        cp=classes.copy()
        cp.remove(c)
        cp=tuple(cp)
        comb_restREV_.append((c,cp))     
    comb_restREV.append(comb_restREV_[0])       
    for c in comb_restREV:
        #print(c)
        dd=X_train.copy()
        dd[y_train.name]=y_train
        dd_test_final=X_test_final.copy()
        dd_test_final[y_test_final.name]=y_test_final   
        dd.loc[dd['types']!=c[0],'types']=-1
        dd.loc[dd['types']==c[0],'types']=0
        dd.loc[dd['types']==-1,'types']=1           
        dd_test_final.loc[dd_test_final['types']!=c[0],'types']=-1
        dd_test_final.loc[dd_test_final['types']==c[0],'types']=0
        dd_test_final.loc[dd_test_final['types']==-1,'types']=1        
        df_test_final_one2restREV_dict[c]=dd_test_final
        df_train_one2restREV_dict[c]=dd   
    df_allsubsets={}
    df_allsubsets['ALLIN']=[df_train_allin_dict,df_test_final_allin_dict]
    df_allsubsets['ONE2ONE']=[df_train_one2one_dict,df_test_final_one2one_dict]
    df_allsubsets['ONE2REST']=[df_train_one2rest_dict,df_test_final_one2rest_dict]
    df_allsubsets['ONE2RESTREV']=[df_train_one2restREV_dict,df_test_final_one2restREV_dict]
    return df_allsubsets 

######################### ####################################
subtypes=['nonfunctional AA',
          'subclinical CS',
          'adrenal CS',
          'pituitary CS' 
          ]
class_target_codes={
    0:'NF',
    1:'SC',
    2:'AD',
    3:'PT',
    (1,2,3):'ALL',
    (0,2,3):'ALL',
    (0,1,3):'ALL',
    (0,1,2):'ALL',
    (0,1,2,3): 'NF SC AD PT'
    }
class_target_short_codes={
    0:'NF',
    1:'SC',
    2:'AD',
    3:'PT',
    }
# In[]
def cushing_analysis(datafile,dataset_strategy,Level_x_medical_test,n_repeats, outer_cv_k_fold_count,inner_cv_k_fold_count,randstateseed=randstateseed):
    """
        #choose medical diagnostic level
        Level_x_medical_test="Stage1"
        Level_x_medical_test="Stage2"
        #choose dataset strategy
        dataset_strategy='ALLIN'
        dataset_strategy='ONE2REST'
        dataset_strategy='ONE2ONE'
    """
    import numpy as np
    #set if to True below for repeated result. False: generates new random numbers
    if True:
        #np.random.seed(249)
        np.random.seed(randstateseed)
    #cretae results table
    table_labels=['ReptNo','Model','Strategy','Diagnostic_Level','Classes','Train/Test','f1_macro Score','Accuracy','Precision_macro','Recall_macro','Balanced_Accuracy','CV fold:Best Parameters']
    df_res=pd.DataFrame({},columns=table_labels)
    table_entry=table_labels
        #cretae results table
    best_table_labels=['ReptNo','Model','Strategy','Diagnostic_Level','Classes','Train/Test','f1_macro Score','Accuracy','Precision_macro','Recall_macro','Balanced_Accuracy','CV fold:Best Parameters','Confusion_Matrix','F1','ACC','TPR','TNR','PPV','NPV','FPR','FNR','FDR','ROC_AUC','PRC_AUC','PRC Avg Precision','CV AUC mean','CV AUC std','Important Features','cv_test_mean_f1', 'cv_test_std95_f1','cv_test_mean_accuracy','cv_test_std95_accuracy','best_gridCV_meanscore']
    best_df_res=pd.DataFrame({},columns=best_table_labels)
    best_table_entry=best_table_labels
    n_classes=len(subtypes)
    classes=list(range(n_classes))
    print('Loading  data from file:',datafile,end='\n')
    #Load data
    dfo,nc,cnames,feat_labels=load_data(datafile,delim='\t') 
        #drop unwanted feature columns
    dfo=dfo[Diagnostic_Level_dict[Level_x_medical_test]]
    #check if dataset class info is consistent with the parameters
    if sorted(subtypes)!=sorted(cnames):
        raise Exception('dataset class info is not consistent with the parameters')
    X_train,y_train,X_test_final,y_test_final= split_data(dfo.copy(),train_size=train_size,test_size_final=test_size_final, first_shuffle=True,sampling_randomseed=randstateseed)
    X_train,y_train,X_test_final,y_test_final,dfs_train,dfs_test_final=preprocess_data(X_train,y_train,X_test_final,y_test_final,imputation='impute_withmedian_log_transform',ufc_normal_upper=  ufc_normal_upper)
    dataset_dict=create_datasubsets(X_train,y_train,X_test_final,y_test_final,classes)   
    dataset_dict=create_datasubsets(X_train,y_train,X_test_final,y_test_final,classes)   
    models=[] 
    for key in  dataset_dict[dataset_strategy][0].keys(): 
    #key code:
        def keystring(keytuple):
            keystr=str(keytuple)            
            for k in class_target_short_codes.keys():               
                keystr=keystr.replace(str(k),class_target_short_codes[k])
            return keystr   
        keystr=keystring(key)  
        print('\n\n**********************************************************************************************')
        print('  ALGORITHM AND MODEL SELECTION')
        print('  DATA ANALYSIS STRATEGY: {}   '.format(dataset_strategy))
        print('  ANALYSIS FOR  CLASS COMPARISON STRATEGY: {} '.format(key))
        print('**********************************************************************************************')     
        df_train=dataset_dict[dataset_strategy][0][key]
        X_train=df_train.drop(columns='types')
        y_train=df_train['types']  
        df_test=dataset_dict[dataset_strategy][1][key]
        X_test_final=df_test.drop(columns='types') 
        y_test_final=df_test['types'] 
        print('\n************DATA INFORMATION [Training]***************')
        print('Total data class counts in training data')
        print(df_train['types'].value_counts().values)
        print(' data size:',df_train.index.size)
        print('Features list',df_train.columns.values)
                # Initializing Classifiers
        DT_clf3 = DecisionTreeClassifier(random_state=randstateseed)
        DT_clf3_title = str(type(DT_clf3)).split('.')[-1][:-2]
        LREG_clf1 = LogisticRegression(solver='newton-cg',random_state=randstateseed,multi_class='auto')
        LREG_clf1_title = str(type(LREG_clf1)).split('.')[-1][:-2]
        KNN_clf2 = KNeighborsClassifier(algorithm='ball_tree')
        KNN_clf2_title = str(type(KNN_clf2)).split('.')[-1][:-2]       
        SVC_clf4 = SVC(random_state=randstateseed)
        SVC_clf4_title = str(type(SVC_clf4)).split('.')[-1][:-2]       
        RF_clf5=  RandomForestClassifier(random_state=randstateseed)
        RF_clf5_title = str(type(RF_clf5)).split('.')[-1][:-2]       
        GB_clf6=  GradientBoostingClassifier(random_state=randstateseed)
        GB_clf6_title = str(type(GB_clf6)).split('.')[-1][:-2]       
        ADA_clf7=  AdaBoostClassifier(random_state=randstateseed)
        ADA_clf7_title = str(type(ADA_clf7)).split('.')[-1][:-2]       
        LDA_clf8= LinearDiscriminantAnalysis()
        LDA_clf8_title = str(type(LDA_clf8)).split('.')[-1][:-2]        
        # Building the pipelines
        LREG_pipe1 = Pipeline([('LREG_clf1', LREG_clf1)])        
        KNN_pipe2 = Pipeline([('KNN_clf2', KNN_clf2)])        
        SVC_pipe4 = Pipeline([('SVC_clf4', SVC_clf4)])        
        LDA_pipe8 = Pipeline([('LDA_clf8', LDA_clf8)])       
        #LogisticRegression
        LREG_param_grid1 = [{'LREG_clf1__penalty': ['l2'],
                             'LREG_clf1__class_weight':[None,'balanced'],
                        'LREG_clf1__C': np.power(10., np.arange(-2, 2))}]
        #KNearest Neigbour
        KNN_param_grid2 = [{'KNN_clf2__n_neighbors': list(range(1, 10)),
                        'KNN_clf2__p': [1, 2]}]
        #Decision Tree
        DT_param_grid3 = [{'max_depth': [4,5,6],
                        'criterion': ['gini', 'entropy'],
                        'class_weight':[None,'balanced']}]
        #SVM Classifier
        SVC_param_grid4 = [{'SVC_clf4__kernel': ['rbf'],
                        'SVC_clf4__C': np.power(10., np.arange(-2, 2)),
                        'SVC_clf4__gamma': (np.power(10., np.arange(-2, 0))).tolist() + ['scale'],
                        'SVC_clf4__class_weight':[None,'balanced']},
                       {'SVC_clf4__kernel': ['linear'],
                        'SVC_clf4__C': np.power(10., np.arange(-2, 2)),'SVC_clf4__class_weight':[None,'balanced']}]
        #random forest search params
        RF_param_grid5=[{'class_weight': ['balanced','balanced_subsample'],
                    'criterion': ["gini",'entropy'],
                    'max_depth': [4,5,6],
                    'n_estimators': [50,100]}]
        #Gradiantboost
        GB_param_grid6=[{
                'learning_rate':[0.1,0.5,1],
                'n_estimators':[50,100],
                'max_depth':[4,5,6],
                'n_iter_no_change':[500,1000]}]
        #Adaptive Boosting
        ADA_param_grid7=[{
             'n_estimators':[50,100],
             'learning_rate':[0.1,0.5,1]}]
        #Linear Dsicriminant Analysis
        LDA_param_grid8=[
                {'LDA_clf8__solver':['lsqr','eigen'],
                'LDA_clf8__shrinkage': [0.1,0.5,1]
                }]
        models.append((LREG_param_grid1,LREG_pipe1,LREG_clf1_title))
        models.append((KNN_param_grid2,KNN_pipe2,KNN_clf2_title))
        models.append((DT_param_grid3,DT_clf3,DT_clf3_title))
        models.append((SVC_param_grid4,SVC_pipe4,SVC_clf4_title))
        models.append((RF_param_grid5,RF_clf5,RF_clf5_title))
        models.append((GB_param_grid6,GB_clf6,GB_clf6_title))
        models.append((ADA_param_grid7,ADA_clf7,ADA_clf7_title))
        models.append((LDA_param_grid8,LDA_pipe8,LDA_clf8_title))
        # Setting up multiple GridSearchCV objects, 1 for each algorithm
        gridcvs = {}
        inner_cv = StratifiedKFold(n_splits=inner_cv_k_fold_count, shuffle=False, random_state=randstateseed)
        #inner_cv = StratifiedShuffleSplit(n_splits=inner_cv_k_fold_count, random_state=randstateseed,train_size=cv_train_size,test_size=cv_test_size)
        scoring2 = {'f1': 'f1_macro','accuracy': 'accuracy','precision':'precision_macro','recall':'recall_macro','balanced_accuracy':'balanced_accuracy'}    
        scoring1 = {'accuracy': 'f1_macro'}    
        print("\n************NESTED CROSS VALIDATION RESULTS*****************")     
        for rp in range(n_repeats): #reperion loop
            print("REPETITION {}/{}".format(rp+1,n_repeats))            
            #randomly shuffle the dataset
            X_train,y_train=shuffle(X_train,y_train)        
            for pgrid, algo, name in models:
                gcv = GridSearchCV(estimator=algo,
                                   param_grid=pgrid,
                                   scoring=scoring1,
                                   n_jobs=1,
                                   cv=inner_cv,
                                   verbose=0,
                                   iid=False,
                                   refit='accuracy')
                gridcvs[name] = gcv                
            outer_cv = StratifiedKFold(n_splits=outer_cv_k_fold_count, shuffle=False, random_state=randstateseed)
            #outer_cv = StratifiedShuffleSplit(n_splits=50, random_state=randstateseed,train_size=cv_train_size,test_size=cv_test_size)
            if STEP1_flag==True:  
                print("outer k-fold count:{} inner k-fold count:{}".format(outer_cv_k_fold_count,inner_cv_k_fold_count))
                for algo_title, gs_est in gridcvs.items():
                    nested_score = cross_validate(gs_est, 
                                                   X=X_train, 
                                                   y=y_train, 
                                                   cv=outer_cv,
                                                   scoring=scoring2,
                                                   return_train_score=True,
                                                   return_estimator=True,
                                                   n_jobs=1)
                    print('\nAlgorithm','\t\t\t','f1 score (Test)','\t\t','Accuracy (Test)')
                    print(algo_title,'\t\t',format(nested_score['test_f1'].mean(),'.3f'),u"\u00B1",format(nested_score['test_f1'].std(),'.3f'),'\t\t',format(nested_score['test_accuracy'].mean(),'.3f'),u"\u00B1",format(nested_score['test_accuracy'].std(),'.3f'))
                         
                    for it in range(outer_cv_k_fold_count):
                        #fill result table
                        table_entry[0]=rp        
                        table_entry[1]=algo_title
                        table_entry[2]=dataset_strategy
                        table_entry[3]=Level_x_medical_test
                        table_entry[4]=keystr
                        table_entry[5]="Train"
                        table_entry[6]=nested_score['train_f1'][it]
                        table_entry[7]=nested_score['train_accuracy'][it]
                        table_entry[8]=nested_score['train_precision'][it]
                        table_entry[9]=nested_score['train_recall'][it]
                        table_entry[10]=nested_score['train_balanced_accuracy'][it]
                        gscv=nested_score['estimator'][it]
                        table_entry[11]=gscv.best_estimator_.get_params()                  
                        df_res = df_res.append(pd.Series(table_entry, index=df_res.columns ), ignore_index=True)
                        #test entries
                        table_entry[0]=rp   
                        table_entry[1]=algo_title
                        table_entry[2]=dataset_strategy
                        table_entry[3]=Level_x_medical_test
                        table_entry[4]=keystr
                        table_entry[5]="Test"
                        table_entry[6]=nested_score['test_f1'][it]
                        table_entry[7]=nested_score['test_accuracy'][it]
                        table_entry[8]=nested_score['test_precision'][it]
                        table_entry[9]=nested_score['test_recall'][it]
                        table_entry[10]=nested_score['test_balanced_accuracy'][it]
                        gscv=nested_score['estimator'][it]
                        table_entry[11]=gscv.best_estimator_.get_params()                  
                        df_res = df_res.append(pd.Series(table_entry, index=df_res.columns ), ignore_index=True)   
            if STEP2_flag==True: 
    #           for algo_title, gs_est in gridcvs.items():
                for gs_est in [gridcvs[RF_clf5_title]]: ##select Random forest
                    algo_title=RF_clf5_title
                    print("\n************TRAINING DATA RESULTS*****************")                                   
                    model_title = algo_title
                    print('\n',model_title.upper())    
                    print('\n************FINDING BEST PARAMETERS***************')
                                    ##All train data
                    print("Best Parameters on all train data") 
                    print(algo_title+":" )
                    gs_est.fit(X=X_train, y=y_train)
                    print("Best grid search CV Parameters: \n{}\n".format(gs_est.best_params_)) 
                    print("Best grid search CV Score: \n{}\n".format(gs_est.best_score_))                    
                    best_est=gs_est.best_estimator_
                    best_params=best_est.get_params()
                    best_gridCV_meanscore=gs_est.best_score_                   
                    best_table_entry[32]=best_gridCV_meanscore                                       
                    model=best_est                   
                    X=X_train
                    y=y_train
                
                    if cross_validaiton_flag:
                        print('\n************CROSS VALIDATION***************')
                        cv_test_mean_f1, cv_test_std95_f1,cv_test_mean_accuracy,cv_test_std95_accuracy=crossvalidation(model,X,y,kfold=outer_cv_k_fold_count,trainsize=cv_train_size,testsize=cv_test_size,sampling_randomseed=randstateseed)
                        best_table_entry[28]=cv_test_mean_f1
                        best_table_entry[29]=cv_test_std95_f1
                        best_table_entry[30]=cv_test_mean_accuracy
                        best_table_entry[31]=cv_test_std95_accuracy                    
                    y_pred=best_est.predict(X_train)
                    acc_score = accuracy_score(y_train,y_pred)
                    f1score,precision,recall,support=precision_recall_fscore_support(y_train,y_pred,average='macro')
                    bacc_score=balanced_accuracy_score(y_train,y_pred)                    
                    model_details = model_title
                    if hasattr(model, "estimators_"):
                        model_details += "({} estimators".format(len(model.estimators_))
                    else:
                        model_details += "([] estimators"                    
                    print(model_details ,"features:") 
                    print('\nModel','\t\t\t','f1 score','\t','Accuracy')
                    print(model_title+'\t',f1score,'\t\t',acc_score)                    
                    best_table_entry[0]=rp   
                    best_table_entry[1]=algo_title
                    best_table_entry[2]=dataset_strategy
                    best_table_entry[3]=Level_x_medical_test
                    best_table_entry[4]=keystr
                    best_table_entry[5]="All Train after nested CV"
                    best_table_entry[6]=f1score
                    best_table_entry[7]=acc_score
                    best_table_entry[8]=precision
                    best_table_entry[9]=recall
                    best_table_entry[10]=bacc_score
                    best_table_entry[11]=best_params                                 
                    if learning_curves_flag:
                        # Cross validation with N iterations to get smoother mean test and train
                        # score curves, each time with M% data randomly selected as a validation set.
                        #cv = ShuffleSplit(n_splits=50, test_size=0.2)
                        #cv=StratifiedKFold(n_splits=crossval_kfold)
                        cv = StratifiedShuffleSplit(n_splits=20,train_size=train_size,test_size=test_size_final,random_state=None)
                        gen_learning_curve(model, key,model_title+" [Training] "+Level_x_medical_test, X, y, (0.0, 1.01), cv=cv, n_jobs=4,scoring='f1_macro',plotshow=True)
                    if auc_roc_with_cv_flag:
                        cv = StratifiedKFold(n_splits=crossval_kfold)
                        mean_auc,std_auc=gen_auc_roc_with_cv(model,key, X,y,model_title+" [Training] "+Level_x_medical_test,cv,plotshow=True)
                        best_table_entry[25]=mean_auc
                        best_table_entry[26]=std_auc
                    if pr_curve_flag:
                        f1sc,pr_auc,avg_pr=gen_pr_curve(model,key,X,y,model_title+" [Training] "+Level_x_medical_test,plotshow=True)
                        best_table_entry[23]=pr_auc
                        best_table_entry[24]=avg_pr                    
                    if classification_report_flag:
                        print('\n',classification_report(y_train, y_pred,digits=3,target_names=[class_target_codes[s]  for s in key]))
                        print('')
                    if confusion_matrix_flag:
                       # Plot non-normalized confusion matrix
                       cm=gen_confusion_matrix(classifier=model,key=key,y_true=y_train, y_pred=y_pred, model_title=model_title+" [Training] "+Level_x_medical_test, show_figures=show_figures)
                       best_table_entry[12]=cm.tolist()
                    if binary_classification_metrics_flag:
                        F1,ACC,TPR,TNR,PPV,NPV,FPR,FNR,FDR=calculate_binary_classification_metrics(classifier=model, key=key,y_true=y_train, y_pred=y_pred)
                        best_table_entry[13]=F1
                        best_table_entry[14]=ACC
                        best_table_entry[15]=TPR
                        best_table_entry[16]=TNR
                        best_table_entry[17]=PPV
                        best_table_entry[18]=NPV
                        best_table_entry[19]=FPR
                        best_table_entry[20]=FNR
                        best_table_entry[21]=FDR
                    if len(model.classes_)>2: #multiclass
                        dfe=calculate_multiclass_classifications_metrics(y_true=y_train,y_pred=y_pred,key=key,target_names=[class_target_codes[s]  for s in key])
                        print("Multiclass report:\n")
                        print(dfe)
                    if auc_roc_flag:
                        roc_auc=gen_roc_curve(model,key,X,y,model_title+" [Testing] "+Level_x_medical_test,plotshow=True)
                        best_table_entry[22]=roc_auc    
                    if gen_feature_importances_flag:
                        if (hasattr(model, 'feature_importances_') and model_title=='RandomForestClassifier'):
                           print(model_title)
                           imp_f=gen_feature_importances(classifier=model,key=key,model_title=model_title+" [Training] "+Level_x_medical_test,X=X_train,y_test_final=y, y_pred_final=y_pred,feature_index=X_train.columns,show_figures=show_figures)
                           best_table_entry[27]=imp_f                        
                    best_df_res = best_df_res.append(pd.Series(best_table_entry, index=best_df_res.columns ), ignore_index=True)   
                    if len(model.classes_)!=2: #multiclass strategy
                        print (best_df_res.iloc[:,6:11].round(3).tail(1))
                    else:
                        print (best_df_res.iloc[:,13:24].round(3).tail(1))                        
                    print("\n************TEST DATA RESULTS*****************")   
                    print('\n************DATA INFORMATION [Test]***************')
                    print('Total data class counts in test data')
                    print(df_test['types'].value_counts().values)
                    print('data size:',df_test.index.size)
                    print('Features list',df_test.columns.values)       
 
                    y_pred=best_est.predict(X_test_final)
                    
                    y_pred_final=y_pred
                    acc_score = accuracy_score(y_test_final,y_pred)
                    f1score,precision,recall,support=precision_recall_fscore_support(y_test_final,y_pred,average='macro')
                    bacc_score=balanced_accuracy_score(y_test_final,y_pred)                    
                    print('\nModel','\t\t\t','f1 score','\t','Accuracy')
                    print(model_title+'\t',f1score,'\t\t',acc_score)                
                    best_table_entry[0]=rp   
                    best_table_entry[1]=algo_title
                    best_table_entry[2]=dataset_strategy
                    best_table_entry[3]=Level_x_medical_test
                    best_table_entry[4]=keystr
                    best_table_entry[5]="Final Test data after nested CV"
                    best_table_entry[6]=f1score
                    best_table_entry[7]=acc_score
                    best_table_entry[8]=precision
                    best_table_entry[9]=recall
                    best_table_entry[10]=bacc_score
                    best_table_entry[11]=best_params              
                    if classification_report_flag:
                        print('\n',classification_report(y_test_final, y_pred_final,digits=3,target_names=[class_target_codes[s]  for s in key]))
                        print('')                   
                    if confusion_matrix_flag:
                       # Plot non-normalized confusion matrix
                       cm=gen_confusion_matrix(classifier=model,key=key,y_true=y_test_final, y_pred=y_pred_final, model_title=model_title+" [Testing] "+Level_x_medical_test, show_figures=show_figures)
                       best_table_entry[12]=cm.tolist()
                    if binary_classification_metrics_flag:
                        F1,ACC,TPR,TNR,PPV,NPV,FPR,FNR,FDR=calculate_binary_classification_metrics(classifier=model, key=key,y_true=y_test_final, y_pred=y_pred_final)
                        best_table_entry[13]=F1
                        best_table_entry[14]=ACC
                        best_table_entry[15]=TPR
                        best_table_entry[16]=TNR
                        best_table_entry[17]=PPV
                        best_table_entry[18]=NPV
                        best_table_entry[19]=FPR
                        best_table_entry[20]=FNR
                        best_table_entry[21]=FDR
                    if len(model.classes_)>2: #multiclass
                        dfe=calculate_multiclass_classifications_metrics(y_true=y_test_final,y_pred=y_pred_final,key=key,target_names=[class_target_codes[s]  for s in key])
                        print("Multiclass report:\n")
                        print(dfe)                                                          
                    if auc_roc_flag:
                        roc_auc=gen_roc_curve(model,key,X_test_final,y_test_final,model_title+" [Testing] "+Level_x_medical_test,plotshow=True)
                        best_table_entry[22]=roc_auc                         
                    if pr_curve_flag:
                        f1sc,pr_auc,avg_pr=gen_pr_curve(model,key,X_test_final,y_test_final,model_title+" [Testing] "+Level_x_medical_test,plotshow=True)
                        best_table_entry[23]=pr_auc
                        best_table_entry[24]=avg_pr
                    best_df_res = best_df_res.append(pd.Series(best_table_entry, index=best_df_res.columns ), ignore_index=True)   
                    if len(model.classes_)!=2: #multiclass strategy
                        print (best_df_res.iloc[:,6:11].round(3).tail(1))
                    else:
                        print (best_df_res.iloc[:,13:24].round(3).tail(1))                        
                    print('\n************FINAL MODEL***************')
                    
                    ##All data 
                    X=pd.concat((X_train,X_test_final), axis=0)
                    y=pd.concat((y_train,y_test_final),axis=0)
                    

                    _,feat_imputation_values=impute_withmedian_log_transform(dfo.drop(columns=dfo.columns.values[-1]),ufc_normal_upper=  ufc_normal_upper) 
                    print("Best Parameters on all data")
                    print(algo_title+":" )
                    gs_est.fit(X=X, y=y)                    
                    final_model=gs_est.best_estimator_
                    final_model_params=best_est.get_params()
                    final_model_score=gs_est.best_score_
                    
                    #save the model to a file
                    modelsave=final_model
                    filename_model = "Final_CSprediction_model"+"_"+dataset_strategy+"_"+Level_x_medical_test+".pkl"
                    pickle.dump(modelsave, open(filename_model, 'wb'))
                    filename_imputa = "Final_CSprediction_model_imputation_params"+"_"+dataset_strategy+"_"+Level_x_medical_test+".pkl"
                    pickle.dump(feat_imputation_values,open(filename_imputa, 'wb'))
                    
                    
                    #model_loaded = pickle.load(open(filename_model,'rb'))
                    #print(model.predict(X_test_final))
               
########## WRITE TO FILE ###############3
    file_title="cushing NESTED CV results"+"_"+dataset_strategy+"_"+Level_x_medical_test+"_rep"+str(n_repeats)+"nestedCV_"+str(outer_cv_k_fold_count)+"x"+str(inner_cv_k_fold_count)+"_s"+str(randstateseed)+".csv"        
    print("writing to file"+file_title)
    df_res.to_csv(file_title)
########## WRITE TO FILE ###############3
    file_title="cushing best param results after nested CV"+"_"+dataset_strategy+"_"+Level_x_medical_test+"_rep"+str(n_repeats)+"nestedCV_"+str(inner_cv_k_fold_count)+"_s"+str(randstateseed)+".csv"        
    print("writing to file"+file_title)
    best_df_res.to_csv(file_title)
    
    return df_res,best_df_res,dfs_train,dfs_test_final

# In[]

table_labels=['ReptNo','Model','Strategy','Diagnostic_Level','Classes','Train/Test','f1_macro Score','Accuracy','Precision_macro','Recall_macro','Balanced_Accuracy','Confusion_Matrix','CV fold:Best Parameters']
df_result_table_BATCH=pd.DataFrame({},columns=table_labels)
df_list=[]
best_table_labels=['ReptNo','Model','Strategy','Diagnostic_Level','Classes','Train/Test','f1_macro Score','Accuracy','Precision_macro','Recall_macro','Balanced_Accuracy','CV fold:Best Parameters','Confusion_Matrix','F1','ACC','TPR','TNR','PPV','NPV','FPR','FNR','FDR','ROC_AUC','PRC_AUC','PRC Avg Precision','CV AUC mean','CV AUC std','Important Features','cv_test_mean_f1', 'cv_test_std95_f1','cv_test_mean_accuracy','cv_test_std95_accuracy','best_gridCV_meanscore']
best_df_result_table_BATCH=pd.DataFrame({},columns=best_table_labels)
best_df_list=[]
number_of_cv_repeat=1
outer_cv_k_fold_count=5
inner_cv_k_fold_count=3
sampleseed=randstateseed    


df,best_df,dfs_train,dfs_test_final=cushing_analysis(datafilename,"ONE2RESTREV","Stage1",n_repeats=number_of_cv_repeat,outer_cv_k_fold_count=outer_cv_k_fold_count,inner_cv_k_fold_count=inner_cv_k_fold_count,randstateseed=sampleseed)
df_list.append(df)
best_df_list.append(best_df)
df,best_df,dfs_train,dfs_test_final=cushing_analysis(datafilename,"ALLIN","Stage2",n_repeats=number_of_cv_repeat,outer_cv_k_fold_count=outer_cv_k_fold_count,inner_cv_k_fold_count=inner_cv_k_fold_count,randstateseed=sampleseed)
df_list.append(df)
best_df_list.append(best_df)
# df,best_df,dfs_train,dfs_test_final=cushing_analysis(datafilename,"ALLIN","Stage1",n_repeats=number_of_cv_repeat,outer_cv_k_fold_count=outer_cv_k_fold_count,inner_cv_k_fold_count=inner_cv_k_fold_count,randstateseed=sampleseed)
# df_list.append(df)
# best_df_list.append(best_df)   
# df,best_df,dfs_train,dfs_test_final=cushing_analysis(datafilename,"ONE2REST","Stage1",n_repeats=number_of_cv_repeat,outer_cv_k_fold_count=outer_cv_k_fold_count,inner_cv_k_fold_count=inner_cv_k_fold_count,randstateseed=sampleseed)
# df_list.append(df)
# best_df_list.append(best_df)
# df,best_df,dfs_train,dfs_test_final=cushing_analysis(datafilename,"ONE2RESTREV","Stage2",n_repeats=number_of_cv_repeat,outer_cv_k_fold_count=outer_cv_k_fold_count,inner_cv_k_fold_count=inner_cv_k_fold_count,randstateseed=sampleseed)
# df_list.append(df)
# best_df_list.append(best_df)
# df,best_df,dfs_train,dfs_test_final=cushing_analysis(datafilename,"ONE2REST","Stage2",n_repeats=number_of_cv_repeat,outer_cv_k_fold_count=outer_cv_k_fold_count,inner_cv_k_fold_count=inner_cv_k_fold_count,randstateseed=sampleseed)
# df_list.append(df)
# best_df_list.append(best_df)
# df,best_df,dfs_train,dfs_test_final=cushing_analysis(datafilename,"ONE2ONE","Stage1",n_repeats=number_of_cv_repeat,outer_cv_k_fold_count=outer_cv_k_fold_count,inner_cv_k_fold_count=inner_cv_k_fold_count,randstateseed=sampleseed)
# df_list.append(df)
# best_df_list.append(best_df)
# df,best_df,dfs_train,dfs_test_final=cushing_analysis(datafilename,"ONE2ONE","Stage2",n_repeats=number_of_cv_repeat,outer_cv_k_fold_count=outer_cv_k_fold_count,inner_cv_k_fold_count=inner_cv_k_fold_count,randstateseed=sampleseed)
# df_list.append(df)
# best_df_list.append(best_df)
df_result_table = pd.concat(df_list, ignore_index=True, sort=False)
df_result_table_BATCH=pd.concat([df_result_table_BATCH,df_result_table], ignore_index=True, sort=False)            
file_title="BATCH ALL cushing CV analysis results"+".csv" 
print("writing to file"+file_title)
df_result_table_BATCH.to_csv(file_title)    
best_df_result_table = pd.concat(best_df_list, ignore_index=True, sort=False)
best_df_result_table_BATCH=pd.concat([best_df_result_table_BATCH,best_df_result_table], ignore_index=True, sort=False)            
file_title="BATCH ALL cushing best param analysis results after NestedCV"+".csv" 
print("writing to file"+file_title)
best_df_result_table_BATCH.to_csv(file_title)    
