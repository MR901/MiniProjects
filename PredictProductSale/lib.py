import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix,  roc_curve
from sklearn.metrics import matthews_corrcoef,accuracy_score
import matplotlib.patches as patches
from collections import defaultdict
from pandas.plotting import table
from matplotlib.gridspec import GridSpec


## Doing Primary analysis
def getMissingPct(df):
    '''    '''
    return [ round((df['IsNullSum'][i] / max(df['count']))*100,2) for i in range(len(df)) ]

def genMsg(txt):
    '''    '''
    print('_'*12+'| Number of feature/s which are {} : {} |'.format(*txt)+'_'*12)

def datasetPrimAnalysis(DF, msg=True):
    '''
    Function which is used to analyze features and provide data insight
    '''
    df_explore = DF.copy()
    if msg: print('Overall dataset shape :', df_explore.shape)
    
    ## Creating a dataset that explain feature/s
    temp = pd.DataFrame(df_explore.isnull().sum(), columns = ['IsNullSum'])
    temp['dtypes'] = df_explore.dtypes.tolist()
    temp['IsNaSum'] = df_explore.isna().sum().tolist()
    
    ## Analyzing Time based Features
    temp_tim = temp.loc[temp['dtypes']=='datetime64[ns]' ,:]
    if (len(temp_tim) > 0):
        df_tim = df_explore.loc[:,temp_tim.index].fillna('Missing-NA')
        if msg: genMsg(['Time based', df_tim.shape[1]])
        temp_tim = temp_tim.join(df_tim.describe().T).fillna('')
        temp_tim['%Missing'] = getMissingPct(temp_tim)
        if msg: display(temp_tim)
    
    
    ## Analyzing Qualitative Features
    temp_cat = temp.loc[temp['dtypes']=='O' ,:]
    if (len(temp_cat) > 0):
        df_cat = df_explore.loc[:,temp_cat.index].fillna('Missing-NA')
        if msg: genMsg(['Qualitative', df_cat.shape[1]])
        temp_cat = temp_cat.join(df_cat.describe().T).fillna('')
        temp_cat['CategoriesName'] = [ list(df_cat[fea].unique()) for fea in temp_cat.index ]
        temp_cat['%Missing'] = getMissingPct(temp_cat)
        if msg: display(temp_cat)
    
    
    ## Analyzing Quantitative Features
    temp_num = temp.loc[((temp['dtypes']=='int') | (temp['dtypes']=='float')),:]
    if (len(temp_num) > 0):
        df_num = df_explore.loc[:,temp_num.index]#.fillna('Missing-NA')
        if msg: genMsg(['Quantitative', df_num.shape[1]])
        temp_num = temp_num.join(df_num.describe().T).fillna('')
        temp_num['%Missing'] = getMissingPct(temp_num)
        if msg: display(temp_num)
    # if temp_cat['dtypes'][i] == 'float', 'int', 'O'

    if len(temp)!= len(temp_tim) + len(temp_cat) + len(temp_num):
        print("Some columns data is missing b/c of data type")
    
    dit = {'TimeBased': temp_tim, 'Categorical': temp_cat, 'Numerical': temp_num}
    return dit


## splitting the dataset
def namestr(obj, namespace): #namestr(primDf, globals())[0]
    return [name for name in namespace if namespace[name] is obj]

def splitTimeSeriesData(df, split_date, msg =True):
    ''' Split DataFramee based on Split Date
    '''
    df_train = df.loc[df['Date'] <= split_date]
    df_test = df.loc[df['Date'] > split_date]
    if msg: print('Original DataFrame Shape: {} \n\tTrain Shape: {}\tTest Shape:{}'.format(
        df.shape, df_train.shape, df_test.shape))
    return df_train, df_test



## Key Generator
def createKey(DF, Key_ColToUse):
    '''
    Use to combine columns to generate a key which is seperated by '|'
    '''
    df = DF.copy()
    for col_ind in range(len(Key_ColToUse)):
        I1 = df.index.tolist()
        I2 = df[Key_ColToUse[col_ind]].astype('str').tolist()
        if col_ind == 0:
            df.index = I2
        else:
            df.index = [ "|".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ] #, I3[ind]
    return df.index 



## Defining Scaling DF
class ScalingDF:
    '''
    This class can be used for scaling features. 
    For the Train Cycle, feat_info_dict (i.e. information aboout the features) shall be 'None' or undefined 
    For the Predict Cycle, feat_info_dict (i.e. information aboout the features) MUST be provided.
        - This feat_info_dict is obtained ad an additional argument the some scaling is done
        - Can again be obtained using 'getInitialFeaturesDescriptiveStats' for the same instance
    
    "getInitialFeaturesDescriptiveStats" provide descriptive information on the in DF on which 'ScalingDF' 
                                        was initialized
    "generateNewFeaturesDescriptiveStats" provides descriptive information on the features after they have 
                                        been tranformed byy any method
    '''
    def __init__(self, df, feat_info_dict = None):
        
        df = df.copy()
        if feat_info_dict is None:
            ## Computing Measures used for Scaling
            feat_info_dict = {}
            for col in df.columns:
                feat_info_dict[col] = {'Min': df[col].min(),
                                       'Median': df[col].median(), 
                                       'Max': df[col].max(), 
                                       'Mean': df[col].mean(), 
                                       'Std': df[col].std()}
        else:
            ## Check if columns are matching if nnot raise ann error
            colNotPresent = len([ False for ele in feat_info_dict.keys() if ele not in df.columns ])
            if colNotPresent > 0:
                raise Exception('Feature that is to be scaled is not present in the provided DF')
        
        self.df = df
        self.feat_info_dict = feat_info_dict
    
    def getInitialFeaturesDescriptiveStats(self):
        return self.feat_info_dict
    def generateNewFeaturesDescriptiveStats(self):
        feat_info_dict = {}
        for col in self.df.columns:
            feat_info_dict[col] = {'Min': self.df[col].min(),
                                   'Median': self.df[col].median(), 
                                   'Max': self.df[col].max(), 
                                   'Mean': self.df[col].mean(), 
                                   'Std': self.df[col].std()}
        return feat_info_dict
        
    def normalization(self):
        print('Scaling dataframe using {} scaler'.format('Normalization'))
        for col in self.df.columns:
            print('|\t', col)
            li = list(self.df[col])
            self.df[col] = [ (elem - self.feat_info_dict[col]['Min']) / \
                            (self.feat_info_dict[col]['Max'] - self.feat_info_dict[col]['Min']) \
                            for elem in li ] 
        return self.df, self.feat_info_dict
    
    def standardization(self):
        print('Scaling dataframe using {} scaler'.format('Standardization'))
        for col in self.df.columns:
            print('|\t', col)
            li = list(self.df[col])
            self.df[col] = [ (elem - self.feat_info_dict[col]['Mean']) / self.feat_info_dict[col]['Std']\
                            for elem in li ]
        return self.df, self.feat_info_dict
    
    def standard_median(self):
        print('Scaling dataframe using {} scaler'.format('Standard_Median'))
        for col in self.df.columns:
            print('|\t', col)
            li = list(self.df[col])
            self.df[col] = [ (elem - self.feat_info_dict[col]['Median']) / self.feat_info_dict[col]['Std'] \
                            for elem in li ] 
        return self.df, self.feat_info_dict

# A = ScalingDF(df.loc[:, df_info['Numerical'].index ])
# i1 = A.getInitialFeaturesDescriptiveStats()
# i2 = A.generateNewFeaturesDescriptiveStats()
# newDF, descStatsDict = A.standardization()
# f1 = A.getInitialFeaturesDescriptiveStats()
# f2 = A.generateNewFeaturesDescriptiveStats()
# i1 == i2, i1==f1, i2==f2, f1==f2





# -------------------------------------------------<< Visualization Related >>------------------------------------------ #

## Plotting Confusion Matrix
def classReport2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)
    
    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data

def metricCalculation(trueLabel, predLabel):
    ''' . '''
    tn, fp, fn, tp = confusion_matrix(trueLabel, predLabel).ravel()

    PCP = tp + fp # Predicted Condition Positive
    PCN = fn + tn # Predicted Condition Negative
    TCP = tp + fn # True Condition Positive
    TCN = fp + tn # True Condition Negative

    TotPop = tp + fp + fn + tn # Total Population

    accuracy = round((tp + tn) / TotPop, 3)
    Prevalence = round((tp + fn) / TotPop, 3)  # TCP/TotPop
    Precision, FalseDiscoveryRate = (round(tp/PCP,3), round(fp/PCP,3)) if PCP != 0 else ('NaN','NaN')
    FalseOmissionRate, NegativePredictiveValue = (round(fn/PCN,3), round(tn/PCN,3)) if PCN != 0 else ('NaN','NaN')
    
    # Precision = tp / PCP #tp/(fp + tp); FalseDiscoveryRate = fp / PCP #fp/(fp + tp)
    # FalseOmissionRate = fn / PCN #fn/(tn + fn); NegativePredictiveValue = tn / PCN #tn/(tn + fn)
    # TruePositiveRate = tp / TCP; FalseNegativeRate = fn / TCP
    # FalsePositiveRate = fp / TCN;  TrueNegativeRate = tn / TCN
    
    TruePositiveRate, FalseNegativeRate = (round(tp/TCP,3), round(fn/TCP,3)) if TCP != 0 else ('NaN','NaN')
    FalsePositiveRate, TrueNegativeRate = (round(fp/TCN,3), round(tn/TCN,3)) if TCN != 0 else ('NaN','NaN')
        
    PositiveLikelihoodRatio = round(TruePositiveRate / FalsePositiveRate,3) if (type(FalsePositiveRate)!=str) & (FalsePositiveRate !=0) else 'NaN'
    NegativeLikelihoodRatio = round(FalseNegativeRate / TrueNegativeRate,3) if (type(TrueNegativeRate)!=str) & (TrueNegativeRate !=0) else 'NaN'
    
    DiagnosticOddsRatio = round(PositiveLikelihoodRatio/NegativeLikelihoodRatio,3) if (type(PositiveLikelihoodRatio)!=str)&(type(NegativeLikelihoodRatio)!=str)&(NegativeLikelihoodRatio!=0) else 'NaN'
    F1Score = (2*tp) / (2*tp + fp + fn)

    MattheawCorrelationCoefficient = round((tp*tn - fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2),3) if ((tp+fp)!=0)&((tp+fn)!=0)&((tn+fp)!=0)&((tn+fn)!=0) else 'NaN'
    
    metric = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'TotPop': TotPop,
              'PCP': PCP, 'PCN': PCN, 'TCP': TCP, 'TCN': TCN,
              'Acc': accuracy, 'Prev':Prevalence, 'Prec':Precision,
              'FDR': FalseDiscoveryRate, 'FOR': FalseOmissionRate, 
              'NPV': NegativePredictiveValue,
              'TPR': TruePositiveRate, 'FPR': FalsePositiveRate,
              'FNR': FalseNegativeRate, 'TNR': TrueNegativeRate,
              'PLR': PositiveLikelihoodRatio, 'NLR': NegativeLikelihoodRatio,
              'DOR': DiagnosticOddsRatio, 'F1S': F1Score,
              'MCC': MattheawCorrelationCoefficient
             }
    return metric

## Plotting Confusion Matrix
def plotConfusionMatrix(y_act, y_pred):
    '''
    To print Multiple metrics related to classification problem
    '''
    mtc = metricCalculation(trueLabel=y_act, predLabel=y_pred)
    fig = plt.figure(figsize=(15,8))
    
    gs=GridSpec(3,2) # 3 rows, 2 columns
    ax1 = fig.add_subplot(gs[:2,:])
    ## Generate Confusion Matrix
    # ax.imshow(df, interpolation='nearest', cmap=plt.cm.gray)
    # Draw the grid boxes
    ax1.set_xlim(-0.5,2.5)
    ax1.set_ylim(2.5,-0.5)
    #ax.plot([x1,x2],[y1,y2], '-k', lw=2)
    # HorizontalLines
    ax1.plot([ 0.5,1.5],[-0.5,-0.5], '-k', lw=4) ## border
    ax1.plot([ 0.0,2.5],[0.0,0.0], '-k', lw=2)
    ax1.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax1.plot([ 0.0,2.5],[1.0,1.0], '-k', lw=2)
    ax1.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax1.plot([ 0.5,2.0],[2.0,2.0], '-k', lw=2)
    ax1.plot([ 0.5,2.5],[2.5,2.5], '-k', lw=4) ## border
    # Vertical Line
    ax1.plot([-0.5,-0.5], [0.5,1.5], '-k', lw=4) ## border
    ax1.plot([0.0,0.0], [ 0.0,1.5], '-k', lw=2)
    ax1.plot([0.5,0.5], [-0.5,2.5], '-k', lw=2)
    ax1.plot([1.0,1.0], [ 0.0,2.5], '-k', lw=2)
    ax1.plot([1.5,1.5], [-0.5,2.5], '-k', lw=2)
    ax1.plot([2.0,2.0], [ 0.0,2.5], '-k', lw=2)
    ax1.plot([2.5,2.5], [ 0.0,2.5], '-k', lw=4) ## border
    # Creating a box
    ax1.plot([0.5,1.5], [0.5,0.5], '-k', lw=4) ##Horiz line
    ax1.plot([0.5,1.5], [1.5,1.5], '-k', lw=4) ##Horiz line
    ax1.plot([0.5,0.5], [0.5,1.5], '-k', lw=4) ##Vert line
    ax1.plot([1.5,1.5], [0.5,1.5], '-k', lw=4) ##Vert line
    ax1.set_facecolor('w')
    # Setting Headings
    ax1.text(1.0, -0.25, s = 'True Condition', fontsize=18, color = 'w',  va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
    ax1.text(-0.25, 1.0, s = 'Predicted\nCondition', fontsize=18, color = 'w', va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
    ax1.text(0.25, 0.75, s = 'Condition\nPositive', fontsize=18, color = 'k', va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
    ax1.text(0.25, 1.25, s = 'Condition\nNegative', fontsize=18, color = 'k', va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
    ax1.text(0.75, 0.25, s = 'Condition\nPositive', fontsize=18, color = 'k', va='center', ha='center', bbox=dict(fc='k', alpha=0, boxstyle='round,pad=1'))
    ax1.text(1.25, 0.25, s = 'Condition\nNegative', fontsize=18, color = 'k', va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
    # Values box
    ax1.text(0.75,0.75, 'True Pos: {}'.format(mtc['TP']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
    ax1.text(1.25,0.75, 'False Pos: {}\nType I error'.format(mtc['FP']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
    ax1.text(0.75,1.25, 'False Neg: {}\nType II error'.format(mtc['FN']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
    ax1.text(1.25,1.25, 'True Neg: {}'.format(mtc['TN']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
    ax1.text(1.75,0.25, 'Prevalence: {}'.format(mtc['Prev']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(2.25,0.25, 'Accuracy: {}'.format(mtc['Acc']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(1.75,0.75, 'Pos Pred Val,\n Precision: {}'.format(mtc['Prec']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(2.25,0.75, 'False Discovery Rate,\n FDR: {}'.format(mtc['FDR']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(1.75,1.25, 'False Omission Rate,\n FOR: {}'.format(mtc['FOR']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(2.25,1.25, 'Neg Pred Value,\n NDV: {}'.format(mtc['NPV']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(0.75,1.75, 'True Pos Rate,\n TPR: {}'.format(mtc['TPR']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(0.75,2.25, 'False Pos Rate,\n FPR: {}'.format(mtc['FPR']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(1.25,1.75, 'False Neg Rate,\n FNR: {}'.format(mtc['FNR']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(1.25,2.25, 'True Neg Rate,\n TNR: {}'.format(mtc['TNR']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(1.75,1.75, 'Pos Likelihood Ratio,\n LR+: {}'.format(mtc['PLR']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(1.75,2.25, 'Neg Likelihood Ratio,\n LR-: {}'.format(mtc['NLR']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(2.25,1.75, 'Diag Odds Rat,\n DOR: {}'.format(mtc['DOR']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(2.25,2.25, 'F1 Score: {}'.format(round(mtc['F1S'],3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    ax1.text(0.25,0.25, 'Matthews Corr Coeff,\n MCC: {}'.format(mtc['MCC']), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
    # ax.bbox([[0.08, 0.125], [0.95, 0.88]], facecolor='0.2', alpha=0.5)
    ax1.add_patch( patches.Rectangle( (0.5, -0.5), 1.0, 0.5, facecolor='k', alpha=1.0) )
    ax1.add_patch( patches.Rectangle( (-0.5, 0.5), 0.5, 1.0, facecolor='k', alpha=1.0) )
    ax1.add_patch( patches.Rectangle( (0.5, 0.0), 0.5, 0.5, facecolor='b', alpha=0.7) )
    ax1.add_patch( patches.Rectangle( (1.0, 0.0), 0.5, 0.5, facecolor='orange', alpha=0.7) )
    ax1.add_patch( patches.Rectangle( (0.0, 0.5), 0.5, 0.5, facecolor='b', alpha=0.7) )
    ax1.add_patch( patches.Rectangle( (0.0, 1.0), 0.5, 0.5, facecolor='orange', alpha=0.7) )
    # ax.add_patch( patches.Rectangle( (0.5, -0.5), 0.5, 1.0, facecolor='0.2', alpha=0.5) )
    # ax.axhspan(1, 2.25)
    ax1.axis('off')
    # Creating a box for other measure
    ax1.plot([-0.45, 0.45], [1.6,1.6], '-k', lw=2) ##Horiz line
    ax1.plot([-0.45, 0.45], [2.4,2.4], '-k', lw=2) ##Horiz line
    ax1.plot([-0.45,-0.45], [1.6,2.4], '-k', lw=2) ##Vert line
    ax1.plot([ 0.45, 0.45], [1.6,2.4], '-k', lw=2) ##Vert line
#     https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axes.html
#     https://matplotlib.org/gallery/subplots_axes_and_figures/axes_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axes-demo-py
    
    
    ## ROC Curve Plot
#     a = plt.axes([.09, .149, .27, .21], facecolor='lightgrey')
    ax2 = fig.add_subplot(gs[2,0])
    fpr, tpr, thresholds = roc_curve(y_act, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
#     plt.xlim(0,1)
#     plt.ylim(0,1)
    plt.title('roc_curve')
    plt.axis([0, 1, 0, 1])
    plt.axhline(linewidth=3, color="k")
    plt.axvline(linewidth=3, color="k")
    
    ## Summary Stats
    ax2 = fig.add_subplot(gs[2,1], frame_on=False) #, frame_on=True
    plotDF = pd.DataFrame(classReport2dict(classification_report(y_act, y_pred))).T.reset_index()
    ax2.xaxis.set_visible(False) # hide the x axis
    ax2.yaxis.set_visible(False)
    table(ax2, plotDF, loc='center')
    plt.show()

# plotConfusionMatrix(y_ValDF, TB_Predict)


# -------------------------------------------------------------------------------------------------------------------

## Plot labels on Y axis and continous scale on X axis with hue color
def plotFeatureAndProperty(features_li, prop, hue=None, tit=None):
    '''
    Investigate patterns in the amount of missing data in each column.
    '''
    if hue is None:
        HueColor = pd.Series(features_li)
    else:
        if type(hue) is list:
            HueColor = pd.Series(hue)
        elif type(hue) is pd.core.series.Series:
            HueColor = hue
        # raise error
    
    yAxis = features_li
    xAxis = prop
    #HueColor = feat_info.loc[ [ ele in yAxis for ele in feat_info['attribute'] ], 'information_level']
    
    fig = plt.figure(figsize=(12, int(len(yAxis)*0.30)))
    sns.set(style="whitegrid")
    # sns.set(style="darkgrid")
    color = ['b', 'y', 'm', 'r', 'g', 'c', 'aqua', 'sienna', 'lime', 'steelblue', 'hotpink', 'gold',
             'yellow1', 'wheat1', 'violetred1', 'turquoise1', 'tomato1', 'banana', 'bisque4',
             'thistle1', 'tan1', 'steelblue1', 'springgreen1', 'snow3', 'slategray2', 'slateblue2',
             'skyblue2', 'sienna1', 'sgilightblue', 'sgilightgray', 'sgiolivedrab', 'sgisalmon',
             'sgislateblue', 'sgiteal', 'sgigray32', 'sgibeet', 'seagreen2', 'salmon2', 'royalblue2',
             'rosybrown2', 'red1', 'raspberry', 'purple2', 'plum1', 'peachpuff1', 'palevioletred1',
             'paleturquoise2', 'palegreen1', 'orchid1', 'orangered1', 'orange1', 'olivedrab1', 'olive',
             'navajowhite1', 'mediumvioletred', 'mediumpurple1', 'maroon2', 'limegreen', 'lightsalmon4',
             'lightpink1', 'lightcoral', 'indianred1', 'green1', 'gold2', 'firebrick1', 'dodgerblue2',
             'deeppink1', 'deepskyblue1', 'darkseagreen1', 'darkorange1', 'darkolivegreen1', 'darkgreen',
             'darkgoldenrod2', 'crimson', 'chartreuse2', 'cadmiumorange', 'burntumber', 'brown2', 'blue2',
             'antiquewhite4', 'aquamarine4', 'k']
    HueColor = HueColor.reset_index(drop=True)
    CatList = np.sort(HueColor.unique()).tolist()
    for cat in HueColor.unique():
        # print(cat)
        ## hue color should be sorted
        Colo = color[CatList.index(cat)] 
        MinInd = min(HueColor[HueColor == cat].index)
        MaxInd = max(HueColor[HueColor == cat].index)
        plt.barh(yAxis[MinInd:MaxInd+1], width=xAxis[MinInd:MaxInd+1], align='center', label=cat) #, color=Colo -- manual color assignment disabled

    plt.grid(True, color='black', alpha=0.2)
    if tit is not None: plt.title('Features VS {} - In Features'.format(tit), fontsize=15)
    # plt.xlabel('% Missing Observation')
    plt.ylabel('Features')
    plt.gca().invert_yaxis()
    plt.legend(loc='lower right', frameon=True)
    plt.show()
# tempDF = pd.concat([df_info['Caterorical'], df_info['Numerical']], sort=False)
# featName = list(tempDF.index)
# missingPct = list(tempDF['%Missing'])
# plotFeatureAndProperty(featName, missingPct, featName, tit='%Missing')


def plotAccAndErrorWrtThreshold(yActual, yPredictedStocastic):
    ''' . '''
    ResultDF, i, j, step, ClassThres = pd.DataFrame(), 0.0, 1.0, 0.01, []
    while i <= j: ClassThres.append(round(i,3)); i += step

    i = 0
    for limit in ClassThres:
        predicted = np.copy(yPredictedStocastic)
        ResultDF.loc[i, 'ThresholdValue'] = limit
        predicted[predicted > limit] = 1
        predicted[predicted <= limit] = 0
        ResultDF.loc[i, 'Accuracy'] = accuracy_score(yActual, predicted)
        ResultDF.loc[i, 'ErrorRate'] = 1 - ResultDF.loc[i, 'Accuracy']
        ResultDF.loc[i,'MCC'] = matthews_corrcoef(yActual, predicted)
        i += 1
    ser_thres, ser_acc, ser_err = ResultDF['ThresholdValue'], ResultDF['Accuracy'], ResultDF['ErrorRate']
    
    fig = plt.figure(figsize=(15,6))
    ax = fig.gca()
    
    plt.subplot(121)
    ax.set_xticks(np.arange(start = 0, stop = 1.01, step = 0.1))
    ax.set_yticks(np.arange(start = 0, stop = 100.01, step = 10))
    plt.plot(ser_thres, 100*ser_acc, label = 'Accuracy', lw=3)
    plt.plot(ser_thres, 100*ser_err, label = 'Error Rate', lw=3)
    plt.title('Evaluation Parameters VS Threshold', fontsize=15)
    plt.xlabel('Threshold Value', fontsize=13)
    plt.ylabel('Percentage', fontsize=13)
    plt.legend(fontsize = 13)
    #plt.xticks(np.arange(start = 0, stop = 1.01, step = 0.1))
    #plt.yticks(np.arange(start = 0, stop = 100.01, step = 10))
    plt.axis([0,1,0,100])
    plt.axhline(0, color='black',lw=3)
    plt.axvline(0, color='black',lw=3)
    plt.margins(1,1)
    plt.grid(True, color = 'black', alpha = 0.3)
    #plt.label()
    #plt.rc
    
    maxMCC = max(ResultDF['MCC'])
    threshMaxMCC = list(ResultDF.loc[ResultDF['MCC']==maxMCC, 'ThresholdValue'])[0]
    
    plt.subplot(122)
    ax.set_xticks(np.arange(start = 0, stop = 1.01, step = 0.1))
    ax.set_yticks(np.arange(start = 0, stop = 100.01, step = 10))
    plt.plot(ser_thres, ResultDF['MCC'], label = 'MCC', lw=3)
    plt.title('Matthew Correlation Coefficient VS Threshold', fontsize=15)
    plt.xlabel('Threshold Value', fontsize=13)
#     plt.ylabel('Percentage', fontsize=13)
    plt.legend(fontsize = 13)
    #plt.xticks(np.arange(start = 0, stop = 1.01, step = 0.1))
    #plt.yticks(np.arange(start = 0, stop = 100.01, step = 10))
    plt.axis([0,1,0,1])
    plt.axhline(maxMCC, color='black',lw=1.5)
    plt.axvline(threshMaxMCC, color='black',lw=1.5)
    plt.margins(1,1)
    plt.grid(True, color = 'black', alpha = 0.3)
    
    plt.show()
    return threshMaxMCC
# threshold = plotAccAndErrorWrtThreshold(ytest, ypred) 


# --------------------------------------------------<<Plotting Features>>-------------------------------------- #

## General function for plots
def generalInitialSettingForGraph():
    ''' . '''
    sns.set_style('whitegrid') #whitegrid, darkgrid, white, dark
    return plt.subplots(figsize=(13, 4), dpi=80, facecolor='w', edgecolor='k')

def univarateTitle(featureType, tit, ser):
    ''' . '''
    fMsg = '"'+tit+'"' if tit is not None else '"'+ser.name+'"' if type(ser) == pd.core.series.Series else ''
    return 'Plotting distribution for the {} feature {}'.format(featureType, fMsg)

def bivarateTitle(featTit, ser_x, ser_y):
    ''' . '''
    if (featTit[0] is not None) & (featTit[1] is not None):
        return 'graph b/w {} and {}'.format(featTit[0], featTit[1])
    elif (type(ser_x) == pd.core.series.Series) & (type(ser_y) == pd.core.series.Series):
        return 'graph b/w {} and {}'.format(ser_x.name, ser_y.name)

def boundaryAxis(ax, xmin, xmax, ymin, ymax):
    ''' . '''
    ax.axhline(y=ymin, lw=3, color='k')
    ax.axhline(y=ymax, lw=3, color='k')
    ax.axvline(x=xmin, lw=3, color='k')
    ax.axvline(x=xmax, lw=3, color='k')

def genralSettingForGraph(ax):
    ''' . '''
    plt.grid(True, axis='both', color='k', alpha=0.5, lw=0.5)
    xmin, xmax, ymin, ymax = plt.axis()
    boundaryAxis(ax, xmin, xmax, ymin, ymax)
    plt.legend(loc='upper right', frameon=True)
    plt.show()
    # plt.title('Plotting distribution for the feature.', fontsize=15)
    # plt.xlabel(''); plt.ylabel(''); plt.gca().invert_yaxis()

## Univariate
def plotUnivariteDist_Numerical(ser, tit=None):
    '''
    to plot box plot and distribution plot for a variable
    '''
    sns.set_style("whitegrid") #whitegrid, darkgrid, dark
    # Cut the window in 2 parts
    f, (ax_box, ax_hist) = plt.subplots(nrows=2, figsize=(14, 5), sharex=True, \
                                        gridspec_kw={"height_ratios": (.15, .85)},\
                                        dpi=80, facecolor='w', edgecolor='k')
    # Add a graph in each part
    box = sns.boxplot(ser, ax=ax_box, color = 'darkorange')
    dist = sns.distplot(ser, ax=ax_hist, color = 'darkorange',# rug=True, rug_kws={"color": "g"},
                      kde_kws={'color': 'k', 'lw': 2, 'label': 'KernelDensityEstimate'},
                      hist_kws={'linewidth': 1, 'alpha': 0.75, 'color': 'darkorange'} ) # 'histtype': 'step'
    # dist.set(xlim=(0, 400))
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    box.set_title(univarateTitle('numerical', tit, ser), fontdict={'fontsize': 14, 'fontweight' : 2} )
    genralSettingForGraph(ax_hist)

def plotUnivariteDist_Categorical(ser, tit=None):
    '''
    numberic feature can also be plotted
    to plot box plot and distribution plot for a variable
    '''
    ser = pd.Series(ser)
    ind = ser.value_counts()
    plot_df = pd.DataFrame(ind).reset_index().rename(columns={'index':ind.name, ind.name:'count'})
    x, y = plot_df['C'], plot_df['count']
    f, ax = generalInitialSettingForGraph()
    cnt = sns.countplot(x=ser, color = 'darkorange', saturation=0.75)
    pt = sns.pointplot(x=x, y=y, markers='x', linestyles='-', dodge=False, join=True, scale=1.25, \
                       orient='v', color='k')
    cnt.set_title(univarateTitle('Categorical', tit, ser), fontdict={'fontsize': 14, 'fontweight' : 2} )
    genralSettingForGraph(ax)

## Bivariate
def plotBivariate_NumNum(ser_x, ser_y, hue=None, style=None, sizes=None, col=None, row=None, featTit=(None, None)):
    ''' . '''
    f, ax = generalInitialSettingForGraph()
    ## below two statement are generating error --<resolve this> <todo>
    # sns.relplot(x='A', y='B', data=df) -->Works
    # sns.relplot(x=df['A'], y=df['B'], ) -->Doesn't Works
#     lin = sns.relplot(x=ser_x, y=ser_y, hue= hue, style=style, sizes=sizes, col=col,
#                       row=row, kind='line', ci='sd')# , height=3
#     sct = sns.relplot(x=ser_x, y=ser_y, hue= hue, style=style, sizes=sizes) #, palette="ch:r=-.5,l=.75", sizes=(15, 200)
    sct = sns.scatterplot(x=ser_x, y=ser_y, hue= hue, style=style, sizes=sizes)
    sct.set_title(bivarateTitle(featTit, ser_x, ser_y), fontdict={'fontsize': 14, 'fontweight' : 2})
    genralSettingForGraph(ax)

def plotBivariate_NumCat(ser_x, ser_y, hue=None, featTit=(None, None)):
    ''' . '''
    f, ax = generalInitialSettingForGraph()
    ## below two statement are generating error --<resolve this> <todo>
    # cat = sns.catplot(x='A', y='B', data=df,hue=hue) -->Works
    # cat = sns.catplot(x=df['A'], y=df['B'],hue=hue) -->Doesn't Works
    plt.scatter(ser_x, ser_y, c=ser_y)
    plt.title(bivarateTitle(featTit, ser_x, ser_y), fontdict={'fontsize': 14, 'fontweight' : 2})
    genralSettingForGraph(ax)

def plotBivariate_CatCat(ser_x, ser_y, featTit=(None, None)):
    ''' . '''
    #pd.crosstab()
    ser_x, ser_y = pd.Series(ser_x), pd.Series(ser_y)
    tempDF = pd.concat([ser_x, ser_y], axis=1)
    tempDF = pd.DataFrame(tempDF.groupby(by=[ser_x.name, ser_y.name]).size()).reset_index()
    ser_x, ser_y, size = tempDF[ser_x.name], tempDF[ser_y.name], (tempDF[0]/tempDF[0].sum())*100
    
    f, ax = generalInitialSettingForGraph()
    plt.scatter(ser_x, ser_y, s=size) #sns.jointplot
    plt.title(bivarateTitle(featTit, ser_x, ser_y), fontdict={'fontsize': 14, 'fontweight' : 2})
    genralSettingForGraph(ax)

# plotBivariate_CatCat(df['C'], df['C'])

## univariate and bivariate analysis -- automatic datatype detection
def visualizeFeature(ser_x, ser_y):
    '''
    plotting Univariate and bivariate  plots
    '''
    print('Data type of series1 and series2: {} and {}'.format(ser_x.dtype, ser_y.dtype))
    # ---------------------------------< Univariate Analysis >--------------------------------- #
    print('-'*100+ '\nUniVariate Plot\n'+'-'*100)
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    if (ser_x.dtypes == 'int') | (ser_x.dtypes == 'float'):
        plotUnivariteDist_Numerical(ser_x, tit=None)
    elif ser_x.dtypes == 'O':
        plotUnivariteDist_Categorical(ser_x, tit=None)
    elif ser_x.dtypes == 'datetime64[ns]':
        '''Nothing as of Now'''
        ser = pd.Series(ser_x).sort_values().astype('str')
        plotUnivariteDist_Categorical(ser, tit=None)
    print('-'*100)
    # ---------------------------------< Bi-Variate Analysis >--------------------------------- #
    print('-'*100+ '\nBiVariate Plot\n'+'-'*100)
    if (((ser_x.dtypes == 'int') | (ser_x.dtypes == 'float')) & 
        ((ser_y.dtypes == 'int') | (ser_y.dtypes == 'float'))):
        plotBivariate_NumNum(ser_x, ser_y)
    elif (((ser_x.dtypes == 'int') | (ser_x.dtypes == 'float')) &
          (ser_y.dtypes == 'O')):
        plotBivariate_NumCat(ser_x, ser_y)
    elif ((ser_x.dtypes == 'O') & (ser_y.dtypes == 'O')):
        plotBivariate_CatCat(ser_x, ser_y)
    print('-'*100)
    #f.savefig(FileSavingLoc_dir + 'VariablePlot__{}__{}.png'.format(Var, time.time()), bbox_inches="tight")
    
    
# ---------------------------------------------------------------------------------------------------------------#

