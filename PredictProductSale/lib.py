import pandas as pd
from sklearn.metrics import roc_curve

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
            colNotPresent = len([ False for ele in a.keys() if ele not in df.columns ])
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
def Plot_Confusion_Matrix(y_act, y_pred):
    
        print(classification_report(y_ValDF, TB_Predict))
        
        tn, fp, fn, tp = confusion_matrix(y_act, y_pred).ravel()
        
        PCP = tp + fp # Predicted Condition Positive
        PCN = fn + tn # Predicted Condition Negative
        TCP = tp + fn # True Condition Positive
        TCN = fp + tn # True Condition Negative
        
        TotPop = tp + fp + fn + tn # Total Population
        
        accuracy = (tp + tn) / TotPop
        Prevalence = (tp + fn) / TotPop  # TCP/TotPop
        Precision = tp / PCP #tp/(fp + tp)
        FalseDiscoveryRate = fp / PCP #fp/(fp + tp)
        FalseOmissionRate = fn / PCN #fn/(tn + fn)
        NegativePredictiveValue = tn / PCN #tn/(tn + fn)
        
        TruePositiveRate = tp / TCP
        FalsePositiveRate = fp / TCN
        PositiveLikelihoodRatio = TruePositiveRate / FalsePositiveRate
        FalseNegativeRate = fn / TCP
        TrueNegativeRate = tn / TCN
        NegativeLikelihoodRatio = FalseNegativeRatio / TrueNegativeRate
        
        DiagnosticOddsRatio = PositiveLikelihoodRatio / NegativeLikelihoodRatio
        F1Score = (2*tp) / (2*tp + fp + fn)
        
        MattheawCorrelationCoefficient = (tp*tn - fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)

        fig = plt.figure(figsize=(15,8))
        ax  = fig.add_subplot(111)
        # ax.imshow(df, interpolation='nearest', cmap=plt.cm.gray)

        # Draw the grid boxes
        ax.set_xlim(-0.5,2.5)
        ax.set_ylim(2.5,-0.5)

        #ax.plot([x1,x2],[y1,y2], '-k', lw=2)
        ## HorizontalLines
        ax.plot([ 0.5,1.5],[-0.5,-0.5], '-k', lw=4) ## border
        ax.plot([ 0.0,2.5],[0.0,0.0], '-k', lw=2)
        ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
        ax.plot([ 0.0,2.5],[1.0,1.0], '-k', lw=2)
        ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
        ax.plot([ 0.5,2.0],[2.0,2.0], '-k', lw=2)
        ax.plot([ 0.5,2.5],[2.5,2.5], '-k', lw=4) ## border
        ## Vertical Line
        ax.plot([-0.5,-0.5], [0.5,1.5], '-k', lw=4) ## border
        ax.plot([0.0,0.0], [ 0.0,1.5], '-k', lw=2)
        ax.plot([0.5,0.5], [-0.5,2.5], '-k', lw=2)
        ax.plot([1.0,1.0], [ 0.0,2.5], '-k', lw=2)
        ax.plot([1.5,1.5], [-0.5,2.5], '-k', lw=2)
        ax.plot([2.0,2.0], [ 0.0,2.5], '-k', lw=2)
        ax.plot([2.5,2.5], [ 0.0,2.5], '-k', lw=4) ## border
        
        ### Creating a box
        ax.plot([0.5,1.5], [0.5,0.5], '-k', lw=4) ##Horiz line
        ax.plot([0.5,1.5], [1.5,1.5], '-k', lw=4) ##Horiz line
        ax.plot([0.5,0.5], [0.5,1.5], '-k', lw=4) ##Vert line
        ax.plot([1.5,1.5], [0.5,1.5], '-k', lw=4) ##Vert line
        
        ax.set_facecolor("w")
        
        
        ## Setting Headings
        ax.text(1.0, -0.25, s = 'True Condition', fontsize=18, color = 'w',  va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
        ax.text(-0.25, 1.0, s = 'Predicted\nCondition', fontsize=18, color = 'w', va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
        ax.text(0.25, 0.75, s = 'Condition\nPositive', fontsize=18, color = 'k', va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
        ax.text(0.25, 1.25, s = 'Condition\nNegative', fontsize=18, color = 'k', va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
        ax.text(0.75, 0.25, s = 'Condition\nPositive', fontsize=18, color = 'k', va='center', ha='center', bbox=dict(fc='k', alpha=0, boxstyle='round,pad=1'))
        ax.text(1.25, 0.25, s = 'Condition\nNegative', fontsize=18, color = 'k', va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
        ## Values box
        ax.text(0.75,0.75, 'True Pos: {}'.format(round(tp,0)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
        ax.text(1.25,0.75, 'False Pos: {}\nType I error'.format(round(fp,0)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
        ax.text(0.75,1.25, 'False Neg: {}\nType II error'.format(round(fn,0)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))
        ax.text(1.25,1.25, 'True Neg: {}'.format(round(tn,0)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0,boxstyle='round,pad=1'))

        ax.text(1.75,0.25, 'Prevalence: {}'.format(round(Prevalence,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(2.25,0.25, 'Accuracy: {}'.format(round(accuracy,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))

        ax.text(1.75,0.75, 'Pos Pred Val,\n Precision: {}'.format(round(Precision,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(2.25,0.75, 'False Discovery Rate,\n FDR: {}'.format(round(FalseDiscoveryRate,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(1.75,1.25, 'False Omission Rate,\n FOR: {}'.format(round(FalseOmissionRate,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(2.25,1.25, 'Neg Pred Value,\n NDV: {}'.format(round(NegativePredictiveValue,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))

        ax.text(0.75,1.75, 'True Pos Rate,\n TPR: {}'.format(round(TruePositiveRate,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(0.75,2.25, 'Falsse Pos Rate,\n FPR: {}'.format(round(FalsePositiveRate,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(1.25,1.75, 'False Neg Rate,\n FNR: {}'.format(round(FalseNegativeRate,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(1.25,2.25, 'True Neg Rate,\n TNR: {}'.format(round(TrueNegativeRate,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))

        ax.text(1.75,1.75, 'Pos Likelihood Ratio,\n LR+: {}'.format(round(PositiveLikelihoodRatio,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(1.75,2.25, 'Neg Likelihood Ratio,\n LR-: {}'.format(round(NegativeLikelihoodRatio,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(2.25,1.75, 'Diag Odds Rat,\n DOR: {}'.format(round(DiagnosticOddsRatio,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        ax.text(2.25,2.25, 'F1 Score: {}'.format(round(F1Score,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        # ax.text(1.75,0.75, 'PPV,\n Precision: {}'.format(round(xxx,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))

        ax.text(0.25,0.25, 'Matthews Corr Coeff,\n MCC: {}'.format(round(MattheawCorrelationCoefficient,3)), fontsize=13, va='center', ha='center', bbox=dict(fc='w', alpha=0, boxstyle='round,pad=1'))
        # ax.bbox([[0.08, 0.125], [0.95, 0.88]], facecolor='0.2', alpha=0.5)
        ax.add_patch( patches.Rectangle( (0.5, -0.5), 1.0, 0.5, facecolor='k', alpha=1.0) )
        ax.add_patch( patches.Rectangle( (-0.5, 0.5), 0.5, 1.0, facecolor='k', alpha=1.0) )
        ax.add_patch( patches.Rectangle( (0.5, 0.0), 0.5, 0.5, facecolor='b', alpha=0.7) )
        ax.add_patch( patches.Rectangle( (1.0, 0.0), 0.5, 0.5, facecolor='orange', alpha=0.7) )
        ax.add_patch( patches.Rectangle( (0.0, 0.5), 0.5, 0.5, facecolor='b', alpha=0.7) )
        ax.add_patch( patches.Rectangle( (0.0, 1.0), 0.5, 0.5, facecolor='orange', alpha=0.7) )
        # ax.add_patch( patches.Rectangle( (0.5, -0.5), 0.5, 1.0, facecolor='0.2', alpha=0.5) )
        # ax.axhspan(1, 2.25)
        ax.axis('off')


        ### Creating a box for other measure
        ax.plot([-0.45, 0.45], [1.6,1.6], '-k', lw=2) ##Horiz line
        ax.plot([-0.45, 0.45], [2.4,2.4], '-k', lw=2) ##Horiz line
        ax.plot([-0.45,-0.45], [1.6,2.4], '-k', lw=2) ##Vert line
        ax.plot([ 0.45, 0.45], [1.6,2.4], '-k', lw=2) ##Vert line
    #     https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axes.html
    #     https://matplotlib.org/gallery/subplots_axes_and_figures/axes_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axes-demo-py

        a = plt.axes([.09, .149, .27, .21], facecolor='lightgrey')
        fpr, tpr, thresholds = roc_curve(y_TestDF, y_pred)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
    #     plt.xlim(0,1)
    #     plt.ylim(0,1)
        plt.axis([0, 1, 0, 1])
        plt.axhline(linewidth=3, color="k")
        plt.axvline(linewidth=3, color="k")

# Plot_Confusion_Matrix(y_ValDF, TB_Predict)


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





