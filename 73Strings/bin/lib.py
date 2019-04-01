import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

import ast, time
import math
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, Birch
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, LatentDirichletAllocation, FastICA, TruncatedSVD
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
import scipy.cluster.hierarchy as sch

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




# -------------------------------------------------<< Unsupervised Learning >>------------------------------------------ #

def DimensionTransf(AlgoToUse, AlgoConfig, DF):
    df = DF.copy()
    DimensionTransformModels_dict = {
        'PCA': {'Model': PCA(), 'DataTypeBoundation': 'Nil', 'fit': True, 'fit_transform': True, 'transform': True }, 
        'IncPCA': {'Model': IncrementalPCA(), 'DataTypeBoundation': 'Nil', 'fit': True, 'fit_transform': True, 'transform': True }, 
        'KerPCA': {'Model': KernelPCA(), 'DataTypeBoundation': 'Nil', 'fit': True, 'fit_transform': True, 'transform': True }, 
        'LDA': {'Model': LatentDirichletAllocation(), 'DataTypeBoundation': 'Normalized', 'fit': True, 'fit_transform': True, 'transform': True }, 
        'ICA': {'Model': FastICA(), 'DataTypeBoundation': 'Normalized', 'fit': True, 'fit_transform': True, 'transform': True },  
        'TrunSVD': {'Model': TruncatedSVD(), 'DataTypeBoundation': 'Nil', 'fit': True, 'fit_transform': True, 'transform': True }, 

        'MiniBatchSparsePCA': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html#sklearn.decomposition.MiniBatchSparsePCA
        'SparsePCA': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA
        'DictionaryLearning': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html#sklearn.decomposition.DictionaryLearning
        'MiniBatchDictionaryLearning': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html#sklearn.decomposition.MiniBatchDictionaryLearning
        'FactorAnalysis': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis
        'NMF': {} # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF
        }

    params =  ast.literal_eval(AlgoConfig) ##
    Model = DimensionTransformModels_dict[AlgoToUse]['Model']
    Model.set_params(**params)
    ModSpecDataPrep = DimensionTransformModels_dict[AlgoToUse]['DataTypeBoundation']
    print('Transforming Dimensions Using :', AlgoToUse)
    ## Training Model
    if(df is not None):
        print('Developing Model :: On provided Data')
        if(DimensionTransformModels_dict[AlgoToUse]['fit_transform'] == True):
            tempDF = pd.DataFrame(Model.fit_transform(df.loc[:, :]))
            tempDF.rename(columns=dict(zip(tempDF.columns, AlgoToUse + '_var_' + tempDF.columns.astype('str'))), inplace=True)
            # df_transf = df_transf.join(tempDF, rsuffix='_y')
            # Trainset_transformed.columns = Trainset_transformed.columns.astype(str)  # Column name being numeric
        elif((DimensionTransformModels_dict[AlgoToUse]['fit'] == True) & (DimensionTransformModels_dict[AlgoToUse]['transform'] == True)):
            Model.fit(df.loc[:, :])
            tempDF = pd.DataFrame(Model.transform(df.loc[:, :]))
            tempDF.rename(columns=dict(zip(tempDF.columns, AlgoToUse + '_var_' + tempDF.columns.astype('str'))), inplace=True)
            # df_transf = df_transf.join(tempDF, rsuffix='_y')
        else:
            print('Some Error is present')
    return tempDF, Model



def ClusterDevelopment(ClustAlgo, ClustAlgo_ParamConfig, DF):
    df = DF.copy()
    ClustAlgo_params = ast.literal_eval(ClustAlgo_ParamConfig)
    
    ### Defining Models and their property
    ClusteringModels_dict = {
        'KMeans': {'ModelType': 'ClusterModelData', 'Model': KMeans(), 'DataTypeBoundation': 'Nil', 
                            'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False}, 
        'MiniBatchKMeans': {'ModelType': 'ClusterModelData', 'Model': MiniBatchKMeans(), 'DataTypeBoundation': 'Nil', 
                             'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False}, 
        'AffinityPropagation': {'ModelType': 'ClusterModelData', 'Model': AffinityPropagation(), 'DataTypeBoundation': 'Nil', 
                               'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False}, 
        'MeanShift': {'ModelType': 'ClusterModelData', 'Model': MeanShift(), 'DataTypeBoundation': 'Nil', 
                        'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False},
        'Birch': {'ModelType': 'ClusterModelData', 'Model': Birch(), 'DataTypeBoundation': 'Nil', 
                            'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False}, 
        'SpectralClustering': {'ModelType': 'ClusterModelData', 'Model': SpectralClustering(), 'DataTypeBoundation': 'Nil', 
                             'fit': True, 'fit_predict': True, 'predict': False, 'DecisionFunction': False}, 
        'AgglomerativeClustering': {'ModelType': 'ClusterModelData', 'Model': AgglomerativeClustering(), 'DataTypeBoundation': 'Nil', 
                               'fit': True, 'fit_predict': True, 'predict': False, 'DecisionFunction': False}, 
        'DBSCAN': {'ModelType': 'ClusterModelData', 'Model': DBSCAN(), 'DataTypeBoundation': 'Nil', 
                        'fit': True, 'fit_predict': True, 'predict': False, 'DecisionFunction': False},
    }

    Model = ClusteringModels_dict[ClustAlgo]['Model']
    Model.set_params(**ClustAlgo_params)
    ModelSpecificDataPreparation = ClusteringModels_dict[ClustAlgo]['DataTypeBoundation']
    ModelType = ClusteringModels_dict[ClustAlgo]['ModelType']


    ## Training Model
    if(df is not None):
        print('Developing Model :: On provided Data')
        if(ClusteringModels_dict[ClustAlgo]['fit_predict'] == True):
            df[ClustAlgo + '_Predict'] = pd.DataFrame( Model.fit_predict(df) )  
        elif((ClusteringModels_dict[ClustAlgo]['fit'] == True) & (ClusteringModels_dict[ClustAlgo]['predict'] == True)):
            Model.fit(df) 
            df[ClustAlgo + '_Predict'] = pd.DataFrame(Model.predict(df)) 
        else:
            print('Some Error is present')
    return df


def CustomEntropy(labels_true, labels_pred, roundOffTo = 5):
    '''
    formula provided in unknown unknown paper is used
    log to the base 2 is used 
    labels_true need to be in binary format for this i.e. 0 = human and 1 = bot
    '''
    lab_true = [ int(i) for i in labels_true ]
    lab_pred = labels_pred#.copy() #[ int(i) for i in labels_pred ] 

    partitions = pd.Series(lab_pred).unique()  ##by algorithms

    Total_CriticalClass = sum(lab_true)

    entropy = 0
    for p in partitions:
        CriticalClassInThisPartition = sum([ lab_true[ind] for ind in range(len(lab_pred)) if lab_pred[ind] == p ])
        temp = CriticalClassInThisPartition/Total_CriticalClass
        #print('printing temp from Entropy:', temp)
        if(temp != 0):
            entropy -= temp * math.log2(temp)  ## lim x-->0 x*logx = 0
    
    return round(entropy, roundOffTo)

def ComputingClusterEvalMetric(X, labels_true, labels_pred):
    
    RoundOffTo = 5
    
    ## Calculating Adjusted Rand index  # consensus measure
    try:
        ES = CustomEntropy(labels_true, labels_pred, RoundOffTo)
    except Exception as e: 
        print('CustomEntropy Error: ', e)
        ES = None
    
    ## Calculating Adjusted Rand index  # consensus measure
    try:
        ARI =  round(metrics.adjusted_rand_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Adjusted Rand index Error: ', e)
        ARI = None

    ## Calculating Adjusted Mutual Information Based Scores  # consensus measure
    try:
        AMIS = round(metrics.adjusted_mutual_info_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Adjusted Mutual Information Based Scores Error: ', e)
        AMIS = None
    try:
        NMIS = round(metrics.normalized_mutual_info_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Normalized Mutual Information Based Scores Error: ', e)
        NMIS = None

    ## Calculating Homogenity, Completeness and V-measure
    try:
        HS = round(metrics.homogeneity_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Homogenity Error: ', e)
        HS = None
    try:
        CS = round(metrics.completeness_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Completeness Error: ', e)
        CS = None
    try:
        VMS = round(metrics.v_measure_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('V-Measure Error: ', e)
        VMS = None
    #HS_CS_VMS = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)

    ## Calculating Fowlkes-Mallows Scores
    try:
        FMS = round(metrics.fowlkes_mallows_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Fowlkes-Mallows Scores Error: ', e)
        FMS = None

    if(X is not None):
        ## Calculating Silhouette Coefficient
        try:
            SCS = round(metrics.silhouette_score(X, labels_pred, metric='euclidean', sample_size= 25000), RoundOffTo)
            #print("printing Temp Silhouette Coefficient: ", SCS)
            #print(type(SCS))
        except Exception as e: 
            print('Silhouette Coefficient Error: ', e)
            SCS = None

        ## Calculating Calinski-Harabaz Index 
        try:
            CHI = round(metrics.calinski_harabaz_score(X, labels_pred), RoundOffTo)
        except Exception as e: 
            print('Calinski-Harabaz Index Error: ', e)  ## there is no error is Anomaly algorithm was there
            CHI = None
    else:
        SCS = None
        CHI = None

    ClusterEvaluationScore = {
        'Timestamp': time.strftime('%y/%m/%d %Hhr:%Mmin(%Z)', time.gmtime()), 
        'Algorithm': '---', 
        'NoOfCluster': len(pd.Series(labels_pred).unique()), 
         ## Below Metric Do require True Label
        'Cust_EntropyScore': ES,
        'AdjustedRandIndex': ARI, 
        'AdjustedMutualInfoScore': AMIS, 
        'NormalizedMutualInfoScore': NMIS, 
        'HomogenityScore': HS, 
        'CompletenessScore': CS, 
        'V-measureScore': VMS, 
        'FowlkesMallowsScore': FMS, 
        ## Below Metric Doesn't Require True Label
        'SilhouetteCoefficient': SCS, 
        'CalinskiHarabazScore': CHI, 
        }
    return ClusterEvaluationScore


    
# ---------------------------------------------------------------------------------------------------------------#

