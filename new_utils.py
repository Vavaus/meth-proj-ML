import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
import pickle
import warnings
import random
import re

from scipy.stats import linregress, pearsonr, kendalltau, spearmanr, describe

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.ar_model import AutoReg

from typing import Any

from tqdm import tqdm

warnings.filterwarnings('ignore')

class PreProcess:
    def __init__(self, dataframe, impute:bool = False, imp_method:str = 'mean',
                 organism:str = 'mouse'):
        """
            Class for data preparation:

            Clean from NaN or impute -> 
            Select most variable by age tissue ->
            Convert to float32 ->
            Convert age to column and sites as index ->
            Clear data from constant sites to avoid NaNs for correlation.

            ****
            Args: dataframe - pandas df with row beta-values
                  impute - whether to impute NaNs or drop them
                  imp_method - imputation method: zeros or means
            
            ******
            Output: cleaned pandas dataframe

            *******
            Example: cleaned_data = PreProcess(dataframe, False)()
        """
        self.df = dataframe
        self.impute = impute
        self.imp_method = imp_method
        self.organ = organism

    def clear_df(self, threshold: float | Any = None):
        print('*'*30)
        print('Start to clear...')
        if self.impute:
            if self.imp_method == 'mean':
                self.df.fillna(value = self.df.values.mean(), inplace = True)
            elif self.imp_method == 'zeros':
                self.df.fillna(value = 0, inplace = True)
            else:
                raise ValueError('This method is not supported.')
        else:
            if threshold is not None:
                self.df.dropna(axis = 1, inplace = True, thresh = int((1-threshold)*self.df.shape[0]))
            else:
                self.df.dropna(axis = 1, inplace = True)
        print('Finished!')
        print('*'*30)
    
    def select_best_tissue(self, column:str = 'Cell', plot_distrib:bool = True):

        def unique(row):
            nans = row.isna()
            row = row[~nans]
            return np.unique(row).shape[0] > 1
        if self.organ == 'mouse':
            column = 'Tissue'
            droplist = [column, 'Sex', 'ID', 'Strain']
        else:
            column = 'Cell'
            droplist = [column, 'Sex']
        print(f'Looking for {column} with max values...')
        counts = Counter(self.df[column])
        if plot_distrib:
            plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
            plt.title(f'% of age variability by tissue.')
            plt.show()
        keys, vals = list(counts.keys()), list(counts.values())
        best = keys[np.argmax(vals)]

        print(f'Choosing {column} == {best}')
        selected = self.df[self.df[column] == best]

        print('Dropping all unneccessary columns...')
        selected.drop(droplist, inplace = True, axis = 1)

        print('Sorting age and setting it as index...')
        selected['Age'] = selected['Age'].astype('category')
        selected = selected.sort_values(by = 'Age')
        selected.set_index('Age', inplace=True)

        print('Transposing...')
        selected = selected.T

        print('Clearing absolute constant values...')
        selected['unique'] = selected.nunique(axis=1, dropna=True) > 1
        selected = selected[selected.unique == 1]
        selected.drop('unique', axis = 1, inplace = True)
        print('Finished!')
        print('*'*30)

        return selected
    
    def __call__(self, column:str = 'Tissue', plot_distrib:bool = True, threshold: float | Any = None):
        self.clear_df(threshold=threshold)
        res = self.select_best_tissue(column=column, plot_distrib=plot_distrib)
        return res
    
class StatisticCounter:
    def __init__(self, processed_data, typ = pearsonr):
        """
            Class for statistics computation:

            Calculate correlation and linear regression ->
            Calculate correlation of residuals ->
            Calculate sign of correlation ->
            Drop all unneccesary columns.
            
            ****
            Args: processed_data - dataframe after cleaning,
                  typ - type of correlation (basic is one of pearson, spearman, kendall)
            
            ******
            Output: Proccesed dataframe with beta-values and target features

            *******
            Example: 
            data_with_corrs = StatisticCounter(cleaned_data, pearsonr)(plot_distribution=False, show_summary = False)
        """
        self.df = processed_data
        self.x = self.df.columns.get_level_values('Age').astype(np.float32)
        self.typ = typ

    def calculation(self):
        print('Calculating statistics...')
        pval = np.zeros(self.df.shape[0], dtype=np.float32)
        means = np.zeros(self.df.shape[0], dtype=np.float32)
        sd = np.zeros(self.df.shape[0], dtype=np.float32)
        lr = []
        R = np.zeros(self.df.shape[0], dtype=np.float32)
        spear = np.zeros(self.df.shape[0], dtype=np.float32)
        spear_p = np.zeros(self.df.shape[0], dtype=np.float32)

        for index in range(self.df.shape[0]):
            row = self.df.iloc[index].astype(np.float32)
            row = row[:self.x.shape[0]]
            nans = row.isna()
            if nans.all():
                continue
            row_values = row[~nans].to_numpy()
            x_values = self.x[~nans].to_numpy()
            w_temp, b_temp, _, _, _ = linregress(x_values, row_values)
            spear_temp, spear_p_temp = spearmanr(x_values, row_values)
            r_temp, pval_temp = self.typ(x_values, row_values)
            means_temp = row_values.mean()
            sd_temp = row_values.std()
            sd[index] = sd_temp
            means[index] = means_temp
            fit = x_values * w_temp + b_temp
            lr.append(np.array(fit))
            pval[index] = pval_temp
            R[index] = r_temp
            spear[index] = spear_temp
            spear_p[index] = spear_p_temp

        self.df['mean beta'] = means
        self.df['deviation of beta'] = sd
        self.df['lr'] = lr
        self.df['beta ~ Age'] = R
        self.df['R_pval'] = pval
        self.df['spear beta ~ Age'] = spear
        self.df['spear R_pval'] = spear_p
        print('Finished!')
    
    def calculate_resid(self):
        self.calculation()
        corr = np.zeros(self.df.shape[0], dtype=np.float32)
        corr_pval = np.zeros(self.df.shape[0], dtype=np.float32)
        spear_corr = np.zeros(self.df.shape[0], dtype=np.float32)
        spear_corr_pval = np.zeros(self.df.shape[0], dtype=np.float32)

        print('Calculating correlation of residuals absolute values...')

        for index in range(self.df.shape[0]):
            row = self.df.iloc[index]
            row_values = row[:self.x.shape[0]]
            nans = row_values.isna()
            row_values = row_values[~nans].to_numpy()
            x_values = self.x[~nans].to_numpy()
            res_temp = np.abs(row_values - row['lr'])
            corr_res, corr_res_pval = self.typ(x_values, res_temp)
            spear_corr_res, spear_corr_res_pval = spearmanr(x_values, res_temp)
            corr[index] = corr_res
            corr_pval[index] = corr_res_pval
            spear_corr[index] = spear_corr_res
            spear_corr_pval[index] =spear_corr_res_pval

        self.df['delta beta ~ Age'] = corr
        self.df['ResR_pval'] = corr_pval
        self.df['spear delta beta ~ Age'] = spear_corr
        self.df['spear ResR_pval'] = spear_corr_pval


    @staticmethod
    def describe_dist(data, dist):
        summary = describe(data[dist])
        minimum = summary[1][0]
        maximum = summary[1][1]
        mean = summary[2]
        var = summary[3]
        skew = summary[4]
        kurtosis = summary[5]
        print(f'''
              Distribution statistics for {dist}:
              {'-'*40}
              --minimum: {minimum},
              --maximum: {maximum},
              --mean: {mean},
              --standard deviation {np.sqrt(var)},
              --skewness: {skew},
              --kurtosis: {kurtosis}
              {'-'*40}
              ''')
    
    def __call__(self, plot_distribution:bool = True, show_summary:bool = True):
        self.calculate_resid()

        print('Cleaning the dataframe...')
        self.df.drop(['lr'], axis = 1, inplace = True)
        self.df['R_sign'] = np.where(self.df['beta ~ Age'] >= 0, 1, 0)
        print('Finished!')
        print('*'*30)

        if show_summary:
            self.describe_dist(self.df, 'beta ~ Age')
            self.describe_dist(self.df, 'delta beta ~ Age')

        if plot_distribution:
            count = Counter(self.df.R_sign)
            count = pd.DataFrame(count, index=range(1))
            plt.figure(figsize=(12,8))
            sns.barplot(count)
            plt.title('Sign of correlation distribution')
            plt.show()

            df_plot = self.df[['beta ~ Age', 'delta beta ~ Age']]
            plt.figure(figsize=(12,8))
            sns.pairplot(df_plot)
            plt.show()

        return self.df
    
class SeqParser:
    def __init__(self, dataframe, seq:str= 'mouse_dict.pkl'):
        """
            Class for sequence parsing:

            Read sequence ->
            Parse index to chromosome and position columns ->
            Clean strange positions ->
            Calculate window ->
            Count CpGs inside window ->
            Cut sequence by window ->
            Count features inside sequence (default CG and TG) ->
            Delete unneccesary columns.

            ****
            Args: dataframe - data with targets, seq - path to sequence.pkl

            ******
            Output: dataframe with Chr, Pos, Start, End and features columns

            *******
            Example: data_table = SeqParser(data_with_corrs, 'mouse_dict.pkl')(window=10000, features=['CG','TG'])
        """
        self.df = dataframe.reset_index()
        self.seq = pickle.load(open(seq, 'rb'))
    
    def chr_maker(self, window:int = 1024):
        self.df[['Chr', 'Pos']] = self.df['index'].str.split('_', expand=True, n=1)
        self.df['flag'] = self.df.Pos.str.match(r'[A-z]')
        self.df = self.df[self.df.flag == 0]
        self.df.drop(['flag', 'index'], axis = 1, inplace = True)
        self.df['Pos'] = np.uint32(self.df.Pos.values)
        self.df['Start'] = self.df.Pos - window
        self.df['End'] = self.df.Pos + window
        df = self.df[['spear ResR_pval','spear delta beta ~ Age','spear R_pval','spear beta ~ Age','Chr', 'Pos', 'Start', 'End', 'beta ~ Age', 'delta beta ~ Age', 'R_sign', 'mean beta', 'R_pval', 'ResR_pval', 'deviation of beta']]
        return df
    
    def seq_count(self, features, window, df):
        feat_dict = {}
        lowseq = np.zeros(df.shape[0], dtype=np.float32)
        coverage = np.zeros(df.shape[0], dtype=np.uint8)

        for feature in features:
            feat_dict[feature] = np.zeros(df.shape[0], dtype=np.float32)

        for index in range(self.df.shape[0]):
            row = df.iloc[index]
            cpg_low, cpg_high = row['Pos'] - 1, row['Pos'] + 1
            seq = self.seq[row['Chr']][row['Start']:row['End']]
            seq_cpg = self.seq[row['Chr']][cpg_low:cpg_high]
            lowseq_temp = len(re.findall(r'[a-z]', seq)) / (2 * window)
            coverage[index] = len(re.findall(r'[a-z]', seq_cpg)) >= 1
            lowseq[index] = lowseq_temp
            for key in feat_dict.keys():
                feat_temp = seq.upper().count(key)
                feat_dict[key][index] = feat_temp

        df['LowConf'] = lowseq
        df['LowCoverage'] = coverage
        for feature in features:
            df[feature] = feat_dict[feature]
        return df
    

    @staticmethod
    def count_cpg_vectorized(start_array, end_array, vals):
        vals_array = np.asarray(vals)

        vals_sorted = np.sort(vals_array)

        counts = np.zeros(len(start_array), dtype=np.uint32)

        for i in range(len(start_array)):
            start_index = np.searchsorted(vals_sorted, start_array[i])
            end_index = np.searchsorted(vals_sorted, end_array[i], side='right')

            counts[i] = end_index - start_index  

        return counts
    
    def __call__(self, features:list = ['CG', 'TG'], window:int = 1024):
        print('*'*30)
        print('Defining cuts\' bounds...')
        df = self.chr_maker(window=window)
        start_array = df['Start'].values
        end_array = df['End'].values
        vals = df.Pos.values  
        print('Counting CpGs inside cut...')
        counts = self.count_cpg_vectorized(start_array=start_array, end_array=end_array, vals=vals)
        df['count_cpg'] = counts
        print(f'Counting features: {features} and low-confidence regions...')
        df = self.seq_count(df=df, features=features, window=window)
        print('Finished!')
        print('*'*30)
        return df

class Projector:
    def __init__(self, points, type, n_components):
        """
            Mixin class for projection obtaining
        """
        if 'Chr' in points.columns:
            points_ = points.drop('Chr', axis=1)
            self.points = points_
        elif 'Age' in points.columns:
            points_ = points.drop('Age', axis=1)
            self.points = points_
        elif 'R_sign' in points.columns:
            points_ = points.drop('R_sign', axis=1)
            self.points = points_
        else:
            self.points = points
        self.components = n_components
        self.type = type
    
    @property
    def project(self):
        if self.type == 'TSNE':
            projector = TSNE(n_components = self.components)
        elif self.type == 'PCA':
            projector = PCA(n_components = self.components, svd_solver = 'randomized')
        elif self.type == 'UMAP':
            projector = umap.UMAP(n_components=self.components).fit(self.points.values)
            projector = projector.transform(self.points.values)
            return projector
        else:
            projector = TruncatedSVD(n_components = self.components)
        projection = projector.fit_transform(self.points.values)
        return projection

class Plotter(Projector):
    def __init__(self, points, type:str = 'PCA',
                  n_components:int = 2, labeled:bool = False, clust = None, 
                  chrs_label:bool = False, age_label:bool = False, column:str = None,
                  legend:bool = True, inp:list = None):
        """
            Class for projection plotting. Enherits from Projector it's methods and attributes

            ****
            Args: points - data to project, type - type of projector, n_components - num of components,
                  labeled - whether to label points (if clust is None, plots sign of correlation), 
                  clust - labels of clusters
            
            ******
            Output: Plot of projection

            *******
            Example: Plotter(data_cleaned, type = 'PCA', n_components = 2)()
        """
        super(Plotter, self).__init__(points=points, type=type, n_components=n_components)
        self.legend = legend
        points = points.set_index(np.arange(len(points)))
        self.chrs_label = chrs_label
        self.age_label = age_label
        if chrs_label:
            if 'Chr' not in points.columns:
                raise ValueError('\'Chr\' column should be in the dataset')
            else:
                self.chr_dict = {}
                chrs = points.Chr.unique()
                for chr in chrs:
                    self.chr_dict[chr] = points.loc[points.Chr == chr].index
        elif age_label:
            points = points.T.reset_index()
            ages = points.Age.unique()
            self.age_dict = {}
            for age in ages:
                self.age_dict[age] = points.loc[points.Age == age].index

        elif column is not None:
            self.col = True
            unique = points[column].unique()
            self.dict_val = {}
            for val in unique:
                self.dict_val[val] = points.loc[points[column] == val].index
        if labeled:
            self.positive = points[points.R_sign == 1].index
            self.neg = points[points.R_sign == 0].index

        self.clust = clust
        self.projection = self.project
        self.labeled = labeled
    
    @property
    def plot_2d(self):
        z_2d = self.projection

        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap('jet')

        if not self.labeled:
            if self.chrs_label:
                colors = cmap(np.linspace(0, 1.0, len(self.chr_dict.keys())))
                for chr, color in zip(self.chr_dict.keys(), colors):
                    plt.scatter(z_2d[self.chr_dict[chr]][:,0], z_2d[self.chr_dict[chr]][:,1], label = f'{chr}', color=color)
            elif self.clust is not None:
                colors = cmap(np.linspace(0, 1.0, len(self.clust.keys())))
                for clust, color in zip(self.clust.keys(), colors):
                    plt.scatter(z_2d[self.clust[clust]][:,0], z_2d[self.clust[clust]][:,1], label = f'{clust}', color=color)
            elif self.age_label:
                colors = cmap(np.linspace(0, 1.0, len(self.age_dict.keys())))
                for age, color in zip(self.age_dict.keys(), colors):
                    plt.scatter(z_2d[self.age_dict[age]][:,0], z_2d[self.age_dict[age]][:,1], label = f'{age}', color=color)
            elif self.col:
                colors = cmap(np.linspace(0, 1.0, len(self.dict_val.keys())))
                for val, color in zip(self.dict_val.keys(), colors):
                    plt.scatter(z_2d[self.dict_val[val]][:,0], z_2d[self.dict_val[val]][:,1], label = f'{val}', color=color)
            else:
                plt.scatter(z_2d[:, 0], z_2d[:, 1])
        else:
            plt.scatter(z_2d[self.positive][:,0], z_2d[self.positive][:,1], label = 'Positive sign of R')
            plt.scatter(z_2d[self.neg][:,0], z_2d[self.neg][:,1], label = 'Negative sign of R')

        plt.xlabel('Axis 1')
        plt.ylabel('Axis 2')

        plt.title('2D projection')

        plt.grid('True')

        if self.legend:
            plt.legend(loc='upper right')

        plt.show()
    
    @property
    def plot_3d(self):
        z_2d = self.projection
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('jet')
        if not self.labeled:
            if self.chrs_label:
                colors = cmap(np.linspace(0, 1.0, len(self.chr_dict.keys())))
                for chr, color in zip(self.chr_dict.keys(), colors):
                    x = z_2d[self.chr_dict[chr]][:, 0]
                    y = z_2d[self.chr_dict[chr]][:, 1]
                    z = z_2d[self.chr_dict[chr]][:, 2]

                    ax.scatter(x, y, z, label = f'{chr}', color = color)
            elif self.clust is not None:
                colors = cmap(np.linspace(0, 1.0, len(self.clust.keys())))
                for clust, color in zip(self.clust.keys(), colors):
                    x = z_2d[self.clust[clust]][:, 0]
                    y = z_2d[self.clust[clust]][:, 1]
                    z = z_2d[self.clust[clust]][:, 2]

                    ax.scatter(x, y, z, label = f'{clust}', color=color)
            elif self.age_label:
                colors = cmap(np.linspace(0, 1.0, len(self.age_dict.keys())))
                for age, color in zip(self.age_dict.keys(), colors):
                    x = z_2d[self.age_dict[age]][:, 0]
                    y = z_2d[self.age_dict[age]][:, 1]
                    z = z_2d[self.age_dict[age]][:, 2]

                    ax.scatter(x, y, z, label = f'{age}', color = color)
            elif self.col:
                colors = cmap(np.linspace(0, 1.0, len(self.dict_val.keys())))
                for val, color in zip(self.dict_val.keys(), colors):
                    x = z_2d[self.dict_val[val]][:, 0]
                    y = z_2d[self.dict_val[val]][:, 1]
                    z = z_2d[self.dict_val[val]][:, 2]

                    ax.scatter(x, y, z, label = f'{val}', color = color)
            else:
                x = z_2d[:, 0]
                y = z_2d[:, 1]
                z = z_2d[:, 2]

                ax.scatter(x, y, z)
        else:
            x = z_2d[self.positive][:, 0]
            y = z_2d[self.positive][:, 1]
            z = z_2d[self.positive][:, 2]

            x1 = z_2d[self.neg][:, 0]
            y1 = z_2d[self.neg][:, 1]
            z1 = z_2d[self.neg][:, 2]

            ax.scatter(x, y, z, label = 'Positive sign of R')
            ax.scatter(x1, y1, z1, label = 'Negative sign of R')

        ax.set_xlabel('Axis 1')
        ax.set_ylabel('Axis 2')
        ax.set_zlabel('Axis 3')
        ax.set_title('3D projection')

        if self.legend:
            ax.legend()

        plt.show()

    def __call__(self):
        if self.components == 2:
            self.plot_2d
        elif self.components == 3:
            self.plot_3d
        else:
            raise ValueError('No more than 3 components.')

class Cluster(Projector):
    def __init__(self, points, n_components:int = 2, type:str = 'PCA', 
                 cluster:str = 'KMeans', num_clusters:int = 2, use_proj:bool = True):
        """
            Class for clustering. Enherits from Projector.

            ****
            Args: points - data, n_components - num of components, type - type of projector,
                  cluster - type of clustering algorithm, use_proj - whether to use projection

            ******
            Output: indexes of two clusters

            *******
            Example: clusters = Cluster(points, num_clusters = 2, use_proj = False)()
        """
        super(Cluster, self).__init__(points=points, type=type, n_components=n_components)
        self.type_cluster = cluster
        points = points.set_index(np.arange(len(points)))
        if use_proj:
            if 'Chr' in points.columns:
                points = points.drop('Chr', axis = 1)
            elif 'R_sign' in points.columns:
                points = points.drop('R_sign', axis=1)
            self.proj = self.project
        else:
            if 'Chr' in points.columns:
                points = points.drop('Chr', axis = 1)
            elif 'R_sign' in points.columns:
                points = points.drop('R_sign', axis=1)
                self.points = points
            self.proj = points
        self.num_clusters = num_clusters
        self.clust = self.cluster
        
    @property
    def cluster(self):
        if self.type_cluster == 'KMeans':
            model = KMeans(n_clusters=self.num_clusters, n_init='auto')
            clust = model.fit_predict(self.proj)
        elif self.type_cluster == 'Spectral':
            model = SpectralClustering(n_clusters=self.num_clusters)
            clust = model.fit_predict(self.proj)
        elif self.type_cluster == 'Agg':
            model = AgglomerativeClustering(n_clusters = self.num_clusters)
            clust = model.fit_predict(self.proj)
        elif self.type_cluster == 'GMM':
            model = GaussianMixture(n_components = self.num_clusters, random_state = 42)
            clust = model.fit_predict(self.proj)
        
        return clust
    
    def __call__(self):
        result = {}
        for n in range(self.num_clusters):
            result[f'Cluster {n}'] = np.where(self.clust == n)[0]
        return result

class MLHandler:
    def __init__(self, X=None, y=None, score = r2_score, test_size:int = 3, 
                 seed:int = 42, scale:bool = True, numeric_features:list = None):
        """
            Class with ML utils. Doesn't have a unified __call__() method,
            as it is a collection of functions for ML task.

            Train-test split is generated automatically!

            NB! X should contain 'Chr' column, and this should be the only text column

            ****
            Args: (X,y) - data, score - metric,
                  test_size - number of chromosomes for test, 
                  seed - reproducable train-test

            *******
            Example: Depends on a method that you use. Look documentation for methods.
        """

        if (X is not None) & (y is not None):
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_test(X=X, y=y, test_size=test_size, seed=seed)
            if scale:
                scaler = StandardScaler()
                self.X_train[numeric_features] = scaler.fit_transform(self.X_train[numeric_features])
                if test_size != 0:
                    self.X_test[numeric_features] = scaler.transform(self.X_test[numeric_features])

        self.score = score
    
    def try_models(self, models:list, regression:bool = True):
        """
            Method for models trying.

            ****
            Args: models - list of sklearn models, 
                  regression - specifying the task to take a proper metric

            ******
            Output: score dict and barplot

            *******
            Example: 

            models = [LinearRegression(), DecisionTreeRegressor()]
            scores = utils.MLHandler(X = X, y = y, test_size=2).try_models(models=models)

            Also, a list of one element can be specified and then it will be standard .fit -> .predict
            for one model.

            Note: haven't tested for more than six models. Maybe, the barplot will be fucked up
        """
        scores = {}
        if 'Chr' in self.X_train.columns:
            X_train = self.X_train.drop('Chr', axis = 1)
        if 'Chr' in self.X_test.columns:
            X_test = self.X_test.drop('Chr', axis = 1)
        for model in models:
            print('*'*30)
            print(f'Trying model {model}...')
            print('Fit...')
            model.fit(X_train, self.y_train)

            print('Predict...')
            prediction = model.predict(X_test)
            prediction_train = model.predict(X_train)

            if regression:
                score = r2_score(self.y_test, prediction)
                score_train = r2_score(self.y_train, prediction_train)
            else:
                score = f1_score(self.y_test, prediction)
                score_train = f1_score(self.y_train, prediction_train)
            
            if (len(models) == 1)&(regression):
                mae_test = mean_absolute_error(self.y_test, prediction)
                plt.scatter(self.y_test, prediction)
                plt.plot(np.linspace(-1,1,100), np.linspace(-1,1,100), ls = '--', c = 'r')
                plt.text(-1, 1, f'$R2\ test = {np.round(score,2)}$', fontsize = 12, c = 'r')
                plt.text(-1, 0.85, f'$MAE\ test = {np.round(mae_test,2)}$', fontsize = 12, c = 'g')
                plt.title(f'True vs. Predicted. Model: {model}')
                plt.xlabel('True')
                plt.ylabel('Predicted')
                plt.grid()
                plt.show()
            
            score_dict = {'train':score_train, 'test':score}
            scores[f'{model}'] = score_dict
        print('*'*30)
        print('Done!')
        print('Plotting results...')
        sc = pd.DataFrame(scores)
        sc = sc.T.reset_index()
        if regression:
            ylab = r'$R^2$ score'
        else:
            ylab = r'$F1$ score'
        sc.plot(x = 'index', kind = 'bar', stacked=False, 
         rot=30, grid=True, fontsize=6, 
         title='Result of models testing', ylabel=ylab, xlabel='Model')
        plt.show()
        return scores

    @staticmethod
    def train_test(X, y, test_size, seed):
        if seed is not None:
            np.random.seed(seed)
        chrs = X.Chr.unique()
        test = np.random.choice(chrs, size=test_size, replace=False)
        print(f'Selected chromosomes {test} for test')

        X_train = X.loc[~X.Chr.isin(test)].copy()
        X_test = X.loc[X.Chr.isin(test)].copy()

        if y is not None:
            y_train = y.loc[y.index.isin(X_train.index)].copy()
            y_test = y.loc[y.index.isin(X_test.index)].copy()
        else:
            y_train, y_test = 0, 0
            
        return X_train, X_test, y_train, y_test
    

    def permutation_importance(self, model, random_state:int = 42):
        """
            Perform permutation importance for a specific model.

            perm_imp = MLHandler(X, y).permutation_importance(model)

            ******
            Output: sklearn permutation importance object
        """
        if 'Chr' in self.X_test.columns:
            X_test = self.X_test.drop('Chr', axis = 1)

        r = permutation_importance(model, X_test, self.y_test,
                           n_repeats=30,
                           random_state=random_state)

        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(f"{self.X_train.columns[i]:<8}" + ' '
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")
        
        return r
    
    @staticmethod
    def bootstrap(X, y=None, number:int = 200):
        """
            Perform a bootstrap by index.

            This is a staticmethod, so it doesn't depend on a constructor arguments

            ******
            Output: X_bootstrapped, y_bootstrapped
        """
        N = X.shape[0]
        sampler = np.random.randint(low = 0, high = N, size = number)
        sample = X.loc[X.index.isin(sampler)].copy()
        if y is not None:
            sample_y = y.loc[y.index.isin(sampler)].copy()
            return sample, sample_y
        else:
            return sample, None

    @staticmethod
    def bootstrap_value(X, y=None, value:str = 'R', number:int = 200):
        """
            Perform bootstrap by value

            ****
            Args: X, y - data, value - name of value to bootstrap on, number = size

            ******
            Output: X_bootstrapped, y_bootstrapped
        """
        vals = X[value].unique()
        count = Counter(vals)
        weights = count.values()
        prob = []
        for val in weights:
            val = np.round(val/X.shape[0],15)
            prob.append(val)
        prob[-1] = prob[-1] + 1 - sum(prob)
        sampled = np.random.choice(list(vals), p=prob, replace=False, size=number)
        sample = X.loc[X[value].isin(sampled)].copy()
        if y is not None:
            sample_y = y.loc[y.index.isin(sample.index)].copy()
            return sample, sample_y
        else:
            return sample, None

    def bootstrap_master(self, X, y=None, number:int = 2000, value = None, typ:str = 'simple'):
        """
            ??????

            Maybe I don't need this at all
        """
        result_x = []
        result_y = []
        chrs = X.Chr.unique()
        for chr in chrs:
            X = X.loc[X.Chr.isin(chr)].copy()
            y = y.loc[y.index.isin(X.index)].copy()

            if typ == 'simple':
                X, y = self.bootstrap(X, y, number)
            else:
                X, y = self.bootstrap_value(X, y, value, number)
            
            result_x.append(X)
            if y is not None:
                result_y.append(y)
        
        data_strapped_x = pd.concat(result_x, join='inner')
        if y is not None:
            data_strapped_y = pd.concat(result_y, join='inner')
            return data_strapped_x, data_strapped_y
        else:
            return data_strapped_x
    
    @staticmethod
    def return_shuffled(data:list):
        random_keys = [random.random() for _ in data]
        data_shuffled = [x for _, x in sorted(zip(random_keys, data))]
        return data_shuffled

    @staticmethod
    def extract_suffix_number(s):
        match = re.search(r'(\d+)$', s)
        if match:
            return int(match.group(1))
        return float('inf')
    
    def create_weights(self, X_train, y_train, bin_type:str = 'Rice'):
        n = X_train.shape[0]
        if bin_type == 'Rice':
            bin_num = int(2 * (n**(1/3)))
        elif bin_type == 'sqrt':
            bin_num = int(np.sqrt(n))
        elif bin_type == 'auto':
            bin_num = 'auto'
        
        hist, edges = np.histogram(y_train, bins=bin_num)

        weights = np.zeros_like(y_train)

        for idx in range(len(edges) - 1):
            weights[np.where((y_train >= edges[idx]) & (y_train < edges[idx + 1]))[0]] = hist[idx] if hist[idx] > 0 else 1

        weights = weights[:,0]

        return 1 / weights
    
    def cross_val(self, model, cv:int = 5, one_by_one:bool = False, bin_type:str = None):
        """
            Perform KFold cross-validation.

            Can be used for standard KFold cross-val or for single chromosome regime

            ****
            Args: cv - number of folds, one_by_one - test folds have one chromosome

            ******
            Output: cross-val score, plots
        """
        chrs = self.X_train.Chr.unique()
        if not one_by_one:
            vals = self.return_shuffled(list(chrs))
        else:
            vals = sorted(list(chrs), key=self.extract_suffix_number)
        if one_by_one:
            cross = [[val] for val in vals]
        else:
            cross = [vals[i:i + len(vals)//cv] for i in range(0, len(vals), len(vals)//cv)]
        score = [] if not one_by_one else {}
        print('*'*30)
        print('Performing cross validation...')
        for i in tqdm(range(len(cross))):
            set = cross[i]
            X_train = self.X_train.loc[~self.X_train.Chr.isin(set)]
            X_test = self.X_train.loc[self.X_train.Chr.isin(set)]

            y_train = self.y_train.loc[self.y_train.index.isin(X_train.index)]
            y_test = self.y_train.loc[self.y_train.index.isin(X_test.index)]

            X_train.drop('Chr', inplace = True, axis = 1)
            X_test.drop('Chr', inplace = True, axis = 1)

            if bin_type is not None:
                weights = self.create_weights(X_train=X_train, y_train=y_train, bin_type=bin_type)
                model.fit(X_train, y_train, sample_weight=weights)
            else:
                model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            metric = self.score(y_test, prediction)

            if not one_by_one:
                score.append(metric)
            else:
                score[set[0]] = metric
        print('Done!')
        print('*'*30)
        if self.score == r2_score:
            ylab = r'$R^2$ score'
            ylab_each = r'$R^2$ score for each chromosome'
        else:
            ylab = r'$F1$ score'
            ylab_each = r'$F1$ score for each chromosome'
        if not one_by_one:
            sns.boxplot(score)
            plt.title('Cross val score')
            plt.ylabel(ylab)
        else:
            score = pd.DataFrame(score, index=range(len(chrs)))
            plt.figure(figsize=(14,10))
            sns.barplot(score)
            plt.title(ylab_each)
            plt.ylabel(ylab)
        
        return score
    
    def grid_search_cv(self, model, params:dict, cv:int = 5, n_jobs = None, bin_type:str = None):
        """
            Perform grid search over parameters.

            This method can be modified to work better. In progress...

            Haven't done multiprocess iteration yet. In progress...

            ****
            Args: params - parameters dictionary in a default form:
                  {'n_estimators':[100,200], 'max_depth':[1,2,3,...], ...},
                  cv - number of folds, n_jobs - only for models that support it
            
            ******
            Output: best params by score mean value and by standard deviation of score

            Note! Model should be passed without brackets. So, if it's Ridge, 
            than model = Ridge, not Ridge()!!!
        """
        chrs = self.X_train.Chr.unique()
        vals = self.return_shuffled(list(chrs))
        cross = [vals[i:i + len(vals)//cv] for i in range(0, len(vals), len(vals)//cv)]
        grid = ParameterGrid(params)
        param_grid = [prod for prod in grid]
        score_param = []
        print('*'*30)
        print(f'Performing grid search with cv = {cv}, will be {len(param_grid)} params combinations total total...')
        for i in tqdm(range(len(param_grid))):
            param = param_grid[i]
            score_fit = []
            for set in cross:
                X_train = self.X_train.loc[~self.X_train.Chr.isin(set)]
                X_test = self.X_train.loc[self.X_train.Chr.isin(set)]

                y_train = self.y_train.loc[self.y_train.index.isin(X_train.index)]
                y_test = self.y_train.loc[self.y_train.index.isin(X_test.index)]

                X_train.drop('Chr', inplace = True, axis = 1)
                X_test.drop('Chr', inplace = True, axis = 1)

                if n_jobs is not None:
                    model_par = model(**param, n_jobs = n_jobs)
                else:
                    model_par = model(**param)

                if bin_type is not None:
                    weights = self.create_weights(X_train=X_train, y_train=y_train)
                    model_par.fit(X_train, y_train, sample_weight=weights)
                else:
                    model_par.fit(X_train, y_train)

                prediction = model_par.predict(X_test)

                metric = self.score(y_test, prediction)

                score_fit.append(metric)
            score_param.append(score_fit)
        print('Done!')
        print('*'*30)

        means = [np.mean(scores) for scores in score_param]
        stds = [np.std(scores) for scores in score_param]

        best_mean = np.argmax(means)
        best_std = np.argmin(stds)

        best_params_mean = param_grid[best_mean]
        best_params_stds = param_grid[best_std]

        print(f'Best params for mean score are: {best_params_mean}. Their std is {stds[best_mean]}')
        print(f'Best params for stds are: {best_params_stds}. Their mean is {means[best_std]}')

        print('*'*30)
        print('Done selection.')

        return best_params_mean, best_params_stds
    
    def randomized_search_cv(self, model, param_dist:dict, n_iters:int = 50, cv:int = 5, n_jobs = None,
                             bin_type:str = None):
        """
            Perform randomized search cv over sampled from a distribution parameters

            Haven't done multiprocess iteration yet. In progress...

            ****
            Args: model, param_dist - dict where keys are param names and values are
            scipy.stats distributions, n_iters - number of iterations, cv,
            n_jobs - only for models that support it.

            ******
            Output: best params by score mean value and by standard deviation of score

            Note! Model should be passed without brackets. So, if it's Ridge, 
            than model = Ridge, not Ridge()!!!
        """
        chrs = self.X_train.Chr.unique()
        vals = self.return_shuffled(list(chrs))
        cross = [vals[i:i + len(vals)//cv] for i in range(0, len(vals), len(vals)//cv)]
        score_param = []
        param_grid = []
        print('*'*30)
        print(f'Performing grid search with cv = {cv}, will be {n_iters} iterations total...')
        for i in tqdm(range(n_iters)):
            pd = param_dist.copy()
            for key in pd.keys():
                pd[key] = pd[key].rvs(1)[0]
            score_fit = []
            for set in cross:
                X_train = self.X_train.loc[~self.X_train.Chr.isin(set)]
                X_test = self.X_train.loc[self.X_train.Chr.isin(set)]

                y_train = self.y_train.loc[self.y_train.index.isin(X_train.index)]
                y_test = self.y_train.loc[self.y_train.index.isin(X_test.index)]

                X_train.drop('Chr', inplace = True, axis = 1)
                X_test.drop('Chr', inplace = True, axis = 1)

                if n_jobs is not None:
                    model_par = model(**pd, n_jobs = n_jobs, sample_weight=weights)
                else:
                    model_par = model(**pd, sample_weight=weights)

                if bin_type is not None:
                    weights = self.create_weights(X_train=X_train, y_train=y_train)
                    model_par.fit(X_train, y_train, sample_weight=weights)
                else:
                    model_par.fit(X_train, y_train)

                prediction = model_par.predict(X_test)

                metric = self.score(y_test, prediction)

                score_fit.append(metric)
            param_grid.append(pd.values())
            score_param.append(score_fit)
        print('Done!')
        print('*'*30)

        means = [np.mean(scores) for scores in score_param]
        stds = [np.std(scores) for scores in score_param]

        best_mean = np.argmax(means)
        best_std = np.argmin(stds)

        best_params_mean = param_grid[best_mean]
        best_params_stds = param_grid[best_std]

        print(f'Best params for mean score are: {best_params_mean}. Their std is {stds[best_mean]}')
        print(f'Best params for stds are: {best_params_stds}. Their mean is {means[best_std]}')

        print('*'*30)
        print('Done selection.')

        return best_params_mean, best_params_stds


def default_pipe(dataset: str, path:str, impute:bool = False, typ = pearsonr, column:str = 'Cell', plot_distrib:bool = False,
                 plot_distribution:bool = True, show_summary:bool = True,
                 features:list = ['CG', 'TG'], window:int = 10000,
                 imp_method:str = 'mean', threshold: float | Any = None,
                 organ:str = 'mouse'):
    """
        Default pipe for data processing.

        ****
        Args: dataset - raw methylation data,
              impute - whether to impute or not (I haven't),
              imp_method - method of imputation,
              typ - type of correlation (default pearsonr),
              path - path to your .pkl file with DNA sequence,
              plot_distribution - whether to plot correlations distribution,
              show_summary - prints statistics of correlation distribution,
              features - list of features to count in a sequence (default ['CG', 'TG']),
              window - size of window to cut sequence,
              threshold - threshold for NaN in df
        
        ******
        Output: processed dataframe.

        After that you can use annotation in R and ML with the help of MLHandler and sklearn methods.

        Also, there are two more classes: Plotter and Cluster that can help with the analysis.

        All the classes have their documentation, please read them if you think you don't understand 
        something.

        Cluster and Plotter are not tested, such as cross_val and grid_search_cv functions.
        If you'll find some errors, please tell me.
    """
    dataset_ = pd.read_pickle(dataset)
    res1 = PreProcess(dataframe=dataset_, imp_method=imp_method, impute=impute, organism=organ)(threshold=threshold, plot_distrib=plot_distrib, column=column)
    res2 = StatisticCounter(processed_data=res1, typ=typ)(plot_distribution=plot_distribution, show_summary=show_summary)
    res3 = SeqParser(dataframe=res2, seq=path)(features=features, window=window)
    res3.to_pickle('/tank/projects/vivis77_anal/HG19_new_preproc_new_utils_human_full_dict_512_with_sd.pickle')
    return None

def parse_chr(row):
    chr, pos = row['index'].split('_')[0], row['index'].split('_')[1]
    return pd.Series({'Chr': chr, 'Pos': pos})

def raw_prep_bootstrap(dataset):
    """
        Prepare raw dataset (without embeddings) for bootstrapping.

        ****
        Args: dataset - pd.dataframe after preproccessing (Preproccess or StatisticCounter)

        ******
        Output: prepared data

        After this, data can be used with MLHandler's bootstrapping methods. Reccomend 
        using bootstrap_master()
    """
    dataset.reset_index(inplace = True)
    dataset[['Chr', 'Pos']] = dataset.apply(lambda row: parse_chr(row), axis = 1)
    dataset['flag'] = dataset.Pos.str.match(r'[A-z]')
    dataset = dataset[dataset.flag == 0]
    dataset.drop(['flag', 'index'], axis = 1, inplace = True)
    dataset.drop(['Pos'], inplace = True, axis = 1)
    return dataset

def granges_prep(data):
    """
        Function for preparation of data for granges
        and R annotation

        ****
        Args: data - dataframe after default pipe

        ******
        Output: ready for granges dataframe
    """
    data_ = data[['Chr', 'Pos']]
    data_[['Start', 'End']] = pd.Series({'a':data_.Pos, 'b':data_.Pos})
    data_.drop('Pos', axis = 1, inplace = True)
    return data_

def count_ar_vectorized(df, upper_lag: int = 20):
    start_array = df['Start'].values
    vals = df['Pos'].values
    lag_results = []

    for val in tqdm(vals):
        mask = (df['Pos'] >= start_array) & (df['Pos'] <= val)
        cpgs = df.loc[mask, 'R'].values

        num_obs = len(cpgs)

        max_lag = min(upper_lag, num_obs - 1) 
        
        if num_obs < 2:
            lag_results.append(np.nan)  
            continue  

        aic_values = []
        for lag in range(1, max_lag + 1):
            try:
                model = AutoReg(cpgs, lags=lag, trend='n').fit()
                aic_values.append(model.aic)
            except (ZeroDivisionError, ValueError) as e: 
                aic_values.append(np.inf)  
            
        min_aic_index = np.argmin(aic_values)
        lag_results.append(min_aic_index)

    df['Lag'] = lag_results
    return df

