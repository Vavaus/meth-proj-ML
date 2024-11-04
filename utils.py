import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
import pickle
import warnings
import random

from scipy.stats import linregress, pearsonr, kendalltau, spearmanr, describe

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, f1_score, mean_absolute_error

from tqdm import tqdm

warnings.filterwarnings('ignore')

class PreProcess:
    def __init__(self, dataframe, impute:bool = False, imp_method:str = 'mean'):
        """
            Class for data preparation:

            Clean from NaN or impute -> 
            Select most variable by age tissue ->
            Convert to float32 ->
            Convert age to column and sites as index.

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

    def clear_df(self):
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
            self.df.dropna(axis = 1, inplace = True)
        print('Finished!')
        print('*'*30)
    
    def select_best_tissue(self, column:str = 'Tissue'):
        print(f'Looking for {column} with max values...')
        counts = Counter(self.df[column])
        keys, vals = list(counts.keys()), list(counts.values())
        best = keys[np.argmax(vals)]

        print(f'Choosing {column} == {best}')
        selected = self.df[self.df.Tissue == best]

        print('Dropping all unneccessary columns...')
        selected.drop([column, 'Sex', 'Strain', 'ID'], inplace = True, axis = 1)

        print('Sorting age and setting it as index...')
        selected = selected.sort_values(by = 'Age')
        selected.set_index('Age', inplace = True)

        print('Transposing...')
        selected = selected.astype('float32')
        selected = selected.T
        print('Finished!')
        print('*'*30)

        return selected
    
    def __call__(self):
        self.clear_df()
        res = self.select_best_tissue()
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
        self.x = self.df.columns.get_level_values('Age').astype('float32')
        self.typ = typ

    def calculate_stats(self, row):
        row = row.astype('float32')

        w, b, _, _, _ = linregress(self.x, row)

        r, _ = self.typ(self.x, row)
        
        result = pd.Series({'w': w, 'R': r, 'b': b})
        return result

    def calculate_fit(self, row):
        fit = row.w*self.x + row.b
        return pd.Series({'fit': fit})
    
    def calculation(self):
        print('*'*30)
        print('Calculating statistics...')

        self.df[['w', 'R', 'b']] = self.df.apply(lambda row: self.calculate_stats(row), axis = 1)

        print('Calculating LR...')
        self.df['lr'] = self.df.apply(lambda row: self.calculate_fit(row), axis = 1)

        print('Finished!')
        print('*'*30)
    
    def calculate_resid(self):
        self.calculation()
        print('Calculating residuals...')
        self.df['resid'] = self.df.apply(lambda row: row.iloc[:len(self.x)].values - row.lr, axis = 1)
        self.df['resid'] = np.abs(self.df.resid.values)

        print('Calculating correlation of residuals absolute values...')
        self.df['corr_resid'] = self.df.apply(lambda row: self.typ(self.x, row.resid.astype('float32'))[0], axis = 1)

        print('Finished!')
        print('*'*30)
    
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
        self.df.drop(['resid', 'lr', 'b'], axis = 1, inplace = True)
        self.df['R_sign'] = np.where(self.df.R >= 0, 1, 0)
        print('Finished!')
        print('*'*30)

        if show_summary:
            self.describe_dist(self.df, 'R')
            self.describe_dist(self.df, 'corr_resid')

        if plot_distribution:
            count = Counter(self.df.R_sign)
            count = pd.DataFrame(count, index=range(1))
            plt.figure(figsize=(12,8))
            sns.barplot(count)
            plt.title('Sign of correlation distribution')
            plt.show()

            df_plot = self.df[['R', 'corr_resid']]
            plt.figure(figsize=(12,8))
            sns.pairplot(df_plot)
            plt.title('Pairwise distribution of correlations')
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
    
    @staticmethod
    def parse_chr(row):
        chr, pos = row['index'].split('_')[0], row['index'].split('_')[1]
        return pd.Series({'Chr': chr, 'Pos': pos})
    
    def chr_maker(self, window:int = 10000):
        self.df[['Chr', 'Pos']] = self.df.apply(lambda row: self.parse_chr(row), axis = 1)
        self.df['flag'] = self.df.Pos.str.match(r'[A-z]')
        self.df = self.df[self.df.flag == 0]
        self.df.drop(['Len', 'index'], axis = 1, inplace = True)
        self.df['Pos'] = np.uint32(self.df.Pos.values)
        self.df[['Start', 'End']] = self.df.Pos - window, self.df.Pos + window
        df = self.df[['Chr', 'Pos', 'Start', 'End', 'R', 'corr_resid']]
        return df

    def cut_seq(self, df):
        df['Seq'] = df.apply(lambda row: self.seq[row.Chr][row.Start:row.End].upper(), axis = 1)
        return df

    @staticmethod
    def count_cpg_vectorized(start_array, end_array, vals):
        counts = np.zeros(len(start_array), dtype=np.uint32)
        for val in vals:
            counts += (val >= start_array) & (val <= end_array)
        return counts
    
    def count_feature(self, df, features:list):
        for feat in features:
            df[feat] = self.df.apply(lambda row: row.Seq.count(feat), axis = 1)
        return df
    
    def __call__(self, features:list = ['CG', 'TG'], window:int = 10000):
        df = self.chr_maker(window=window)
        start_array = df['Start'].values
        end_array = df['End'].values
        vals = df.Pos.values  
        counts = self.count_cpg_vectorized(start_array=start_array, end_array=end_array, vals=vals)
        df['count_cpg'] = counts
        df = self.cut_seq(df)
        df = self.count_feature(df=df, features=features)
        df = df.drop(['Seq'], axis = 1, inplace = True)
        return df

class Projector:
    def __init__(self, points, type, n_components):
        """
            Mixin class for projection obtaining
        """
        self.points = points
        self.components = n_components
        self.type = type
    
    @property
    def project(self):
        if self.type == 'TSNE':
            projector = TSNE(n_components = self.components)
        elif self.type == 'PCA':
            projector = PCA(n_components = self.components, svd_solver = 'randomized')
        else:
            projector = TruncatedSVD(n_components = self.components)
        projection = projector.fit_transform(self.points)
        return projection

class Plotter(Projector):
    def __init__(self, points, type:str = 'PCA',
                  n_components:int = 2, labeled:bool = False, clust = None):
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
        if labeled:
            if not clust:
                self.positive = points[points.R_sign == 1].index
                self.neg = points[points.R_sign == 0].index
            else:
                self.positive = clust['Cluster 0']
                self.neg = clust['Cluster 1']
        self.projection = self.project
        self.labeled = labeled
    
    @property
    def plot_2d(self):
        z_2d = self.projection

        plt.figure(figsize=(12, 10))

        if not self.labeled:
            plt.scatter(z_2d[:, 0], z_2d[:, 1])
        else:
            plt.scatter(z_2d[self.positive][:,0], z_2d[self.positive][:,1], label = 'Cluster 0')
            plt.scatter(z_2d[self.neg][:,0], z_2d[self.neg][:,1], label = 'Cluster 1')

        plt.xlabel('Axis 1')
        plt.ylabel('Axis 2')

        plt.title('2D projection')

        plt.grid('True')

        plt.legend()

        plt.show()
    
    @property
    def plot_3d(self):
        z_2d = self.projection
        if not self.labeled:
            x = z_2d[:, 0]
            y = z_2d[:, 1]
            z = z_2d[:, 2]
        else:
            x = z_2d[self.positive][:, 0]
            y = z_2d[self.positive][:, 1]
            z = z_2d[self.positive][:, 2]

            x1 = z_2d[self.neg][:, 0]
            y1 = z_2d[self.neg][:, 1]
            z1 = z_2d[self.neg][:, 2]
            
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, label = 'Cluster 0')
        if self.labeled:
            ax.scatter(x1, y1, z1, label = 'Cluster 1')

        ax.set_xlabel('Axis 1')
        ax.set_ylabel('Axis 2')
        ax.set_zlabel('Axis 3')
        ax.set_title('3D projection')
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
        if use_proj:
            self.proj = self.project
        else:
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
                 seed:int = 42, random_state = None):
        """
            Class with ML utils. Doesn't have a unified __call__() method,
            as it is a collection of functions for ML task.

            Train-test split is generated automatically!

            NB! X should contain 'Chr' column, and this should be the only text column

            ****
            Args: model - sklearn model, (X,y) - data, score - metric,
                  test_size - number of chromosomes for test, seed - reproducable train-test,
                  random_state - reproducable cross val split

            *******
            Example: Depends on a method that you use. Look documentation for methods.
        """

        self.random_state = random_state

        if X is not None:
            self.chrs = X.Chr.unique()

        if (X is not None) & (y is not None):
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_test(X=X, y=y, test_size=test_size, seed=seed)

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
        for model in models:
            print('*'*30)
            print(f'Trying model {model}...')
            print('Fit...')
            model.fit(self.X_train, self.y_train)

            print('Predict...')
            prediction = model.predict(self.X_test)
            prediction_train = model.predict(self.X_train)

            if regression:
                score = r2_score(self.y_test, prediction)
                score_train = r2_score(self.y_train, prediction_train)
            else:
                score = f1_score(self.y_test, prediction)
                score_train = f1_score(self.y_test, prediction_train)
            
            if (len(models) == 1)&(regression):
                mae_test = mean_absolute_error(self.y_test, prediction)
                plt.scatter(self.y_test, prediction)
                plt.plot(np.linspace(-1,1,100), np.linspace(-1,1,100), ls = '--', c = 'r')
                plt.text(-1, 1, f'$R2\ test = {np.round(score,2)}$', fontsize = 12, c = 'r')
                plt.text(-1, 0.85, f'$MAE\ test = {np.round(mae_test,2)}$', fontsize = 12, c = 'g')
                plt.title(f'True vs. Pred. Model: {model}')
                plt.xlabel('True')
                plt.ylabel('Pred')
                plt.grid()
                plt.show()
            
            score_dict = {'train':score_train, 'test':score, 'diff': np.abs(score_train - score)}
            scores[f'{model}'] = score_dict
        print('*'*30)
        print('Done!')
        print('Plotting results...')
        sc = pd.DataFrame(scores)
        sc = sc.T.reset_index()
        sc.plot(x = 'index', kind = 'bar', stacked=False, 
         rot=30, grid=True, fontsize=6, 
         title='Result of testing', ylabel='Score', xlabel='Model')
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

        X_test.drop(['Chr'], axis = 1, inplace = True)
        X_train.drop(['Chr'], axis = 1, inplace = True)

        y_train = y.loc[y.index.isin(X_train.index)].copy()
        y_test = y.loc[y.index.isin(X_test.index)].copy()

        return X_train, X_test, y_train, y_test
    

    def permutation_importance(self, model):
        """
            Perform permutation importance for a specific model.

            perm_imp = MLHandler(X, y).permutation_importance(model)

            ******
            Output: sklearn permutation importance object
        """
        r = permutation_importance(model, self.X_test, self.y_test,
                           n_repeats=30,
                           random_state=0)

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
        sampled = np.random.choice(vals, p=weights, replace=False, size=number)
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
        for chr in self.chrs:
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

    def shuffle(self):
        """
            Util function for reproducable cross val split

            Better not to touch
        """
        if self.random_state is not None:
            return self.random_state
        else:
            return np.random.uniform(low=0, high=1, size=1)
    
    def cross_val(self, model, cv:int = 5, one_by_one:bool = False):
        """
            Perform KFold cross-validation.

            Can be used for standard KFold cross-val or for single chromosome regime

            ****
            Args: cv - number of folds, one_by_one - test folds have one chromosome

            ******
            Output: cross-val score, plots
        """
        vals = random.shuffle(self.chrs, self.shuffle())
        if one_by_one:
            cv = 1
        cross = [vals[i:i + len(vals)//cv] for i in range(0, len(vals), len(vals)//cv)]
        score = [] if not one_by_one else {}
        print('*'*30)
        print('Performing cross validation...')
        for i in tqdm(range(len(cross))):
            set = cross[i]
            X_train = self.X_train.loc[~self.X_train.Chr.isin(set)].copy()
            X_test = self.X_train.loc[self.X_train.Chr.isin(set)].copy()

            y_train = self.y_train.loc[self.y_train.index.isin(X_train.index)].copy()
            y_test = self.y_train.loc[self.y_train.index.isin(X_test.index)].copy()

            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            metric = self.score(y_test, prediction)

            if not one_by_one:
                score.append(metric)
            else:
                score[set] = metric
        print('Done!')
        print('*'*30)
        if not one_by_one:
            sns.boxplot(score)
            plt.title('Cross val score')
            plt.ylabel(f'{self.score}')
        else:
            score = pd.DataFrame(score)
            sns.barplot(score)
            plt.title(f'Chromosomes {self.score}')
            plt.ylabel(f'{self.score}')
        
        return score
    
    def grid_search_cv(self, model, params:dict, cv:int = 5):
        """
            Perform grid search over parameters.

            This method can be modified to work better. In progress...

            ****
            Args: params - parameters dictionary in a default form:
                  {'n_estimators':[100,200], 'max_depth':[1,2,3,...], ...},
                  cv - number of folds
            
            ******
            Output: best params by score mean value and by standard deviation of score
        """
        vals = random.shuffle(self.chrs, self.shuffle())
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
                X_train = self.X_train.loc[~self.X_train.Chr.isin(set)].copy()
                X_test = self.X_train.loc[self.X_train.Chr.isin(set)].copy()

                y_train = self.y_train.loc[self.y_train.index.isin(X_train.index)].copy()
                y_test = self.y_train.loc[self.y_train.index.isin(X_test.index)].copy()

                model = model(**param)

                model.fit(X_train, y_train)

                prediction = model.predict(X_test)

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

def default_pipe(dataset, path, impute:bool = False, typ = pearsonr,
                 plot_distribution:bool = True, show_summary:bool = True,
                 features:list = ['CG', 'TG'], window:int = 10000,
                 imp_method:str = 'mean'):
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
              window - size of window to cut sequence
        
        ******
        Output: processed dataframe.

        After that you can use annotation in R and ML with the help of MLHandler and sklearn methods.

        Also, there are two more classes: Plotter and Cluster that can help with the analysis.

        All the classes have their documentation, please read them if you think you don't understand 
        something.

        Cluster and Plotter are not tested, such as cross_val and grid_search_cv functions.
        If you'll find some errors, please tell me.
    """
    res1 = PreProcess(dataframe=dataset, imp_method=imp_method, impute=impute)()
    res2 = StatisticCounter(processed_data=res1, typ=typ)(plot_distribution=plot_distribution, 
                                                          show_summary=show_summary)
    res3 = SeqParser(dataframe=res2, seq=path)(features=features, window=window)
    return res3

def parse_chr(row):
    chr, pos = row['index'].split('_')[0], row['index'].split('_')[1]
    return pd.Series({'Chr': chr, 'Pos': pos})

def raw_prep_bootstrap(dataset):
    dataset.reset_index(inplace = True)
    dataset[['Chr', 'Pos']] = dataset.apply(lambda row: parse_chr(row), axis = 1)
    dataset['flag'] = dataset.Pos.str.match(r'[A-z]')
    dataset = dataset[dataset.flag == 0]
    dataset.drop(['Len', 'index'], axis = 1, inplace = True)
    dataset.drop(['Pos'], inplace = True, axis = 1)
    return dataset

