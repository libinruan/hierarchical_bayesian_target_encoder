#%%
import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold      

class Timer:
    def __enter__(self):
        self.start=time.time()
        return self
    def __exit__(self, *args):
        self.end=time.time()
        self.hour, temp = divmod((self.end - self.start), 3600)
        self.min, self.second = divmod(temp, 60)
        self.hour, self.min, self.second = int(self.hour), int(self.min), round(self.second, 2)
        return self

class BayCatEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 group_cols, 
                 target_col='target', 
                 N_min=1, 
                 drop_original=False, 
                 CV=True, 
                 n_fold=5,
                 drop_intermediate=False,
                 delimiter='.',
                 verbosity=True,
                 random_seed=2020):
        self.group_cols = [group_cols] if isinstance(group_cols, str) else group_cols # List of column names combination: e.g. ['n1.n2.n4', 'n3.n4', 'n2'].
        self.target_col = target_col # String: 'target' by default.
        self.stats = defaultdict(dict) # key: column names combination; value: corresponding info about n, N, and computed code.
        self.N_min = N_min # regularization control
        self.drop_original = drop_original # toggle key for whether to drop original column name(s) or not.
        self.CV = CV # Bool
        self.n_fold = n_fold
        self.drop_intermediate = drop_intermediate
        self.delimiter = delimiter
        self.verbosity = verbosity # Bool
        self.seed = random_seed
        self.set_original_col = set()

    def fit(self, X, y):     
        self.col_subsets = self._generate_subsets(self.group_cols)
        df = pd.concat([X.copy(), y.copy()], axis=1)
        assert(isinstance(self.target_col, str))
        df.columns = X.columns.tolist() + [self.target_col] 
        assert(self._check_col_consistency(X))
        if not self.CV:
            self._single_fit(df)
        else:
            self._cv_fit(df)
        return self

    def _single_fit(self, df):
        for subset in self.col_subsets:
            df_stat, stat, cross_features = self._update(df, subset)
            features_encoded = cross_features + '_code'
            self.stats[cross_features] = pd.merge(
                stat, 
                df_stat.groupby(subset)[features_encoded].mean(), 
                left_index=True, 
                right_index=True)       
        return self 

    def _cv_fit(self, df):
        kf = StratifiedKFold(n_splits = self.n_fold, shuffle = True, random_state=self.seed)
        size_col_subsets = len(self.col_subsets)
        count_subset = 0
        for subset in self.col_subsets:
            count_subset += 1
            with Timer() as t:
                for i, (tr_idx, val_idx) in enumerate(kf.split(df.drop(self.target_col, axis=1), df[self.target_col])):
                    if self.verbosity: print(f'{subset} - Order {count_subset}/{size_col_subsets} - Round {i+1}/{self.n_fold}')
                    df_tr, df_val = df.iloc[tr_idx].copy(), df.iloc[val_idx].copy() # Vital for avoid "A value is trying to be set on a copy of a slice from a DataFrame." warning.
                    df_stat, stat, cross_features = self._update(df_tr, subset)
                    features_encoded = cross_features + '_code'
                    df.loc[df.index[val_idx], features_encoded] = pd.merge(
                            df_val[subset], 
                            df_stat.groupby(subset)[features_encoded].mean(),
                            left_on=subset,
                            right_index=True,
                            how='left'
                        )[features_encoded].copy() \
                        .fillna(df[self.target_col].mean())  
                self.stats[cross_features] = df.groupby(subset)[features_encoded].mean().to_frame()
            if self.verbosity: print(f'time elapsed: {t.hour} hours {t.min} mins {t.second} seconds')   
        return self        

    def _update(self, df, subset):
        self.global_prior_mean = df[self.target_col].mean()
        if len(subset) == 1:
            self.set_original_col.add(*subset)
            upper_level_cols = 'global'
            if not upper_level_cols + '_prior_mean' in df.columns:
                df.loc[:, upper_level_cols + '_prior_mean'] = self.global_prior_mean
        else:
            upper_level_cols = self.delimiter.join(subset[:-1]) # e.g. the n1.n2 subset's upper level feature is `n1`.
            if not upper_level_cols + '_prior_mean' in df.columns: 
                df.loc[:, upper_level_cols + '_prior_mean'] = pd.merge(
                        df[subset[:-1]], 
                        self.stats[upper_level_cols][upper_level_cols + '_code'], 
                        left_on=subset[:-1], 
                        right_index=True, 
                        how='left'
                    )[upper_level_cols + '_code'].copy()
        
        stat = df.groupby(subset).agg(
            n=(self.target_col, 'sum'),
            N=(self.target_col, 'count'),
            prior_mean=(upper_level_cols + '_prior_mean', 'mean')
        )
        # Calculate posterior mean
        df_stat = pd.merge(df[subset], stat, left_on=subset, right_index=True, how='left')
        df_stat['n'].mask(df_stat['n'].isnull(), df_stat['prior_mean'], inplace=True) 
        df_stat['N'].fillna(1., inplace=True)
        df_stat.loc[:, 'N_prior'] = df_stat['N'].map(lambda x: max(self.N_min - x, 0))
        df_stat.loc[:, 'alpha_prior'] = df_stat['prior_mean'] * df_stat['N_prior']
        df_stat.loc[:, 'beta_prior'] = (1. - df_stat['prior_mean']) * df_stat['N_prior'] # Large N -> zero N_prior -> zero alpha_prior and zero beta_prior -> if n is zero as well -> alpha prior, beta prior both zero -> alpha zero -> posterior mean = zero as well.           
        if len(subset) == 1:
            cross_features = subset[0]
        else:
            cross_features = self.delimiter.join(subset)
        df_stat.loc[:, cross_features + '_code'] = df_stat.apply(self._stat_mean, axis=1) # core # TEST set!!
        return df_stat, stat, cross_features

    def _generate_subsets(self, groups, delimiter='.'):
        subsets = defaultdict(list)    
        for g in groups:
            chain = g.split(delimiter)
            for i in range(len(chain)):
                if chain[i] and not chain[:i+1] in subsets[i]: subsets[i].append(chain[:i+1])
        ret = []
        for _, v in subsets.items():
            if not v in ret: ret.extend(v)
        return ret        

    def _stat_mean(self, X):
        df = X.copy()
        alpha = df['alpha_prior'] + df['n']
        beta = df['beta_prior'] + df['N'] - df['n']
        return alpha / (alpha + beta)

    def _check_col_consistency(self, df): 
        """Check whether columns specified in `self.group_cols` are all included in `df`.
        """        
        s = set()
        for col_subset in self.col_subsets:
            s |= set(col_subset)
        for col in s:
            if not col in df.columns: return False
        return True        

    def transform(self, X):
        assert(self._check_col_consistency(X))
        for subset in self.col_subsets:
            key = '.'.join(subset)
            X = pd.merge(
                    X, 
                    self.stats[key][key + '_code'], 
                    left_on=subset, 
                    right_index=True, 
                    how='left')
            if len(subset) == 1:
                X[key + '_code'].fillna(self.global_prior_mean)
            else:
                parent_key = '.'.join(subset[:-1]) + '_code'            
                X[key + '_code'].fillna(X[parent_key], inplace=True)
        if self.drop_original:
            for col in self.set_original_col:
                X.drop(col, axis=1, inplace=True)
                X.rename(columns={col+'_code': col}, inplace=True)
        if self.drop_intermediate: 
            for col in X.columns:
                if col.endswith('_code') and not col.strip('_code') in self.group_cols:
                    X.drop(col, axis=1, inplace=True)
        return X

#%%
if __name__ == '__main__':
    np.random.seed(1)
    k = 15
    n1 = np.random.choice(['a','b'], k)
    n2 = np.random.choice(['c','d'], k)
    n3 = np.random.choice(['e','f'], k)
    target = np.random.randint(0, 2, size=k)
    train = pd.DataFrame(
        {'n1': n1, 'n2': n2, 'n3':n3, 'target': target}, 
        columns=['n1', 'n2', 'n3', 'target']
    )
    train.columns = ['n1','n2','n3', 'target']
    
    display(train)

    k = 6
    n4 = np.random.choice(['a','b'], k)
    n5 = np.random.choice(['c','d'], k)
    n6 = np.random.choice(['e','f'], k)
    test = pd.DataFrame({'n4': n4, 'n2': n5, 'n3':n6})
    test.columns = ['n1','n2','n3']
    
    display(test)
    
    te = BayCatEncoder(
            'n1.n2.n3', #['n1.n2.n3', 'n2.n3', 'n3'], 
            target_col='target', 
            drop_original=False, 
            drop_intermediate=False,
            CV=True, 
            n_fold=3
        ) \
        .fit(train.drop('target', axis=1), train.target) 
    # te.transform(test)
    display(te.transform(test))

# %%
