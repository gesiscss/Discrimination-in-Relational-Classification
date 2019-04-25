import pandas as pd
import os, sys, time, argparse, warnings
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from libs.s3d.utils import create_paths

warnings.simplefilter("once")

random_state=None
PATH_DATA = 'splitted_data'

class DataSplitter(object):
    ''' split data;
        stratefied for classification; regulalr k-fold for regression
        for regression data, standardize data (both X and y; using scaler fit by training set on test set)
    '''
    def __init__(self, data_path, data_name, output,
                 classification_flag=True):

        ''' initializer
            parameters
            ----------
            data_path : str
                input data
            data_name : str
                data name
            classification_flag : bool
                whether it's for regression or classification
            output : string
                output folder directory
        '''
        if not data_path.endswith('.csv'):
            warnings.warn('.csv is required')
            sys.exit(0)

        self.datafn = data_path
        self.data_name = data_name
        self.output = os.path.join(output, PATH_DATA)
        if not create_paths(self.output):
            warnings.warn('Error while trying to create output folder: {}'.format(self.output))
            sys.exit(0)

        self.data = pd.read_csv(self.datafn)
        self.nrows, _ = self.data.shape
        self.classification_flag = classification_flag

    def split_data(self, num_folds=5, num_jobs=1):
        ''' generate equal folds of data. this is mainly for s3d
            apply stratification to account for class imbalance
            make this parallelizable
            parameters
            ----------
            num_folds : int
                the number of folds to use for cross validation
            the function will export each fold into `output`
            with names formatted as `data_name_i.csv` where `i` is the fold index
        '''

        self.num_folds = num_folds

        X = self.data[self.data.columns[self.data.columns!='target']].values
        if self.classification_flag:
            y = self.data['target'].values.astype(int)
            kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True,
                                 random_state=random_state)
        else:
            y = self.data['target'].values
            kf = KFold(n_splits=self.num_folds, shuffle=True,
                       random_state=random_state)
        ## split
        print('splitting {} data ({} rows) into {} folds'.format(self.data_name,
                                                                 self.nrows, self.num_folds))
        ## export different folds
        print('using {} cores'.format(num_jobs))
        num_jobs = min([num_jobs, self.num_folds])
        Parallel(n_jobs=num_jobs)(delayed(self.save_folds)(i, train_index, test_index)\
                                  for i, (train_index, test_index) in enumerate(kf.split(X, y)))


    def save_folds(self, i, train_index, test_index):
        start = time.time()
        print('working on fold {}'.format(i), end=' ')
        ## create the corresponding fold-folder to save test and train/test datasets
        out =  os.path.join(self.output,str(i))
        if not os.path.exists(out):
            os.makedirs(out)

        if self.classification_flag:
            train_index = self.adjust_rows(train_index)

        ## export train/tune dataset: use stratified row indices to rearrange the training set and make it stratified
        train_values = self.data.values[train_index]
        test_values = self.data.values[test_index]
        ## fit a scaler using training data
        ## transofrm is for regression
        if not self.classification_flag:
            scaler = StandardScaler()
            train_values = scaler.fit_transform(train_values)
            test_values = scaler.transform(test_values)
        ## save
        pd.DataFrame(train_values, columns=self.data.columns.values).to_csv(os.path.join(out,'train.csv'), index=False)
        pd.DataFrame(test_values, columns=self.data.columns.values).to_csv(os.path.join(out,'test.csv'), index=False)

        ## also export the number of rows for train/test into a text file
        with open(os.path.join(out, 'num_rows.csv'), 'w') as f:
            ## first train, then test
            f.write(str(train_index.size)+'\n')
            f.write(str(test_index.size)+'\n')
        assert train_index.size+test_index.size == self.nrows
        print('fold {0} (elapsed time: {1:.2f} seconds)'.format(i, time.time()-start))


    def adjust_rows(self, train_index):
        ''' adjust the rows in the training sets so that
            every interval of subsets in training will have the same ratio of positive/negative
            where intervals are determined by `num_folds-1`
        '''
        train_values = self.data.values[train_index]
        train_data = pd.DataFrame(train_values, columns=self.data.columns.values)

        X = train_data[train_data.columns[train_data.columns!='target']].values
        y = train_data['target'].values

        skf = StratifiedKFold(n_splits=self.num_folds-1, shuffle=True, random_state=random_state)

        stratified_row_indices = list()
        for _, tst_idx in skf.split(X, y):
            ## for every test fold, i will append it to make a "stratified training set"
            stratified_row_indices.extend(tst_idx)
        return pd.np.array(stratified_row_indices)

def main(csv_fn, data_name, output_folder, classification_flag, num_folds, num_jobs):
    classification_flag = bool(classification_flag)
    ds = DataSplitter(csv_fn, data_name, output_folder, classification_flag)
    ds.split_data(num_folds, num_jobs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", "csv_fn", type=str, help="data to be splitted")
    parser.add_argument("-n", "data_name", type=str, help="data name")
    parser.add_argument("-o", "output_folder", type=str, help="where to store results")
    parser.add_argument("-nf", "num_folds", type=int, help="number of folds")

    ## optional
    parser.add_argument("-cf", "--classification-flag", type=int,
                        choices=[0, 1], default=1,
                        help="whether the dataset is for classification or not (default 1 - yes); 0 for regression")
    parser.add_argument("-j", "--num-jobs", type=int, default=1,
                        help="the number of parallel jobs (default 1)")

    args = parser.parse_args()
    main(args.csv_fn, args.data_name, args.output_folder, args.classification_flag, args.num_folds, args.num_jobs)

