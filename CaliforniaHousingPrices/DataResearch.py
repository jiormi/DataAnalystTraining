import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from zlib import crc32
from sklearn.model_selection import train_test_split


#---------------- Split the data into a test sample ------------------

'''
Here, the book just goes through different items we have to take into account.
We need to be careful on the fact that with the simple shuffle_and_split_data function,
everytime that we want to split the data, we will have a different labelling, and eventually
we will have indeed run over the full dataset. So we should do the permutation
either choosing an specific seed, or adding a new index parameter and always choose
the test instances coming into the test dataset based on that. Also, we need to have a data
which is a real representation of the population, not skewed.
'''

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32
    
def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def shuffle_and_split_data(data,test_ratio):
    '''
    Each row has a label, the idea here is just to swap the labels in a random way
    just taking a percentage of the initial dataset. In this way we can choose our
    test sample. The function just returns the test and the trainning dataset.
    The iloc function just works like dataframe.iloc[row_selection, column_selection].
    In our case, we just take the indices of either train and test datasets.
    '''
    
    shuffled_indices = np.random.permutation(len(data))
    #print(f'data: {data}\n')
    #print(f'len(data): {len(data)}\n')
    #print(f'shuffled_indices: {shuffled_indices}\n')
    test_set_size = int(len(data) * test_ratio)
    #print(f'test_set_size: {test_set_size}\n')
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    #print(f'data.iloc[train_indices]: {data.iloc[train_indices]}\n')
    #print(f'data.iloc[test_indices: {data.iloc[test_indices]}\n')
    return data.iloc[train_indices], data.iloc[test_indices]
    


#---------------- Loading the data ------------------------

def load_california_housing_data(download_url=None, extract_to="cal_housing"):
    """
    Downloads, extracts, and loads the California Housing dataset from the StatLib repository.

    Args:
        download_url (str): Optional custom URL for the dataset.
        extract_to (str): Folder where data will be extracted.

    Returns:
        pd.DataFrame: The California housing data as a pandas DataFrame.
    """
    url = download_url or "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz"
    file_name = "cal_housing.tgz"

    # Download the dataset if it doesn't exist
    if not os.path.exists(file_name):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, file_name)
        print("Download complete.")

    # Extract the dataset if not already done
    if not os.path.exists(extract_to):
        print("Extracting dataset...")
        with tarfile.open(file_name) as tar:
            tar.extractall(path=extract_to)
        print("Extraction complete.")

    # Load the data into a DataFrame
    csv_path = os.path.join(extract_to, "CaliforniaHousing/cal_housing.data")
    column_names = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income", "median_house_value",
    ]
    df = pd.read_csv(csv_path, header=None, names=column_names)

    return df

# Example usage
if __name__ == "__main__":
    housing = load_california_housing_data()
    print("PRELIMINARY STATS FROM DATA LOADED:\n")
    print(housing.head())
    print("-------------------------------------------------------------------------------------------")
    print(housing.info())
    print("-------------------------------------------------------------------------------------------")
   # print(housing["housing_median_age"].value_counts())
    print(housing.describe())

    housing.hist(bins=100, figsize=(12,8))
    plt.show()
    plt.show()
    plt.savefig("HousingDataExample.png")

    # We now put the test data aside:
    #shuffle_and_split_data(housing,0.2)

    #housing_with_id = housing.reset_index() # adds an `index` column

    #------------------------Split the dataset using a hash-------------------

    #train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

    #-------------------Split the dataset using the longitude and latitude----

    '''
    This can be used if you dont want to add a new feature to use as index, because once you do it,
    you need to bear in that in mind for the rest of the code.
    
    '''

    #housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    #train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

    #-------------------Split with train_test_split ---------------------------

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    #-------------------Check if the data is skewed ---------------------------

    '''
    From the book: Suppose you’ve chatted with some experts who told you that the median
    income is a very important attribute to predict median housing prices. You
    may want to ensure that the test set is representative of the various categories
    of incomes in the whole dataset. 

    The idea then is to store the data into a distribution with several categories, and see
    if there are some kind of skewness towards values larger than 60000$ for instance.
    This value can be used to do the split.

    The general idea is not to have just a random sampling splitting of the data but rather
    having several splits based on a parameter which either has good properties (not skewed)
    or it is just important for our study
    
    '''

    housing["income_cat"] = pd.cut(housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6.,np.inf],
    labels=[1, 2, 3, 4, 5])

    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.show()
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    plt.show()
    plt.savefig("Income_cat_histogram.png")

    print(len(train_set))
    print(len(test_set))

    #------------------ We further split the data to ensure representative items ----------
    
    '''splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    strat_splits = []
    for train_index, test_index in splitter.split(housing, housing["income_cat"]):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append([strat_train_set_n, strat_test_set_n])

    strat_train_set, strat_test_set = strat_splits[0]'''

    #------------- We further split the data to ensure representative items: Shorter way ----------

    strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

    print("income_cat proportions:\n")
    print(f'{strat_test_set["income_cat"].value_counts() / len(strat_test_set)}')



