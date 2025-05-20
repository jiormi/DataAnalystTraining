import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from zlib import crc32
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel



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

    #-------------------Split with train_test_split ---------------------------

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    #print(len(train_set))
    #print(len(test_set))

    #------------------ We further split the data to ensure representative items (by hand) ----------
    
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

    #print("income_cat proportions:\n")
    #print(f'{strat_test_set["income_cat"].value_counts() / len(strat_test_set)}')


    # I will quickly check here the content of the strat splits
    '''
    I check here the content of the different splits . This is useful also as a cross-validation of ML models. To be seen later on.
    
    '''
    '''
    for i, strsplit in enumerate(strat_splits):
        print(f'Split {i}: Training dataset:\n')
        print(strsplit[0].head())
        print(f'Split {i}: Test dataset:\n')
        print(strsplit[1].head())

    '''
    '''

    We can even go further and follow the book suggestion, we can try to check whether this worked out or not.
    We can pick the first item from the strat_splits object, actually the first split, and compare the proportions
    based on the income category for a random sampling and for the stratified sampling. This can be built up such as;
    
    '''

    #strat_train_set, strat_test_set = strat_splits[0]

    #------------- Comparison stratified vs Random sampling ----------

    '''
    overall = housing['income_cat'].value_counts() / len(housing)
    stratified = strat_test_set['income_cat'].value_counts() / len(strat_test_set)
    random = test_set['income_cat'].value_counts() / len(test_set)
    err_stratified = (overall - stratified) / overall
    err_random = (overall - random) / overall

    summary_table = pd.DataFrame({
    'Overall %': overall,
    'Stratified %': stratified,
    'Random %': random,
    'Error Strat %': err_stratified,
    'Error Random %': err_random
    })
    '''

    '''
    We should observe that the average error in the random sampling is much higher than the stratified one

    '''


    #print('Comparison in bias for the random sampling and the stratified sampling for the first split:\n')
    #print(f'{summary_table}')

 
    '''
    At this stage, we now have overlooked the data, so it is time to get into the trainning process in a deeper way.
    Let's first make a copy of our original dataset, just in case we need it later.
    '''

    housing = strat_train_set.copy()

    # Geographical scatter plot

    plt.figure() 
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
    plt.show()
    plt.savefig('ScatterGeo.png')

    # Geographical scatter fancy plot
    
    plt.figure() 
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
    s=housing["population"] / 100, label="population",
    c="median_house_value", cmap="jet", colorbar=True,
    legend=True, sharex=False, figsize=(10, 7))
    plt.show()
    plt.savefig('ScatterGeoFancyColor.png')

    
    # Pearson correlation by each pair of variables

    corr_matrix = housing.corr()

    # Correlation of the variables with the Median House Value
    
    corr_with_Median_House_Value = corr_matrix["median_house_value"].sort_values(ascending=False)
    print(f'{corr_with_Median_House_Value}')

    plt.figure()
    attributes = ["median_house_value", "median_income", "total_rooms",
    "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()
    plt.savefig('ScatterMatrixPlots.png')

    '''
    There are some attributes that can be wither meaningless or super important if we carefully think about it.
    We can build up several variables in that way:

    '''

    housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["people_per_house"] = housing["population"] / housing["households"]

    print('Correlation matrix after adding some new features')

    corr_matrix = housing.corr()
    corr_with_Median_House_Value_NewInputs = corr_matrix["median_house_value"].sort_values(ascending=False)

    print(corr_with_Median_House_Value_NewInputs)

    plt.figure()
    attributes = ["median_house_value", "median_income", "rooms_per_house",
    "bedrooms_ratio","people_per_house"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()
    plt.savefig('ScatterMatrixPlotsNewInputs.png')



    # ============================== Until here is the data investigation. Now we start with the data preparation =================

    # ============= Numerical attributes


    '''
    Since our final aim is to predict the median house value, we ware going to separate the target value and the labels
    '''

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # Clean the data. All nonvalues or zero values will be changed by the median of that assemble

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.select_dtypes(include=[np.number]) # The imputer only works with numerical values. Those columns with non numerical values are excluded
    imputer.fit(housing_num)

    print('Check of the output of the imputer. Just the median of each attribute. They both should be equal')
    print(imputer.statistics_)
    print(housing_num.median().values)

    # Fill the data exchanging missing values with the mean

    X = imputer.transform(housing_num)

    #print(X)

    # Remember that the transform method from the imputer package does not retrieve a DataFrame, but Numpy vectors or Scikit-Learn sparse matrices

    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
    index=housing_num.index)

    #print(housing_tr)

    # ===================== Scaling and Standarization ====================

    # Scaling: Basically is just scale the data within a range you can choose

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

    # Standarization: It is basically an standarization as always, you just substract the mean value and divide by the standard deviation
    # Good thing: This way of manipulating your data will have the advantage of being less affected by extreme values

    std_scaler = StandardScaler()
    housing_num_std_scaled = std_scaler.fit_transform(housing_num)


    # Deal with heavy tails. Several options arise here. This means that the extreme values are kinda likelly

    # Power low heavy tail --> use logarigthm. Example:

    plt.figure()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Histogram of raw population
    axs[0].hist(housing['population'], bins=100)
    axs[0].set_xlabel('Population')
    axs[0].set_ylabel('Number of districts')

    # Right: Histogram of log-transformed population
    axs[1].hist(np.log(housing['population']), bins=100)
    axs[1].set_xlabel('Log of population')

    plt.show()
    plt.savefig('PopulationHistogramLog.png')

    # bucketizing --> use Gaussian RBF. Example:
    

    age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Histogram (left axis)
    counts, bins, patches = ax1.hist(housing['housing_median_age'], bins=100, color='royalblue', alpha=0.8)
    ax1.set_xlabel('Housing median age')
    ax1.set_ylabel('Number of districts', color='black')

    # Create second axis for age similarity
    ax2 = ax1.twinx()
    ax2.plot(age_simil_35, color='blue', linewidth=2, label='gamma = 0.10')
    ax2.plot(age_simil_35, color='blue', linestyle='--', linewidth=2, label='gamma = 0.03')
    ax2.set_ylabel('Age similarity', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Add legend
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig('HousingMedianAverageBucketing.png')
    




    








    


