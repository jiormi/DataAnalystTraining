import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

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



