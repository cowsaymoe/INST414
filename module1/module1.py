"""
Process the data and generates graphs.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main(dataset_path = 'BigmacPrice.csv'):
    """
    Main function to process the data and generate graphs.
    
    Args:
        dataset_path (str): Path to the dataset file.
    """

    dataset = process_data(pd.read_csv(dataset_path))


    # Convert date to datetime
    dataset['date'] = pd.to_datetime(dataset['date'])

    # Plot Big Mac prices over time globally
    plt.figure(figsize=(10,6))
    sns.lineplot(x='date', y='dollar_price', data=dataset, hue='name', legend=False)
    plt.title('Big Mac Prices Over Time (2000-2022)')
    plt.xlabel('Year')
    plt.ylabel('Price in USD')
    plt.show()

    summary_stats = dataset.describe()
    print(summary_stats)


def process_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data to get the average price of Big Mac in each country.
    
    Args:
        dataset (pd.DataFrame): The dataset.
    
    Returns:
        pd.DataFrame: The processed data.
    """
    print(dataset.head())
    
    # Remove missing values
    dataset.dropna(inplace=True)


    # Remove 'Euro Area' from the dataset
    dataset = dataset[dataset['name'] != 'Euro Area']

    # Remove outliers
    outliers = dataset[dataset['dollar_price'] > 20]
    dataset = dataset[dataset['dollar_price'] <= 20]


    # If a country has two entries per year, remove the second entry
    dataset = dataset.drop_duplicates(subset=['name', 'date'], keep='first')

    print(dataset.head())

    return dataset
    


if __name__ == "__main__":
    main()