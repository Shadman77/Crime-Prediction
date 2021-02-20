import pandas as pd


def get_df_info(df):
    print(df.head())
    print("\nThe unique columns are")
    print(df.columns)
    print("\nThe number of rows are")
    print(len(df.index))
    print("\nThe number of unique ids are")
    print(len(df['ID'].unique()))
    print("\nThe number of unique case numbers are")
    print(len(df['Case Number'].unique()))
    print("The number of null values in the column case numbers are")
    print(df['Case Number'].isna().sum())
    # Since there are no missing values for case number and the number of unique case number
    # is slightly lower than the number of rows,
    # this means from some cases there are multiple entries

    print("\nThe unique IUCR(Illinois Uniform Crime Reporting code) are")
    print(df['IUCR'].unique())
    print("The number of unique IUCR")
    print(len(df['IUCR'].unique()))
    print("The number of missing values in this column is IUCR")
    print(df['IUCR'].isna().sum())


if __name__ == "__main__":
    df = pd.read_csv('data/sana_main_file2.csv')
    get_df_info(df)
