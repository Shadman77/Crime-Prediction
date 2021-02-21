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

    # print("\nThe unique IUCR(Illinois Uniform Crime Reporting code) are")
    # print(df['IUCR'].unique())
    print("The number of unique IUCR")
    print(len(df['IUCR'].unique()))
    print("The number of missing values in this column is IUCR")
    print(df['IUCR'].isna().sum())

    print("\nThe number of unique Primary Type are:")
    print(len(df["Primary Type"].unique()))
    print("The unique values are: ")
    print(df["Primary Type"].unique())
    print("The number of missing values are")
    print(df['Primary Type'].isna().sum())

    print("\nThe number of unique Description are:")
    print(len(df["Description"].unique()))
    # print("The unique values are: ")
    # print(df["Description"].unique())
    # Too many
    print("The number of missing values are")
    print(df['Description'].isna().sum())

    print("\nNumber of null values in Date is " + str(df['Date'].isna().sum()))
    print("Number of null values in Block is " + str(df['Block'].isna().sum()))
    print("Number of null values in Location Description is " + str(df['Location Description'].isna().sum())) #Drop?
    print("Number of unique values in Location Description is " + str(len(df['Location Description'].unique())))
    print("Number of null values in Arrest is " + str(df['Arrest'].isna().sum()))
    print("Number of null values in Domestic is " + str(df['Domestic'].isna().sum()))
    print("Number of null values in Beat is " + str(df['Beat'].isna().sum()))
    print("Number of null values in District is " + str(df['District'].isna().sum())) #Drop?
    print("Number of null values in Ward is " + str(df['Ward'].isna().sum())) #Ward?
    print("Number of null values in Community Area is " + str(df['Community Area'].isna().sum())) #Community Area?
    print("Number of null values in Year is " + str(df['Year'].isna().sum()))
    print("Number of null values in Updated On is " + str(df['Updated On'].isna().sum()))
    print("Number of null values in Latitude is " + str(df['Latitude'].isna().sum())) #Average?
    print("Number of null values in Longitude is " + str(df['Longitude'].isna().sum())) #Average?
    print("Number of null values in Crime Solved is " + str(df['Crime Solved'].isna().sum()))
    print("Number of null values in Perpetrator Sex is " + str(df['Perpetrator Sex'].isna().sum()))
    print("Number of null values in Perpetrator Age is " + str(df['Perpetrator Age'].isna().sum()))
    print("Number of null values in Perpetrator Race is " + str(df['Perpetrator Race'].isna().sum()))
    print("Number of null values in Perpetrator Ethnicity is " + str(df['Perpetrator Ethnicity'].isna().sum()))
    print("Number of null values in Weapon is " + str(df['Weapon'].isna().sum()))


if __name__ == "__main__":
    df = pd.read_csv('data/sana_main_file2.csv')
    get_df_info(df)
