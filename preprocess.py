import pandas as pd
import datetime
from modules import data_visualization, data_formatting
from imblearn.over_sampling import SMOTE
import sys


def get_df_info(df):
	print(df.head())
	print("\nThe unique columns are")
	print(df.columns)
	print("\nThe data types are")
	print(df.dtypes)
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
	print()
	# Illinois Uniform Crime Reporting code
	print("Number of null values in IUCR is " + str(df['IUCR'].isna().sum()))
	print("Number of unique values in IUCR is " +
			str(len(df['IUCR'].unique())))
	print()
	print("Number of null values in FBI Code is " +
			str(df['FBI Code'].isna().sum()))
	print("Number of unique values in FBI Code is " +
			str(len(df['FBI Code'].unique())))
	print()
	print("Number of null values in Block is " + str(df['Block'].isna().sum()))
	print("Number of unique values in Block is " +
			str(len(df['Block'].unique())))
	print()
	print("Number of null values in Location Description is " +
			str(df['Location Description'].isna().sum()))  # Drop?
	print("Number of unique values in Location Description is " +
			str(len(df['Location Description'].unique())))
	print()
	print("Number of null values in Arrest is " +
			str(df['Arrest'].isna().sum()))
	print("Number of unique values in Arrest is " +
			str(len(df['Arrest'].unique())))
	print()
	print("Number of null values in Domestic is " +
			str(df['Domestic'].isna().sum()))
	print("Number of unique values in Domestic is " +
			str(len(df['Domestic'].unique())))
	print()
	print("Number of null values in Beat is " + str(df['Beat'].isna().sum()))
	print("Number of unique values in Beat is " +
			str(len(df['Beat'].unique())))
	print()
	print("Number of null values in District is " +
			str(df['District'].isna().sum()))  # Drop?
	print("Number of unique values in District is " +
			str(len(df['District'].unique())))
	print()
	print("Number of null values in Ward is " +
			str(df['Ward'].isna().sum()))  # Ward?
	print("Number of unique values in Ward is " +
			str(len(df['Ward'].unique())))
	print()
	print("Number of null values in Community Area is " +
			str(df['Community Area'].isna().sum()))  # Community Area?
	print("Number of unique values in Community Area is " +
			str(len(df['Community Area'].unique())))
	print()
	print("Number of null values in Year is " + str(df['Year'].isna().sum()))
	print("Number of unique values in Year is " +
			str(len(df['Year'].unique())))
	print()
	print("Number of null values in Updated On is " +
			str(df['Updated On'].isna().sum()))
	print("Number of unique values in Updated On is " +
			str(len(df['Updated On'].unique())))
	print()
	print("Number of null values in Latitude is " +
			str(df['Latitude'].isna().sum()))  # Average?
	print("Number of unique values in Latitude is " +
			str(len(df['Latitude'].unique())))
	print()
	print("Number of null values in Longitude is " +
			str(df['Longitude'].isna().sum()))  # Average?
	print("Number of unique values in Longitude is " +
			str(len(df['Longitude'].unique())))
	print()
	print("Number of null values in X Coordinate is " +
			str(df['X Coordinate'].isna().sum()))  # Average?
	print("Number of unique values in X Coordinate is " +
			str(len(df['X Coordinate'].unique())))
	print()
	print("Number of null values in Y Coordinate is " +
			str(df['Y Coordinate'].isna().sum()))  # Average?
	print("Number of unique values in Y Coordinate is " +
			str(len(df['Y Coordinate'].unique())))
	print()
	print("Number of null values in Crime Solved is " +
			str(df['Crime Solved'].isna().sum()))
	print("Number of unique values in Crime Solved is " +
			str(len(df['Crime Solved'].unique())))
	print("Unique values in Crime Solved are " + str(df['Crime Solved'].unique()))
	print()
	print("Number of null values in Perpetrator Sex is " +
			str(df['Perpetrator Sex'].isna().sum()))
	print("Number of unique values in Perpetrator Sex is " +
			str(len(df['Perpetrator Sex'].unique())))
	print("Unique values in Perpetrator Sex are " + str(df['Perpetrator Sex'].unique()))
	print()
	print("Number of null values in Perpetrator Age is " +
			str(df['Perpetrator Age'].isna().sum()))
	print("Number of unique values in Perpetrator Age is " +
			str(len(df['Perpetrator Age'].unique())))
	print("Number of rows where age is 0 is:", len(df[df['Perpetrator Age'] == 0]))
	print()
	print("Number of null values in Perpetrator Race is " +
			str(df['Perpetrator Race'].isna().sum()))
	print("Number of unique values in Perpetrator Race is " +
			str(len(df['Perpetrator Race'].unique())))
	print()
	print("Number of null values in Perpetrator Ethnicity is " +
			str(df['Perpetrator Ethnicity'].isna().sum()))
	print("Number of unique values in Perpetrator Ethnicity is " +
			str(len(df['Perpetrator Ethnicity'].unique())))
	print()
	print("Number of null values in Weapon is " +
			str(df['Weapon'].isna().sum()))
	print("Number of unique values in Weapon is " +
			str(len(df['Weapon'].unique())))
	print()


# get rid of the time and convert date to the format YYYYMMDD
def convert_date_format(x):
	x = x.strip()
	x = x.split(" ")[0]
	x = x.split("/")
	# return datetime.datetime(int(x[2]), int(x[0]), int(x[1]), 0, 0).strftime("%s") 
	return datetime.datetime(int(x[2]), int(x[0]), int(x[1]), 0, 0) / 1000 #strftime("%s") is depreciate


def data_cleaning(df):

    # Drop rows
	df = df.dropna(axis=0, subset=['Location Description'])  # 466
	df = df.dropna(axis=0, subset=['District'])  # 1
	df = df.dropna(axis=0, subset=['Ward'])  # 10
	df = df.dropna(axis=0, subset=['Community Area'])  # 40
	df = df.dropna(axis=0, subset=['Latitude']) #7351
	df = df.dropna(axis=0, subset=['Longitude']) #7351

    # Convert Perpetrator Age to int64 and drop rows containing 0
	df['Perpetrator Age'] = df['Perpetrator Age'].replace(" ", "0")
	df['Perpetrator Age'] = df['Perpetrator Age'].astype('int64')
	df = df[df['Perpetrator Age'] != 0] # 211079

	return df


def group_primary_type(x):
    zero = ['PUBLIC PEACE VIOLATION', 'WEAPONS VIOLATION', 'OTHER OFFENSE', 'DECEPTIVE PRACTICE',
            'CRIMINAL TRESPASS', 'OBSCENITY', 'INTERFERENCE WITH PUBLIC OFFICER', 'GAMBLING', 'INTIMIDATION',
            'LIQUOR LAW VIOLATION', 'NON-CRIMINAL', 'PUBLIC INDECENCY', 'CONCEALED CARRY LICENSE VIOLATION',
            'NON - CRIMINAL', 'NON-CRIMINAL (SUBJECT SPECIFIED)']
    one = ['BATTERY', 'THEFT', 'ROBBERY', 'MOTOR VEHICLE THEFT', 'ASSAULT', 'CRIMINAL DAMAGE', 'BURGLARY', 'STALKING',
           'PROSTITUTION', 'ARSON']
    two = ['CRIM SEXUAL ASSAULT', 'SEX OFFENSE', 'NARCOTICS', 'OTHER NARCOTIC VIOLATION', 'OFFENSE INVOLVING CHILDREN',
           'KIDNAPPING', 'HOMICIDE', 'HUMAN TRAFFICKING']

    if x in zero:
        return 0
    if x in one:
        return 1
    if x in two:
        return 2


if __name__ == "__main__":
	df = pd.read_csv('data/sana_main_file2.csv')
	get_df_info(df)

	# dropping columns
	'''
	Dropping City and State cause they are not accurate
	Dropping Block since the number of block will make it hard to encode
	Dropping 'X Coordinate', 'Y Coordinate', 'Location' since Latitude, Logitude is already given
	FBI Code is dropped to reduce the complexity
	Dropping Description since we already have IUCR
	'''
	df = df.drop(columns=['ID', 'Case Number', 'City', 'State',
							'Block', 'X Coordinate', 'Y Coordinate', 'Location', 'FBI Code', 'Description'])
	print(df.columns)

	# Change date format
	df["Date"] = df["Date"].apply(convert_date_format)
	df["Updated On"] = df["Updated On"].apply(convert_date_format)
	print(df["Date"].head)
	print(df["Updated On"].head)

	# Data cleaning
	print("Number of rows before cleaning", len(df.index))
	df = data_cleaning(df)
	print("Number of rows after cleaning", len(df.index))
	print("Null values after cleaning:")
	print(df.isnull().sum())
	print()

	# Group the target column (Primary Type)
	df['Primary Type'] = df['Primary Type'].apply(group_primary_type)
	print(df.dtypes)

#     Data Visualization
	data_visualization.distribution(df)
#     data_visualization.count_plot(df)
	data_visualization.count_plot_hue(df)
#     data_visualization.lm_plot_hue(df)

    # Dropping date columns since it turns out the year is the only one we need
	df = df.drop(columns=["Date", "Updated On"])
	print("\nThe unique columns after dropping Date & Updated On are")
	print(df.columns)

	# Seperating X, y
	X = df.drop(columns=["Primary Type"])
	y = df["Primary Type"]
	print(X.head())
	print(y.head())

	# Data formatting
	print("Length before data formatting: ", len(X.index))
	X = data_formatting.format_X(X)
	print("After data formatting: ")
	print(X.columns)


	# Applying Smote
	df = X
	df["Primary Type"] = y
	data_visualization.count_plot_hue_format(df)

	print("Length before applying smote: ", len(X.index))
	
	oversample = SMOTE()
	
	X, y = oversample.fit_resample(X, y)
	print("Length after applying smote: ", len(X.index))
	
	df = X
	df["Primary Type"] = y
	data_visualization.count_plot_hue_format(df)

	# Save the dataset
	df.to_csv('data/smote.csv', index=False)

