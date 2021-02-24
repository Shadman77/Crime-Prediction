import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def __distribution_util(df, col):
    sns.distplot(df[col])
    plt.ticklabel_format(style='plain')
    plt.show()


def distribution(df):
    cols = ["Date", "Updated On", "Beat", "District", "Ward", "Community Area" ,"Perpetrator Age"]
    for col in cols:
        __distribution_util(df, col)


def __count_plot_util(df, col):
    sns.countplot(x=col, data=df)
    plt.show()


def count_plot(df):
    cols = ["Primary Type", "Arrest", "Domestic", "Crime Solved", "Perpetrator Sex", "Perpetrator Race", "Perpetrator Ethnicity", "Weapon"]
    for col in cols:
        __count_plot_util(df, col)

def __count_plot_hue_util(df, col):
    sns.countplot(x=col, data=df, hue="Primary Type")
    plt.show()

def count_plot_hue(df):
    cols = ["Arrest", "Domestic", "Crime Solved", "Perpetrator Sex", "Perpetrator Race", "Perpetrator Ethnicity", "Weapon"]
    for col in cols:
        __count_plot_hue_util(df, col)
