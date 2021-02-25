import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def __distribution_util(df, col):
    sns.distplot(df[col])
    plt.ticklabel_format(style='plain')
    plt.show()


def distribution(df):
    cols = ["Date", "Updated On", "Beat", "District",
            "Ward", "Community Area", "Perpetrator Age"]
    for col in cols:
        __distribution_util(df, col)


def __count_plot_util(df, col):
    sns.countplot(x=col, data=df)
    plt.xticks(rotation = 45)
    plt.show()


def count_plot(df):
    cols = ["Primary Type", "Year", "Arrest", "Domestic", "Crime Solved",
            "Perpetrator Sex", "Perpetrator Race", "Perpetrator Ethnicity", "Weapon"]
    for col in cols:
        __count_plot_util(df, col)


def __count_plot_hue_util(df, col):
    sns.countplot(x=col, data=df, hue="Primary Type")
    plt.xticks(rotation = 45)
    plt.show()


def count_plot_hue(df):
    cols = ["Year", "Arrest", "Domestic", "Crime Solved", "Perpetrator Sex",
            "Perpetrator Race", "Perpetrator Ethnicity", "Weapon"]
    for col in cols:
        __count_plot_hue_util(df, col)

def count_plot_hue_format(df):
    cols = ["Year", "Arrest", "Domestic", "Crime Solved", "Perpetrator Sex"]
    for col in cols:
        __count_plot_hue_util(df, col)


def __lm_plot_hue_util(df, x, y):
    sns.lmplot(x=x, y=y, hue="Primary Type", data=df)
    plt.show()


def lm_plot_hue(df):
    __lm_plot_hue_util(df, "Beat", "Perpetrator Age")
