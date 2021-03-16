import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def __distribution_util(df, col):
    sns.distplot(df[col])
    plt.ticklabel_format(style='plain')
    # plt.xticks(rotation = 45)
    plt.show()


def distribution(df):
    cols = ["Date", "Updated On", "Beat", "District",
            "Ward", "Community Area", "Perpetrator Age"]
    for col in cols:
        try:
            __distribution_util(df, col)
        except:
            print("Not plotting ", col)


def __count_plot_util(df, col):
    sns.countplot(x=col, data=df)
    plt.xticks(rotation = 45)
    plt.show()


def count_plot(df):
    cols = ["Primary Type", "Year", "Arrest", "Domestic", "Crime Solved",
            "Perpetrator Sex", "Perpetrator Race", "Perpetrator Ethnicity", "Weapon"]
    for col in cols:
        try:
            __count_plot_util(df, col)
        except:
            print("Not plotting ", col)


def __count_plot_hue_util(df, col):
    sns.countplot(x=col, data=df, hue="Primary Type")
    plt.xticks(rotation = 45)
    plt.show()


def primary_type_name(x):
    if x == False or x == 0:
        return "Non-Severe Crime"
    else:
        return "Severe Crime"

def count_plot_hue(df):
    df['Primary Type'] = df['Primary Type'].apply(primary_type_name)
    cols = ["Year", "Arrest", "Domestic", "Crime Solved", "Perpetrator Sex",
            "Perpetrator Race", "Perpetrator Ethnicity", "Weapon"]
    for col in cols:
        try:
            __count_plot_hue_util(df, col)
        except:
            print("Not plotting ", col)


def count_plot_hue_format(df):
    df['Primary Type'] = df['Primary Type'].apply(primary_type_name)
    __count_plot_util(df, "Primary Type")
    cols = ["Year", "Arrest", "Domestic", "Perpetrator Sex"]
    for col in cols:
        __count_plot_hue_util(df, col)


def __lm_plot_hue_util(df, x, y):
    sns.lmplot(x=x, y=y, hue="Primary Type", data=df)
    plt.show()


def lm_plot_hue(df):
    __lm_plot_hue_util(df, "Beat", "Perpetrator Age")
