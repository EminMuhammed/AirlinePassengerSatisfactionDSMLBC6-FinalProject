import pandas as pd
from helpers.eda import grab_col_names
from helpers.data_prep import check_outlier, replace_with_thresholds
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)


def get_data():
    return pd.read_csv('./data/train.csv').drop(['Unnamed: 0','id'], axis=1)



def get_corr_matrix(df):
    f, ax = plt.subplots(figsize=[20, 15])
    sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.show()


def age_engineering(df):
    # ERGEN ERKEK - KADIN 0-17
    df.loc[(df["Age"] <= 17) & (df["Gender"] == "Male"),
           "NEW_GENDER_CAT"] = "AdolescentMale"
    df.loc[(df["Age"] <= 17) & (df["Gender"] == "Female"),
           "NEW_GENDER_CAT"] = "AdolescentFemale"
    # GENÇ ERKEK-KADIN 18-65
    df.loc[(df["Age"] >= 18) & (df["Age"] <= 65) & (
        df["Gender"] == "Male"), "NEW_GENDER_CAT"] = "YouthMale"
    df.loc[(df["Age"] >= 18) & (df["Age"] <= 65) & (
        df["Gender"] == "Female"), "NEW_GENDER_CAT"] = "YouthFemale"
    # ORTA YAŞ ERKEK-KADIN 66-79
    df.loc[(df["Age"] >= 66) & (df["Age"] <= 79) & (
        df["Gender"] == "Male"), "NEW_GENDER_CAT"] = "MiddleAgeMale"
    df.loc[(df["Age"] >= 66) & (df["Age"] <= 79) & (
        df["Gender"] == "Female"), "NEW_GENDER_CAT"] = "MiddleAgeFemale"
    # YAŞLI
    df.loc[(df["Age"] > 79) & (df["Gender"] == "Female"),
           "NEW_GENDER_CAT"] = "OldAgeFemale"
    df.loc[(df["Age"] > 79) & (df["Gender"] == "Male"),
           "NEW_GENDER_CAT"] = "OldAgeMale"
    return df


def customer_value_engineering(df):
    df.loc[:, 'New_Value'] = 'Normal'
    df.loc[((df['Customer Type'] == 'Loyal Customer') | (df['Customer Type'] == 'disloyal Customer')) & ((df['Type of Travel'] == 'Business Travel') | (
        df['Type of Travel'] == 'Personal Travel')) & (df['Class'] == 'Business') & (df['satisfaction'] == 'satisfied'), 'New_Value'] = 'Most Valuable'
    df.loc[(df['Customer Type'] == 'Loyal Customer') & ((df['Type of Travel'] == 'Business Travel') | (df['Type of Travel']
                                                                                                       == 'Personal Travel')) & (df['Class'] == 'Eco Plus') & (df['satisfaction'] == 'satisfied'), 'New_Value'] = 'Valuable'
    df.loc[((df['Customer Type'] == 'Loyal Customer') | (df['Customer Type'] == 'disloyal Customer')) & ((df['Type of Travel'] == 'Business Travel') | (
        df['Type of Travel'] == 'Personal Travel')) & (df['Class'] == 'Eco Plus') & (df['satisfaction'] == 'satisfied'), 'New_Value'] = 'Average Valuable'
    df.loc[(df['Departure Delay in Minutes'] > df['Departure Delay in Minutes'].mean()) & (df['Arrival Delay in Minutes'] > df['Arrival Delay in Minutes'].mean()) & (
        df['Flight Distance'] > df['Flight Distance'].mean()) & (df['satisfaction'] == 'satisfied'), 'New_Value'] = 'Best Valuable'
    return df


def asel_mebaysan_preprocess(df, na='drop', fix_outlier=True):
    cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=5)
    num_cols = [col for col in num_cols if col not in "id"]
    #################################
    #### * Missing Values ####
    #################################
    if na == 'drop':
        df = df.dropna()
    elif na == 'group':
        df["Arrival Delay in Minutes"].fillna(df.groupby(['Customer Type', 'Type of Travel', 'Class', 'satisfaction'])[
                                              "Arrival Delay in Minutes"].transform("mean"), inplace=True)
    elif na == 'knn':
        dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
        scaler = MinMaxScaler()
        dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
        imputer = KNNImputer(n_neighbors=5)
        dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
        dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
        df["Arrival Delay in Minutes"] = dff[["Arrival Delay in Minutes"]]

    #################################
    #### * Outliers ####
    #################################
    if fix_outlier:
        for num in num_cols:
            q1 = 0.05
            q3 = 0.95
            if check_outlier(df, num, q1, q3):
                # sns.boxplot(x=f'{num}',data=df, whis=[0.05, 0.95])
                # plt.show()
                # grab_outliers(df,num,True,0.05,0.95)
                # df[num].describe([0.10, 0.20, 0.30, 0.40, 0.50,
                #              0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
                replace_with_thresholds(df, num, q1, q3)
                # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    else:
        pass

    #################################
    #### * Feature Engineering ####
    #################################
    df = age_engineering(df)
    df = customer_value_engineering(df)
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5)
    num_cols = [col for col in num_cols if col not in "id"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df