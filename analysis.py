import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, confusion_matrix, r2_score
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import graphviz
import jenkspy

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'EPV_code')
DATADIR = os.path.join(DIRNAME, 'data')


def concat_dataframes(main_df, df):
    """
    appends new dataframe to main data frame, by checking whether the dataframe is empty

    Parameters
    -----------
    main_df: main DataFrame
    df: the DataFrame that needs to be concatenated to the main DataFrame

    Returns
    -----------
    main_df: main DataFrame with appended df
    """
    if isinstance(main_df, pd.DataFrame):
        main_df = pd.concat([main_df, df], axis=0)
    else:  # first iteration
        main_df = df
    return main_df


def load_epv_dataframes_and_merge():
    """
    creates paths and reads EPV DataFrames and merges them into one DataFrame

    Returns
    -----------
        merged_epv: merged DataFrame of all EPV csv files
    """
    directory = os.fsencode(DATADIR + r'\xml_files')
    merged_epv = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        base_file = filename.split('.xml')[0]
        epv_df_path = f'{DATADIR} \\EPV_features\\{base_file}_EPV_features_NEW.csv'
        temp_df = pd.read_csv(epv_df_path, index_col=0)
        temp_df['MatchId'] = base_file
        merged_epv = concat_dataframes(merged_epv, temp_df)
    merged_epv = merged_epv.reset_index()
    return merged_epv


def change_degrees_based_on_playing_direction(events, column_name):
    """
    Changes the degrees based on playing direction. Degrees are formed from -180 to 180.

    Parameters
    -----------
        events: events DataFrame
        column_name: name of column that needs to be changed
    Returns
    -----------
        events: events DataFrame
    """
    mask_pos_pass = ((events.playing_direction == -1) & (events[column_name] > 0))
    mask_neg_pass = ((events.playing_direction == -1) & (events[column_name] < 0))
    events.loc[mask_pos_pass, column_name] -= 180
    events.loc[mask_neg_pass, column_name] += 180
    return events


def convert_to_single_direction(events):
    """
    Converts positional data and columns derived from positional data to a single playing direction.
    This to ensure that features hold the highest descriptive value.
    All teams already play, from either left to right for the whole match or vice versa.
    So the only thing we have to do , is flip all right -> left playing teams in the dataset.

    Parameters
    -----------
        events: events DataFrame
    Returns
    -----------
        events: events DataFrame with changed columns
    """

    columns_positional = ['Start X', 'Start Y', 'End X', 'End Y', 'EPV_target_x', 'EPV_target_y']
    # select only home events to flip selected columns and orientation of passing player
    events.loc[events.playing_direction == -1, columns_positional] *= -1

    # change degrees of three columns
    events = change_degrees_based_on_playing_direction(events, 'pass_direction_degrees_optimal')
    events = change_degrees_based_on_playing_direction(events, 'player_orientation')
    return events


def get_features_and_target_df(df, target_column_name):
    """
    select the target and feature columns that are needed as our start point. Dropping all non-features.

    Parameters
    -----------
        df: DataFrame consisting of all columns, including unneeded columns
        target_column_name: column name of the target variable
    Returns
    -----------
    features: features DataFrame
    target: target Series (max_EEPV_added: the added value of the optimal passes)
    """
    columns = ['Time [s]', 'End Time [s]', 'Start X', 'Start Y', 'BodyPart', 'time_after_transition', 'backwards_pass',
               'forward_pass', 'sideways_pass', 'pass_direction_degrees_optimal', 'pass_length_optimal',
               'diagonal_pass_optimal', 'player_speed', 'player_distance_covered_5sec', 'player_orientation',
               'dist_def_line_own', 'dist_def_line_opponent', 'length_playing_field', 'num_defenders_closer_to_goal',
               'def_team_length', 'att_team_length', 'speed_attackers', 'speed_defenders', 'speed_average',
               'average_distance_defenders', 'distance_closest_defender', 'num_defenders_within_5m',
               'num_defenders_within_15m', 'average_distance_attackers', 'distance_closest_attacker',
               'num_attackers_within_5m', 'num_attackers_within_15m', 'pitch_control_percentage', 'length_x_direction',
               'length_y_direction'
               ]
    features = df[columns]
    target = df[target_column_name]
    return features, target


def outlier_detection(df):
    """
    detects outliers by using a 3 * standard deviation check (Chauvenet's criterion). \
    If the number of outliers is not equal to the number of NaN values, we make a plot, for visual inspection.

    Parameters
    -----------
        df: DataFrame

    # Rationale behind outlier handling:

    'End X': falls within the logical bounds of the field dimension
     'time_after_transition': falls within the logical bounds
     'pass_length': falls within the possible bounds of 90 meters
     'player_speed', 'speed_attackers', speed_defenders': show outlier values of 5 to 6 m/s,
            this is equivalent to approximately 18-22 km/h. Where it is unlikely that all players on the
            pitch are on average running at a pace of  18-22 km/h, it is practically possible.
            And imputing these will probably lead to more knowledge loss than gain.
     'player_distance_covered_5sec': falls within the logical bounds (maximum of 55 meters)
     'player_orientation': falls within the logical bounds
     'dist_def_line_own': falls within the logical bounds
     'dist_def_line_opponent': falls within the logical bounds
     'length_playing_field': falls within the logical bounds
     'num_defenders_closer_to_goal': 0 is unlogical, upon further investigation, there are 5 where the
             speed and distance features are not calculated. Will delete all 5 rows from the DataFrame
     deleting 5 rows, since for all these rows the speed and distance features are not calculated
        # Found by visual inspection of the boxplot for 'num_defenders_closer_to_goal', since this value was 0
     'def_team_length': falls within the logical bounds (max 90m)
     'att_team_length': falls within the logical bounds (max 90m)
     'speed_average': falls within the logical bounds (max 6 m/s)
     'average_distance_defenders': has 5 anomalies with a value of zero, see player_speed for explanation
     'average_distance_attackers': has 5 anomalies with a value of zero, see player_speed for explanation
     'num_defenders_within_5m': falls within the logical bounds
     'num_defenders_within_15m': falls within the logical bounds
     'num_attackers_within_5m': falls within the logical bounds
     'num_attackers_within_15m': falls within the logical bounds
     'distance_closest_defender': shows a maximum of 30m. This is theoretically possible, if for instance there is a
            clearance after a corner.
     'distance_closest_attacker': falls within the logical bounds (shows a maximum of 24m)
     'pitch_control_percentage': falls within the logical bounds of 0 - 100 %
     'EEPV_added': shows a max EEPV added of 0.07, which is still logical
    """
    warnings.filterwarnings("ignore")
    # get the number of outliers
    for i, col in enumerate(df.columns):
        column_name = col
        number_of_outliers = len(df[(np.abs(df[column_name] - df[column_name].mean()) >= 3 * df[column_name].std()) &
                                    (df[column_name].notnull())])
        # plot the variables with outliers
        if df[column_name].isna().sum() != number_of_outliers:
            plt.figure(i)
            sns.boxplot(x=df[column_name])


def knn_imputation(features):
    """
    imputes the NaN values of the feature columns using k-Nearest Neighbours

    Parameters
    -----------
        features: features DataFrame
    Returns
    -----------
        features: features DataFrame
    """
    cols = X.columns
    # define imputer
    imputer = KNNImputer()
    # fit on the dataset
    imputer.fit(features)
    # transform the dataset
    features_trans = imputer.transform(features)
    # imputed features
    features = pd.DataFrame(data=features_trans, columns=cols)
    return features


def data_cleaning(X, y, target_column_name, train_or_test):
    """
    Cleans the features and target data, removes columns with to many NaN, also deletes 5 rows, that were found to be
    incorrect via visual inspection of Figures from outlier_detection()
    # Origin seems to be that gk_numbers were not found in feature engineering.

    Parameters
    -----------
        X: features DataFrame
        y: target variable Series
        target_column_name: column name of the target variable
        train_or_test: string variable, if 'test' we make a change
    Returns
    -----------
        X: features DataFrame
    """
    y = np.sqrt(y)
    # taking the square root, results in 1 nan value for the test set. We delete this row from both the y
    if train_or_test == 'test':
        # training and test data:
        idx = y[y.apply(np.isnan)].index[0]
        y = y[y.index != idx]
        X = X[X.index != idx]

    # combine features and target, so we can delete the same rows in both the features and target variables
    df = pd.concat([X, y], axis=1)

    # deleting 5 rows, since for all these rows the speed and distance features are not calculated
    # Found by visual inspection of the boxplot for 'num_defenders_closer_to_goal', since this value was 0
    # delete incorrect speed distances rows
    df = df[df['num_defenders_closer_to_goal'] != 0]
    X = df.loc[:, df.columns != target_column_name]

    # Imputate the NaN values using KNN imputation
    X = knn_imputation(X)
    y = df[target_column_name]
    return X, y


def plot_normal_distribution(data, skewness, title):
    """
    plots the normal distribution of a given dataset and adds the amount of skew to the title.

    Parameters
    -----------
       data: the target variable data (y_train)
       skewness: value for the amount of skewness in the data
       title: short description of the data, that should be represented in the title
    Returns
    -----------
        X: features DataFrame
    """
    fig, ax = plt.subplots()
    ax.hist(data, edgecolor='white', bins=50)
    plt.gca().set(title='Target distribution (' + title + '), skewness = ' + str(round(skewness, 2)),
                  ylabel='Frequency')
    plt.show()


# Distribution target value:
def plot_skewness_distributions(y_train):
    """
    plots three distributions, of which the first is the original dataset.
    The second is the logarithmic corrected dataset and the third the square root corrected dataset
    Results show that square root corrected is the best choice for correction

    Parameters
    -----------
        y_train: training split of target variable
    """

    # normal target value
    skewness = y_train.skew()
    plot_normal_distribution(y_train, skewness, 'original')

    # logarithmic correction
    y_train_log = np.log(y_train)
    skewness = y_train_log.skew()
    plot_normal_distribution(y_train_log, skewness, 'log corrected')

    # sqrt transform:
    y_train_sqrt = np.sqrt(y_train)
    skewness = y_train_sqrt.skew()
    plot_normal_distribution(y_train_sqrt, skewness, 'sqrt corrected')


def Jenks_algorithm_to_bin_in_3_classes(y_train, y_test):
    """
    Implements the Jenks algorithm to cluster the continuous target variable.
    This algorithm allows us to find natural breaks in a 1D-array.
    Three classes are selected, aiming to represent a low-value pass,
    a medium-value pass and a high-value pass.

    Parameters
    -----------
       y_train: the continuous target array of the training set
       y_test: the continuous target array of the test set
    Returns
    -----------
       y_binned: the target array of the training set (with 3 distinct classes)
    """
    breaks = jenkspy.jenks_breaks(y_train, nb_class=3)
    y_train_binned = pd.cut(y_train,
                            bins=breaks,
                            labels=[0, 1, 2],
                            include_lowest=True)

    y_test_binned = y_test.copy()
    y_test_binned.loc[y_test < breaks[1]] = 0
    y_test_binned.loc[(y_test > breaks[1]) & (y_test < breaks[2])] = 1
    y_test_binned.loc[y_test > breaks[2]] = 2
    return y_train_binned, y_test_binned


def forward_feature_selection_linear_regression(X_train, y_train):
    """
    Selects features using Feedforward Feature Selection using a Linear Regression.
    -- RATIONALE  I had aimed to write my own function to let the number of features to select be variable, however
    due to time constraints I did not implement such a version. For now I selected the number of features (9), based on
    visual inspection of the Forward Feature Selection plots. --
    Parameters
    -----------
    Returns
    -----------
    """
    regr = LinearRegression()
    # Build step forward feature selection
    sfs = SequentialFeatureSelector(regr,
                                    k_features=9,
                                    forward=True,
                                    floating=False,
                                    verbose=2,
                                    scoring='r2',
                                    cv=5)

    # Perform Sequential Feedforward Selection
    sfs = sfs.fit(X_train, y_train)
    selected_feature_names = sfs.k_feature_names_
    return selected_feature_names


def forward_feature_selection_decision_tree(X_train, y_train_binned):
    """
    Selects features using Feedforward Feature Selection using a Decision Tree Classifier.
    -- RATIONALE  I had aimed to write my own function to let the number of features to select be variable, however
    due to time constraints I did not implement such a version. For now I selected the number of features (7), based on
    visual inspection of the Forward Feature Selection plots. --
    Parameters
    -----------
    X_train: training split of feature variables with continuous values
    y_train_binned: training split of feature variables with 3 class values
    Returns
    -----------
    """
    clf = tree.DecisionTreeClassifier()
    # Build step forward feature selection
    sfs = SequentialFeatureSelector(clf,
                                    k_features=7,
                                    forward=True,
                                    floating=False,
                                    verbose=2,
                                    scoring='r2',
                                    cv=5)

    # Perform Sequential Feature Selection
    sfs = sfs.fit(X_train, y_train_binned)
    selected_feature_names = sfs.k_feature_names_
    return selected_feature_names


def forward_feature_selection(X_train, y_train, X_test, y_train_binned):
    """
    Selects features for both a Linear Regression model and a Decision Tree Classifier.
    Uses Sequential Feedforward Feature Selection.

    Parameters
    -----------
    X_train: training split of feature variables
    X_test: test split of feature variables
    y_train_binned: training split of target variable that are binned in three classes
    Returns
    -----------
    X_train_fs_regr: training split of selected feature variables for Regression
    X_test_fs_regr: test split of selected target variables for Regression
    X_train_fs_clf: training split of selected feature variables for Classification
    X_test_fs_clf: test split of selected target variables for Classification
    """
    selected_feature_names = forward_feature_selection_linear_regression(X_train, y_train)
    # create X_train with selected features for Linear Regression
    X_train_fs_regr = X_train[list(selected_feature_names)]
    X_test_fs_regr = X_test[list(selected_feature_names)]
    selected_feature_names = forward_feature_selection_decision_tree(X_train, y_train_binned)
    # create X_train with selected features for Linear Regression
    X_train_fs_clf = X_train[list(selected_feature_names)]
    X_test_fs_clf = X_test[list(selected_feature_names)]

    return X_train_fs_regr, X_test_fs_regr, X_train_fs_clf, X_test_fs_clf


def plot_feed_forward_models():
    """
    Plots the performance for each iteration of the feedforward model.
    The number of features chosen are 15 and 20, since these showed the best result

    """
    # create Linear Regression model
    regr = LinearRegression()

    sfs_model = SequentialFeatureSelector(regr,
                                          k_features=15,
                                          forward=True,
                                          floating=False,
                                          scoring='neg_mean_squared_error',
                                          cv=10)

    sfs_model = sfs_model.fit(X_train, y_train)
    plot_sfs(sfs_model.get_metric_dict(), kind='std_err')
    plt.title('Sequential Forward Selection Linear Regression (w. StdErr)')
    plt.grid()
    plt.show()

    # Same for the Decision Tree, with some different settings
    clf = tree.DecisionTreeClassifier()

    sfs_model = SequentialFeatureSelector(clf,
                                          k_features=20,
                                          forward=True,
                                          floating=False,
                                          scoring='accuracy',
                                          cv=10)
    sfs_model = sfs_model.fit(X_train, y_train_binned)
    plot_sfs(sfs_model.get_metric_dict(), kind='std_err')
    plt.title('Sequential Forward Selection Decision Tree (w. StdErr)')
    plt.grid()
    plt.show()


def feature_adaptations(epv_df):
    """
    Makes feature adaptations, so adds the length x direction and y direction, next to the start and start positions.
    ## Forgot to add these in feature engineering.
    Additionally converts tracking based features data to conform a one directional playing. To enable consistency
    in the features.

    Parameters
    -----------
       epv_df: the EPV DataFrame with events data
    Returns
    -----------
       epv_df: the EPV DataFrame with events data
    """

    for i, row in epv_df.iterrows():
        epv_df.loc[i, 'length_x_direction'] = epv_df.loc[i, 'EPV_target_x'] - epv_df.loc[i, 'Start X']
        epv_df.loc[i, 'length_y_direction'] = epv_df.loc[i, 'EPV_target_y'] - epv_df.loc[i, 'Start Y']
    epv_df = convert_to_single_direction(epv_df)
    return epv_df


def validation_results_of_Linear_Regression(X_train, X_test, y_train, y_test, X_train_fs_regr, X_test_fs_regr):
    """
    Shows general Regression Analysis validation results: MSE, MAE, RMSE, R2

    Parameters
    -----------
       X_train: training split of feature variables
       X_test: test split of feature variables
       y_train: training split of target variable
       y_test: test split of target variables
       X_train_fs_regr: training split of feature variable with selected features
       X_test_fs_regr: test split of feature variable with selected features
    """

    print('LR, Benchmark:')
    # create benchmark predictions:
    yhat = pd.Series(([0.118] * len(y_test)))
    show_regression_results(y_test, yhat)

    print('\n' + 'LR, without Feature Selection:')
    m = LinearRegression()
    m.fit(X_train, y_train)
    yhat = m.predict(X_test)

    show_regression_results(y_test, yhat)
    print('\n' + 'LR, using Feedforward Feature Selection:')
    m = LinearRegression()
    m.fit(X_train_fs_regr, y_train)
    yhat = m.predict(X_test_fs_regr)
    # evaluate predictions
    show_regression_results(y_test, yhat)


def show_regression_results(y, yhat):
    """
    Shows general Regression Analysis validation results:
       MSE: Mean Squared Error
       MAE: Mean Absolute Error
       RMSE: Root Mean Square Error
       R2

    Parameters
    -----------
       y: test split of the target variables
       yhat: predicted target variables
    """
    d = y - yhat
    mse = np.mean(d ** 2)
    mae = np.mean(abs(d))
    rmse = np.sqrt(mse)
    r2 = r2_score(y, yhat)

    print("Evaluation:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R-Squared:", r2)


def show_decision_tree_results(y, yhat):
    """
    Shows the evaluation results of a single Decision Tree Classifier.
    Shows the accuracy, f1-score and a normalised  confusion matrix

    Parameters
    -----------
       y: test split of the target variables
       yhat: predicted target variables
    """
    y_test_data = np.array(y).reshape(-1, 1)
    temp = np.array(yhat).reshape(-1, 1)
    # y_test - temp
    # print(y_test)
    accuracy = sklearn.metrics.accuracy_score(y_test_data, temp)  # yhat)
    f1_score = sklearn.metrics.f1_score(y_test_data, yhat, average='weighted')  # , labels=[0, 1, 2]

    print("Evaluation:")
    print("accuracy:", accuracy)
    print("f1_score:", f1_score)

    # show confusion matrix:
    cm = confusion_matrix(y, yhat, normalize='true', )
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()


def validation_results_of_Decision_Tree(X_train, X_test, y_train_binned, y_test_binned, X_train_fs_clf, X_test_fs_clf):
    """
    Outputs the validation results (accuracy, f1-score and confusion matrix ) of four models.
    - Decision tree benchmark (with only class 1 prediction (since this is the most prevalent class))
    - Decision tree with all features
    - Decision tree with selected features
    - Random Forest Classifier for comparison

    Parameters
    -----------
       X_train: training split of feature variables
       X_test: test split of feature variables
       y_train_binned: training split of target variable that are binned in three classes
       y_test_binned: test split of target variable that are binned in three classes
       X_train_fs_clf: training split of feature variable that are binned in three classes with selected features
       X_test_fs_clf: test split of feature variable that are binned in three classes with selected features
    """
    # Baseline:
    yhat_benchmark = pd.Series(([1] * len(y_test)))
    show_decision_tree_results(y_test_binned.values, yhat_benchmark)

    # All features:
    clf_all_features = tree.DecisionTreeClassifier(criterion='gini')
    clf_all_features = clf_all_features.fit(np.array(X_train.values), y_train_binned.values)
    yhat_all_features = clf_all_features.predict(np.array(X_test.values))
    show_decision_tree_results(y_test_binned.values, yhat_all_features)

    # Selected features:
    clf_fs = tree.DecisionTreeClassifier(criterion='gini')
    clf_fs = clf_fs.fit(np.array(X_train_fs_clf.values), y_train_binned.values)
    yhat_fs = clf_fs.predict(np.array(X_test_fs_clf.values))
    show_decision_tree_results(y_test_binned.values, yhat_fs)

    # Additionally, I also included a non tuned RandomForestClassifier to compare results.
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=800, random_state=0, criterion='gini')
    rf = rf.fit(np.array(X_train.values), y_train_binned.values)
    yhat_rf = rf.predict(np.array(X_test.values))
    # function to make an image of the decision tree
    show_decision_tree_results(y_test_binned.values, yhat_rf)


def make_decision_tree_plot(X_train, y_train_binned, X_test):
    """
    Makes a .dot file of the Decision Tree and saves it to a folder.

    Parameters
    -----------
       X_train: training split of feature variables
       y_train_binned: training split of target variable with 3 class bins
       X_test: test split of the feature variables
    """
    # create model with max_depth of 4 for clarity of the first 4 layers
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(X_train, y_train_binned)
    clf.predict(X_test)

    dot_data = tree.export_graphviz(clf, out_file=None)

    folder = DIRNAME + '\\Figures\\'
    base_file = 'decision_tree'
    dot_filename = base_file + '_max_depth_4.dot'
    dot_file_path = folder + dot_filename
    tree.export_graphviz(clf, out_file=dot_file_path,
                         feature_names=X_train.columns,
                         class_names='test',
                         filled=True, rounded=True,
                         special_characters=True)
    graphviz.Source(dot_data)


def plot_feature_importances_linear_regression(regr, X_train, y_train):
    """
    plots the feature importances of the 10 highest scoring features based on a Linear Regression model

    Parameters
    -----------
       regr: Linear Regression model
       X_train: training split of feature variables
       y_train: training split of target variable
    """
    regr.fit(X_train, y_train)
    importance = regr.coef_
    # select 10 highest scoring features
    indices = importance.argsort()[-10:][::-1][:10]
    # plot the feature importances of the forest
    fig, ax = plt.subplots()
    plt.title("Feature importances")
    plt.barh(range(10), importance[indices], align="center")
    plt.yticks(range(X_train.shape[1]), X_train.columns[indices])
    plt.ylim([-1, 10])
    plt.title('Feature importances Linear Regression')
    plt.show()


def plot_feature_importances_decision_tree(clf, X, y):  # simple regression analysis
    """ 
    plots the feature importances of the 10 highest scoring features based on a Decision Tree Classifier model

    Parameters
    -----------
       X: features
       y: target
       clf: Decision Tree Classifier model
    Returns
    -----------
       fi: DataFrame with the feature importances
    """

    clf.fit(X, y)
    # plot the 10 highest feature importances using a barplot
    fi = pd.DataFrame({'cols': X.columns, 'imp': clf.feature_importances_}).sort_values(
        'imp', ascending=False)
    fi[:10].plot('cols', 'imp', 'barh', figsize=(12, 7), legend=False
                 ).set_title(label='Relative Feature Importances Decision Tree', size=20)
    plt.yticks(np.arange(10), X.columns)
    plt.show()
    return fi


def create_figures(X_train, y_train, y_train_binned, regr, clf):
    """
    creates multiple figures used for the analysis:
    - pairplot () # Which should be adapted to which features you want to highlight
    - correlation heatmap
    - Skewness plots
    - Feature importances using Linear Regression
    - Feature importances using Decision Tree Classifier

    Parameters
    -----------
        X_train: training split of feature variables
        y_train: training split of target variable
        y_train_binned: binned training split of the target variable
        regr: Linear Regression model
        clf: Decision Tree Classifier model
    """
    sns.pairplot(X_train[['pitch_control_percentage', 'Start X', 'pass_length_optimal']])
    plot_skewness_distributions(y_train)
    plot_feature_importances_linear_regression(regr, X_train, y_train)
    plot_feature_importances_decision_tree(clf, X_train, y_train_binned)
    make_decision_tree_plot(clf, X_train, y_train_binned)


def run_analysis():
    """
    Runs analysis of the transition passes dataframes.
    Firstly, makes some feature adaptations, followed by selecting and splitting the data. Encoding, outlier detection,
    distribution correction and data cleaning is performed. Continuous target variable is binned into 3 classes, for
    classification. Features are selected using feedforward models. And finally Linear and Decision Tree Classification
    is performed, and corresponding validation scores are presented.

    Additionally, a create figures function is called to show the figures that are were used in the analysis process
    """
    epv_df = load_epv_dataframes_and_merge()
    target_column_name = 'max_EEPV_added'
    df = feature_adaptations(epv_df)

    X, y = get_features_and_target_df(epv_df, target_column_name)

    # One hot encoding of the features. Body part features
    X = pd.get_dummies(X)
    # delete the following just created columns, since they only have NaN values
    X = X.drop(columns=['BodyPart_Chest', 'BodyPart_Hands (dropped)', 'BodyPart_Header', 'BodyPart_Left foot'])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    # perform visual inspection on box plots, to adapt strategy in data cleaning process
    outlier_detection(X_train)
    # After data inspection of the target value, we see few relative high values,and a high amount of low values around
    # zero. For this reason, I suspected that that there is a skewness in the data.
    # Thus we plot the distribution and perform square root and logarithmic adaptation of the target variable
    plot_skewness_distributions(y_train)
    # This leads to the conclusion that we should apply a square root adaptation of the target variable,
    # in data_cleaning()
    # Clean data for both the train and test set
    X_train, y_train = data_cleaning(X_train, y_train, target_column_name, 'train')
    X_test, y_test = data_cleaning(X_test, y_test, target_column_name, 'test')

    # change continuous target to class target using binning.
    # We create 3 classes, imitating a low value pass, medium value pass and high value pass
    y_train_binned, y_test_binned = Jenks_algorithm_to_bin_in_3_classes(y_train, y_test)

    # For both models: plots the feedforward model performance for each number of features it selects
    plot_feed_forward_models()
    # for now used a discrete feedforward model, instead of building one that selects number of features based
    # on a a threshold, e.g. linked to the significance of the explained variance of the new variable added.
    X_train_fs_regr, X_test_fs_regr, X_train_fs_clf, X_test_fs_clf = forward_feature_selection(X_train, y_train,
                                                                                               X_test, y_train_binned)
    validation_results_of_Linear_Regression(X_train, X_test, y_train, y_test, X_train_fs_regr, X_test_fs_regr)

    validation_results_of_Decision_Tree(X_train, X_test, y_train, y_test, X_train_fs_regr, X_test_fs_regr)

    # create two models and make corresponding figures that were used during the analysis
    regr = LinearRegression()
    clf = tree.DecisionTreeClassifier()
    create_figures(X_train, y_train, y_train_binned, regr, clf)


