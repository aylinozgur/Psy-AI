
import pickle
import warnings

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', 400)
pd.set_option('display.width', 10000)
pd.set_option('display.max_rows', 1000)


df_analysis = pd.read_csv(r"C:\Users\bkbar\PycharmProjects\addiction_project\df_analysis.csv")
df_analysis_copy = df_analysis.copy()

df_recommendation = pd.read_csv(r"C:\Users\bkbar\PycharmProjects\addiction_project\df_tips.csv")
df_df_recommendation_copy = df_analysis.copy()

df_analysis.columns = ["".join(char if char.isalnum() else "_" for char in col) for col in df_analysis.columns]
df_recommendation.columns = ["".join(char if char.isalnum() else "_" for char in col) for col in df_recommendation.columns]

def ordinal_adiction_question_enumeration(dataframe):

    new_column_names = {
        "Education_Occupation__currently_doing_": "ORDINAL_Q1",
        "Social_media_apps_in_which_you_have_accounts": "ORDINAL_Q2",
        "Apps_that_you_are_frequently_using": "ORDINAL_Q3",
        "you_are_using_social_media_since_": "ORDINAL_Q4",
        "You_are_using_social_media_since_": "ORDINAL_Q4",
        "Average_time_you_spent_on_social_media_everyday": "ORDINAL_Q5",
        "How_often_do_you_respond_to_the_notification_of_social_media_": "ORDINAL_Q6",
        "You_spend_a_lot_of_time_thinking_about_social_media_or_planning_how_to_use_it_": "ADDICTION_Q1",
        "You_feel_an_urge_to_use_social_media_more_and_more_": "ADDICTION_Q2",
        "You_use_social_media_in_order_to_forget_about_personal_problems_": "ADDICTION_Q3",
        "You_have_tried_to_cut_down_on_the_use_of_social_media_without_success_": "ADDICTION_Q4",
        "You_become_restless_or_troubled_if_you_are_prohibited_from_using_social_media_": "ADDICTION_Q5",
        "You_use_social_media_in_a_way_that_achieving_your_goals_academic_score_becomes_tough_stressful_": "ADDICTION_Q6"
    }

    dataframe.rename(columns=new_column_names, inplace=True)

ordinal_adiction_question_enumeration(df_analysis)

def selfesteem_question_enumeration(dataframe):

    new_column_names_2 = {
        "I_am_satisfied_with_work_what_I_do_": "SELFESTEEM_Q1_POS",
        "At_times_I_think_I_am_no_good_at_all_": "SELFESTEEM_Q2_NEG",
        "_I_feel_that_I_have_a_good_qualities_": "SELFESTEEM_Q3_POS",
        "I_am_able_to_do_things_as_well_as_most_other_people__": "SELFESTEEM_Q4_POS",
        "I_do_not_feel_much_proud_of_my_ability_": "SELFESTEEM_Q5_NEG",
        "Sometimes__I__feel_my_expertise_has_no_use_": "SELFESTEEM_Q6_NEG",
        "I_feel_that_I_m_here_with_purpose_as_others_": "SELFESTEEM_Q7_POS",
        "I_wish_I_could_have_more_respect_for_myself__": "SELFESTEEM_Q8_NEG",
        "I_feel_that_I_can_not_achieve_goals_dreams__as_others_do_": "SELFESTEEM_Q9_NEG",
        "I_take_a_positive_attitude_toward_myself_": "SELFESTEEM_Q10_POS"
    }

    dataframe.rename(columns=new_column_names_2, inplace=True)

selfesteem_question_enumeration(df_analysis)


def job_category_sep(job):

    ogrenci_kelimeler = ['(UG)', '(PG)', 'PhD', 'scholar', 'પરીક્ષાની', 'net', 'B.ed', 'M.A', 'M. Ed']


    calisan_kelimeler = ['Doctor', 'Job', 'Housewife', 'Singer', 'writer', 'translator',
                         'Preparation for competitive exams', 'Preparation for government job',
                         'Government job ni taiyaari', 'Unemployee', 'Police constable', 'running']

    job = job.strip()  # Boşlukları temizleme

    if any(kelime in job for kelime in ogrenci_kelimeler):
        return 'Student'
    elif any(kelime in job for kelime in calisan_kelimeler):
        return 'Employed'
    else:
        return 'Others'

df_analysis["ORDINAL_Q1"] = df_analysis["ORDINAL_Q1"].apply(job_category_sep)


def social_media_enc(dataframe):

    social_media_platforms = ['WhatsApp', 'Youtube', 'Instagram', 'Facebook', 'Twitter', 'Telegram', 'LinkedIn']


    for platform in social_media_platforms:

        dataframe[f'HAVE_{platform}_ACCOUNT'] = dataframe["ORDINAL_Q2"].apply(
            lambda x: 1 if platform in x else 0)

social_media_enc(df_analysis)


def social_media_active_enc(dataframe):


    social_media_platforms = ['WhatsApp', 'Youtube', 'Instagram', 'Facebook', 'Twitter', 'Telegram', 'LinkedIn']

    for platform in social_media_platforms:

        dataframe[f'ACTIVE_{platform}_USER'] = dataframe['ORDINAL_Q3'].apply(
            lambda x: 1 if platform in x else 0)

social_media_active_enc(df_analysis)


def social_media_calculator(dataframe, col_names):
    social_media_counts_q2 = []
    social_media_counts_q3 = []

    for index, row in dataframe.iterrows():
        content_count_q2 = 0
        content_count_q3 = 0
        for col_name in col_names:
            social_media_content = str(row[col_name]).split(', ')
            if col_name == "ORDINAL_Q2":
                content_count_q2 += len(social_media_content)
            elif col_name == "ORDINAL_Q3":
                content_count_q3 += len(social_media_content)

        social_media_counts_q2.append(content_count_q2)
        social_media_counts_q3.append(content_count_q3)

    dataframe['SOCIAL_MEDIA_COUNT_COMBINED_ORDINAL_Q2'] = social_media_counts_q2
    dataframe['SOCIAL_MEDIA_COUNT_COMBINED_ORDINAL_Q3'] = social_media_counts_q3

    dataframe.drop(["ORDINAL_Q2", "ORDINAL_Q3"], axis=1, inplace=True)


social_media_calculator(df_analysis, col_names=["ORDINAL_Q2", "ORDINAL_Q3"])
"""
def addiction_questions_enc(dataframe):

    encode_dict = {
        'Very rarely': 1,
        'rarely': 2,
        'sometimes': 3,
        'often': 4,
        'Very often': 5
    }

    for i in range(1, 7):
        dataframe[f"ADDICTION_Q{i}"] = dataframe[f"ADDICTION_Q{i}"].map(encode_dict)

addiction_questions_enc(df_analysis)

def selfestem_question_enc(dataframe):

    encode_dict_POS = {
        'Strongly agree': 4,
        'Agree': 3,
        'Strongly disagree': 1,
        'Disagree': 2
    }

    encode_dict_NEG = {
        'Strongly agree': 1,
        'Agree': 2,
        'Strongly disagree': 4,
        'Disagree': 3
    }

    # 'Experience' sütunu için encode işlemi uygulama
    for i in range(1, 11):
        column_name = f"SELFESTEEM_Q{i}"
        if 'POS' in column_name:
            dataframe[column_name] = dataframe[column_name].map(encode_dict_POS)
        elif 'NEG' in column_name:
            dataframe[column_name] = dataframe[column_name].map(encode_dict_NEG)

selfestem_question_enc(df_analysis)
"""
def ordinal_enc(dataframe):

    columns_to_encode_addiction = ["ADDICTION_Q1", "ADDICTION_Q2", "ADDICTION_Q3", "ADDICTION_Q4", "ADDICTION_Q5", "ADDICTION_Q6"]
    columns_to_encode_categories_selfesteem_pos = ["SELFESTEEM_Q1_POS", "SELFESTEEM_Q3_POS", "SELFESTEEM_Q4_POS", "SELFESTEEM_Q7_POS", "SELFESTEEM_Q10_POS"]
    columns_to_encode_categories_selfesteem_neg = ["SELFESTEEM_Q2_NEG", "SELFESTEEM_Q5_NEG", "SELFESTEEM_Q6_NEG", "SELFESTEEM_Q8_NEG", "SELFESTEEM_Q9_NEG"]

    columns_to_encode_ORDINAL_Q4 = ["ORDINAL_Q4"]
    columns_to_encode_ORDINAL_Q5 = ["ORDINAL_Q5"]
    columns_to_encode_ORDINAL_Q6 = ["ORDINAL_Q6"]

    categories_addiction = [['Very rarely', 'rarely', 'sometimes', 'often', 'Very often']]
    categories_selfesteem_pos = [['Strongly disagree', 'Disagree', 'Agree', 'Strongly agree']]
    categories_selfesteem_neg = [['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree']]

    categories_ORDINAL_Q4 = [["less than a year", "1 year", "2 years", "3 years", "4 years",
                              "5 years", "more than 5 years"]]

    categories_ORDINAL_Q5 = [["less than a hour", "1 - 2hours", "2 - 3 hours", "3 - 4 hours", "> 4 hours"]]

    categories_ORDINAL_Q6 = [["I put off notifications", "I Don't respond if it's not important",
                              "Whenever took a phone", "As soon as possible",
                              "Immediately only if it's important", "Immediately"]]

    for column in columns_to_encode_addiction:
        encoder = OrdinalEncoder(categories=categories_addiction)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])

    for column in columns_to_encode_categories_selfesteem_pos:
        encoder = OrdinalEncoder(categories=categories_selfesteem_pos)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])

    for column in columns_to_encode_categories_selfesteem_neg:
        encoder = OrdinalEncoder(categories=categories_selfesteem_neg)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])

    for column in columns_to_encode_ORDINAL_Q4:
        encoder = OrdinalEncoder(categories=categories_ORDINAL_Q4)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])
        print("ORDINAL Q4", encoder.categories)

    for column in columns_to_encode_ORDINAL_Q5:
        encoder = OrdinalEncoder(categories=categories_ORDINAL_Q5)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])

    for column in columns_to_encode_ORDINAL_Q6:
        encoder = OrdinalEncoder(categories=categories_ORDINAL_Q6, handle_unknown='use_encoded_value', unknown_value=-1)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])

ordinal_enc(df_analysis)
def data_fixer(dataframe):
    numeric_columns = dataframe.select_dtypes(include='number').columns
    dataframe[numeric_columns] = dataframe[numeric_columns].astype(int)
    dataframe.rename(str.upper, axis='columns', inplace=True)

data_fixer(df_analysis)
def label_enc(dataframe):

    columns_to_encode_addiction = ["AGE", "GENDER", "ORDINAL_Q1", "ORDINAL_Q4", "ORDINAL_Q5"]

    for column in columns_to_encode_addiction:
        label_encoder = LabelEncoder()
        dataframe[column] = label_encoder.fit_transform(dataframe[[column]])
        #print(column + ": " + label_encoder.classes_)
        print(column)
        print(label_encoder.classes_)

label_enc(df_analysis)

def categorize_addiction_score(row):
    if 13 < row <= 24:
        return 2
    elif 9 < row <= 13:
        return 1
    else:
        return 0

def calculate_addiction_score(dataframe):
    addiction_columns = ["ADDICTION_Q1", "ADDICTION_Q2", "ADDICTION_Q3",
                         "ADDICTION_Q4", "ADDICTION_Q5", "ADDICTION_Q6"]
    dataframe["ADDICTION_SCORE"] = dataframe[addiction_columns].sum(axis=1)
    dataframe["ADDICTION_CATEGORY"] = dataframe["ADDICTION_SCORE"].apply(categorize_addiction_score)

calculate_addiction_score(df_analysis)

# (SOCIAL_MEDIA_COUNT_COMBINED_ORDINAL_Q3 / SOCIAL_MEDIA_COUNT_COMBINED_ORDINAL_Q2) * SELFESTEEM_SCORE = SELFESTEEM_SCORE_WEİGHTED
# SELFESTEEM_Qi_POS = SELFESTEM_POS_SCORE
# SELFESTEEM_Qi_NEG = SELFESTEM_NEG_SCORE
def categorize_selfesteem_score(row):
    if 20 < row <= 30:
        return 0
    elif 16 < row <= 19:
        return 1
    else:
        return 2
#
def calculate_selfesteem_score(dataframe):
    selfesteem_columns = ["SELFESTEEM_Q1_POS", "SELFESTEEM_Q2_NEG", "SELFESTEEM_Q3_POS",
                         "SELFESTEEM_Q4_POS", "SELFESTEEM_Q5_NEG", "SELFESTEEM_Q6_NEG",
                         "SELFESTEEM_Q7_POS" ,"SELFESTEEM_Q8_NEG",  "SELFESTEEM_Q9_NEG",  "SELFESTEEM_Q10_POS"]
    dataframe["SELFESTEEM_SCORE"] = dataframe[selfesteem_columns].sum(axis=1)
    # Calculate SELFESTEEM_SCORE_WEIGHTED
    dataframe['SELFESTEEM_SCORE_WEIGHTED'] = (dataframe['SOCIAL_MEDIA_COUNT_COMBINED_ORDINAL_Q3'] / dataframe[
        'SOCIAL_MEDIA_COUNT_COMBINED_ORDINAL_Q2']) * dataframe['SELFESTEEM_SCORE']

    # Define columns related to positive and negative scores
    positive_columns = ["SELFESTEEM_Q1_POS", "SELFESTEEM_Q3_POS", "SELFESTEEM_Q4_POS", "SELFESTEEM_Q7_POS",
                        "SELFESTEEM_Q10_POS"]
    negative_columns = ["SELFESTEEM_Q2_NEG", "SELFESTEEM_Q5_NEG", "SELFESTEEM_Q6_NEG", "SELFESTEEM_Q8_NEG",
                        "SELFESTEEM_Q9_NEG"]

    # Calculate SELFESTEEM_Qi_POS and SELFESTEEM_Qi_NEG
    dataframe['SELFESTEEM_Qi_POS'] = dataframe[positive_columns].sum(axis=1)
    dataframe['SELFESTEEM_Qi_NEG'] = dataframe[negative_columns].sum(axis=1)
    # feature olarak ekle.
    dataframe["SELFESTEEM_CATEGORY"] = dataframe["SELFESTEEM_SCORE"].apply(categorize_selfesteem_score)

# Fonksiyonu çağırma
calculate_selfesteem_score(df_analysis)


df_analysis['ADDICTION_CATEGORY'] = df_analysis['ADDICTION_CATEGORY'].replace('2', '1')
df_analysis_3_category = df_analysis
df_analysis["ADDICTION_CATEGORY"].value_counts()

df_analysis['ADDICTION_CATEGORY'] = [1 if value == 2 else value for value in df_analysis['ADDICTION_CATEGORY']]

df_profile_analysis = df_analysis.drop(columns=[col for col in df_analysis.columns if 'ADDICTION_Q' in col or 'SCORE' in col])
cat_variables_df_profile_analysis = ["SELFESTEEM_Q1_POS", "SELFESTEEM_Q2_NEG", "SELFESTEEM_Q3_POS","SELFESTEEM_Q4_POS", "SELFESTEEM_Q5_NEG",
                                     "SELFESTEEM_Q6_NEG","SELFESTEEM_Q7_POS" ,"SELFESTEEM_Q8_NEG",  "SELFESTEEM_Q9_NEG",  "SELFESTEEM_Q10_POS",
                                     "GENDER",  "ORDINAL_Q1",  "ORDINAL_Q4",  "ORDINAL_Q5","ORDINAL_Q6", "HAVE_WHATSAPP_ACCOUNT",  "HAVE_YOUTUBE_ACCOUNT",
                                     "HAVE_INSTAGRAM_ACCOUNT",  "HAVE_FACEBOOK_ACCOUNT",  "HAVE_TWITTER_ACCOUNT",  "HAVE_TELEGRAM_ACCOUNT",
                                     "HAVE_LINKEDIN_ACCOUNT", "ACTIVE_WHATSAPP_USER","ACTIVE_YOUTUBE_USER","ACTIVE_INSTAGRAM_USER",  "ACTIVE_FACEBOOK_USER",
                                     "ACTIVE_TWITTER_USER","ACTIVE_TELEGRAM_USER",  "ACTIVE_LINKEDIN_USER", "SELFESTEEM_CATEGORY"]







#df_analysis.to_csv('df_analysis_grafik.csv', index=False)
#df_recommendation.to_csv('df_recommendation_grafik.csv', index=False)
#Pipeline
"""
def data_prep(dataframe):
    dataframe.columns = dataframe.columns.str.replace('[^a-zA-Z0-9]', '_')
    ordinal_adiction_question_enumeration(dataframe)
    selfesteem_question_enumeration(dataframe)

    def job_category_sep(job):

        ogrenci_kelimeler = ['(UG)', '(PG)', 'PhD', 'scholar', 'પરીક્ષાની', 'net', 'B.ed', 'M.A', 'M. Ed']

        calisan_kelimeler = ['Doctor', 'Job', 'Housewife', 'Singer', 'writer', 'translator',
                             'Preparation for competitive exams', 'Preparation for government job',
                             'Government job ni taiyaari', 'Unemployee', 'Police constable', 'running']

        job = job.strip()  # Boşlukları temizleme

        if any(kelime in job for kelime in ogrenci_kelimeler):
            return 'Student'
        elif any(kelime in job for kelime in calisan_kelimeler):
            return 'Employed'
        else:
            return 'Others'

    dataframe["ORDINAL_Q1"] = dataframe["ORDINAL_Q1"].apply(job_category_sep)

    social_media_calculator(dataframe, col_names=["ORDINAL_Q2", "ORDINAL_Q3"])
    ordinal_enc(dataframe)
    data_fixer(dataframe)
    label_enc(dataframe)

    def categorize_addiction_score(row):
        if 13 < row <= 24:
            return 2
        elif 9 < row <= 13:
            return 1
        else:
            return 0

    calculate_addiction_score(dataframe)

    def categorize_selfesteem_score(row):
        if 20 < row <= 30:
            return 0
        elif 16 < row <= 19:
            return 1
        else:
            return 2

    calculate_selfesteem_score(dataframe)

    dataframe['ADDICTION_CATEGORY'] = dataframe['ADDICTION_CATEGORY'].replace('2', '1')
    df_analysis_3_category = dataframe
    dataframe["ADDICTION_CATEGORY"].value_counts()

    dataframe['ADDICTION_CATEGORY'] = [1 if value == 2 else value for value in dataframe['ADDICTION_CATEGORY']]

    df_profile_analysis = dataframe.drop(
        columns=[col for col in dataframe.columns if 'ADDICTION_Q' in col or 'SCORE' in col])
    cat_variables_df_profile_analysis = ["SELFESTEEM_Q1_POS", "SELFESTEEM_Q2_NEG", "SELFESTEEM_Q3_POS",
                                         "SELFESTEEM_Q4_POS", "SELFESTEEM_Q5_NEG", "SELFESTEEM_Q6_NEG",
                                         "SELFESTEEM_Q7_POS", "SELFESTEEM_Q8_NEG", "SELFESTEEM_Q9_NEG",
                                         "SELFESTEEM_Q10_POS",
                                         "GENDER", "ORDINAL_Q1", "ORDINAL_Q4", "ORDINAL_Q5",
                                         "ORDINAL_Q6", "HAVE_OTHER_ACCOUNT", "HAVE_WHATSAPP_ACCOUNT",
                                         "HAVE_YOUTUBE_ACCOUNT",
                                         "HAVE_INSTAGRAM_ACCOUNT", "HAVE_FACEBOOK_ACCOUNT", "HAVE_TWITTER_ACCOUNT",
                                         "HAVE_TELEGRAM_ACCOUNT",
                                         "HAVE_LINKEDIN_ACCOUNT", "ACTIVE_OTHER_USER", "ACTIVE_WHATSAPP_USER",
                                         "ACTIVE_YOUTUBE_USER", "ACTIVE_INSTAGRAM_USER", "ACTIVE_FACEBOOK_USER",
                                         "ACTIVE_TWITTER_USER", "ACTIVE_TELEGRAM_USER", "ACTIVE_LINKEDIN_USER",
                                         "SELFESTEEM_CATEGORY"]

    y = df_profile_analysis.drop('ADDICTION_CATEGORY', axis=1)
    X = df_profile_analysis('ADDICTION_CATEGORY', axis=1)

    return X, y

data_prep()
"""


X = df_profile_analysis.drop('ADDICTION_CATEGORY', axis=1)  # Features
y = df_profile_analysis['ADDICTION_CATEGORY']




##########################################################################
#CatBoost
##########################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=45)
# Create a CatBoostClassifier instance
catboost_model = CatBoostClassifier(random_state=17, cat_features=cat_variables_df_profile_analysis)


#############################################################
"""
def objective(trial):
    catboost_params = {
        'iterations': trial.suggest_int('iterations_catboost', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate_catboost', 0.01, 0.1),
        'depth': trial.suggest_int('depth_catboost', 3, 12),
        # Add other CatBoost hyperparameters here
    }

    catboost = CatBoostClassifier(**catboost_params)

    # Construct the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('catboost', catboost)
    ])

    # Stratified K-Fold Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Calculate the average accuracy across folds
    average_accuracy = sum(accuracies) / len(accuracies)
    return average_accuracy

    roc_auc_scores = []  # To store ROC AUC scores for each class

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Predict probabilities
        y_pred_proba = pipeline.predict_proba(X_test)

        # Calculate ROC AUC for each class
        for class_index in range(y_pred_proba.shape[1]):
            roc_auc_class = roc_auc_score(y_test == class_index, y_pred_proba[:, class_index])
            roc_auc_scores.append(roc_auc_class)

    # Calculate the average ROC AUC across all classes and folds
    average_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
    return average_roc_auc

# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # You can increase the number of trials for better optimization
"""

# Hyperparametre Optuna Part
###################################################################
#Best Parameters: {'iterations_catboost': 348, 'learning_rate_catboost': 0.0605740834289543, 'depth_catboost': 10}

#Final Model
best_params = {'l2_leaf_reg': 3.5697905727515273, 'iterations': 488, 'depth': 9, 'learning_rate': 0.015088593820859258}  # Fix parameter key
#value: 0.6617336152219873 and parameters: {'iterations_catboost': 488, 'learning_rate_catboost': 0.015088593820859258, 'depth_catboost': 9}. Best is trial 53 with value: 0.6849894291754757.

catboost_random_final = CatBoostClassifier(
    random_state=17,
    cat_features=cat_variables_df_profile_analysis,
    **best_params
)

catboost_random_final.fit(X_train, y_train)

y_pred = catboost_random_final.predict(X_test)  # Fix predict parameter order

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_pred_proba = catboost_random_final.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, y_pred_proba)


print("ROC AUC Scores for Each Class:", roc) #0.8208333333333333
print("f1 Scores for Each Class:", f1) #0.5625000000000001
print("Accuracy Scores for Each Class:", accuracy) #0.6818181818181818

pickle.dump(catboost_random_final, open("../../Downloads/miuul_final_model_catboost.pkl", "wb"))
y_test

################################################################
# FEATURE IMPORTANCE
################################################################
"""
def plot_catboost_importances(model, plot=False, num=10):
    feat_imp = pd.DataFrame({'feature': model.feature_names_,
                             'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="importance", y="feature", data=feat_imp.head(25))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

# Usage for CatBoost
feat_imp_catboost = plot_catboost_importances(catboost_random_final, num=30, plot=True)
"""


"""
#df_recommendation
ordinal_adiction_question_enumeration(df_recommendation)

def tips_questions_enumeration(dataframe):
    new_column_names_3 = {
        "Recognizing_Accepting_social_media_addiction" : "TIPS_Q1",
        "Turn_off_the_notifications" : "TIPS_Q2",
        "Delete_unused_apps_social_media_accounts" : "TIPS_Q3",
        "Limit_your_screen_time" : "TIPS_Q4",
        "Use_your_phone_with_purpose__When_you_want_to_use_your_phone__consider_the_reason_why_" : "TIPS_Q5",
        "Avoid_sleeping_with_mobile__keeping_digital_devices_away_for_at_least_one_hour_before_sleep_" : "TIPS_Q6",
        "Remove_your_phone_from_your_morning_routine_": "TIPS_Q7",
        "Keep_your_phone_out_of_reach_to_focus_on_work": "TIPS_Q8",
        "Use_social_media_as_a_treat__allow_yourself_to_use_social_media_for_sometime_when_you_have_done_your_work_achieve_something_": "TIPS_Q9",
        "Meet_people_offline_whenever_possible__Check_In_With_Friends_and_Family_": "TIPS_Q10",
        "Purge_your__friends__and__follow__lists_time_to_time__People_who_probably_don_t_add_positive_value_to_your_life_and_often_trigger_you_into_unnecessary_conversations_in_social_media_": "TIPS_Q11",
        "Take_a_break_from_social_media_time_to_time__once_in_a_week_": "TIPS_Q12",
        "Invest_time_in_physical_activity_outdoor_games": "TIPS_Q13",
        "Get_a_new_hobby": "TIPS_Q14",
        "Have_lunch_dinner_with_family_without_phones_": "TIPS_Q15",
        "Turn_display_to_grey_pixel": "TIPS_Q16",
        "Plan_your_time_with_conscious_choice": "TIPS_Q17"
    }
    dataframe.rename(columns=new_column_names_3, inplace=True)

tips_questions_enumeration(df_recommendation)
df_recommendation.rename(str.upper, axis='columns', inplace=True)



def ordinal_encoder_recommendation(dataframe):
    columns_to_encode_addiction = ["ADDICTION_Q1", "ADDICTION_Q2", "ADDICTION_Q3", "ADDICTION_Q4", "ADDICTION_Q5",
                                   "ADDICTION_Q6"]

    columns_to_encode_ORDINAL_Q4 = ["ORDINAL_Q4"]
    columns_to_encode_ORDINAL_Q5 = ["ORDINAL_Q5"]
    columns_to_encode_ORDINAL_Q6 = ["ORDINAL_Q6"]

    categories_addiction = [['Very rarely', 'rarely', 'sometimes', 'often', 'Very often']]

    categories_ORDINAL_Q4 = [["less than a year", "1 year", "2 years", "3 years", "4 years",
                                         "5 years", "more than 5 years"]]

    categories_ORDINAL_Q5 = [["less than a hour", "1 - 2 hours", "2 - 3 hours", "3 - 4 hours", "more than 4 hours"]]

    categories_ORDINAL_Q6 = [["I put off notifications", "I Don't respond if it's not important",
                                         "Whenever took a phone", "As soon as possible",
                                         "Immediately only if it's important", "Immediately"]]

    for column in columns_to_encode_addiction:
        encoder = OrdinalEncoder(categories=categories_addiction)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])

    for column in columns_to_encode_ORDINAL_Q4:
        encoder = OrdinalEncoder(categories=categories_ORDINAL_Q4)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])

    for column in columns_to_encode_ORDINAL_Q5:
        encoder = OrdinalEncoder(categories=categories_ORDINAL_Q5)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])

    for column in columns_to_encode_ORDINAL_Q6:
        encoder = OrdinalEncoder(categories=categories_ORDINAL_Q6, handle_unknown='use_encoded_value', unknown_value=-1)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])



ordinal_encoder_recommendation(df_recommendation)

def label_enc_recommendation(dataframe):

    columns_to_encode_addiction = ["AGE", "GENDER"]

    for column in columns_to_encode_addiction:
        label_encoder = LabelEncoder()
        dataframe[column] = label_encoder.fit_transform(dataframe[[column]])

label_enc_recommendation(df_recommendation)

calculate_addiction_score(df_recommendation)


#KORELASYON MATRİX'İ

tips_columns = [col for col in df_recommendation.columns if 'TIPS' in col]
correlation_matrix = df_recommendation[tips_columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix for TIPS Columns')
plt.show()

correlation_matrix = df_recommendation[tips_columns].corr()

result_df = pd.DataFrame(columns=['Variable1', 'Variable2', 'Correlation Coefficient', 'P Value'])

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        var1 = correlation_matrix.columns[i]
        var2 = correlation_matrix.columns[j]
        correlation_coefficient, p_value = pearsonr(df_recommendation[var1], df_recommendation[var2])

        if p_value < 0.05:
            result_df = pd.concat([result_df, pd.DataFrame({
                'Variable1': [var1],
                'Variable2': [var2],
                'Correlation Coefficient': [correlation_coefficient],
                'P Value': [p_value]
            })], ignore_index=True)


tips_significant_correlations = result_df[result_df["Correlation Coefficient"] <= 0.1]
tips_significant_correlations.shape #15 tane ikili var.


# df_recommendation[""]
"""