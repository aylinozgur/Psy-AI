import pickle
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from streamlit_option_menu import option_menu
from json import load


class Config:
    def __init__(self):
        with open("language.json") as f:
            self.config_data = load(f)

    def __getattr__(self, name):
        try:
            return self.config_data[name]
        except KeyError:
            raise AttributeError(f"Configuration key '{name}' does not exist.")


config = Config()


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


def social_media_enc(dataframe):
    social_media_platforms = ['WhatsApp', 'Youtube', 'Instagram', 'Facebook', 'Twitter', 'Telegram', 'LinkedIn']

    dataframe['HAVE_OTHER_ACCOUNT'] = dataframe["ORDINAL_Q2"].apply(
        lambda x: 1 if any(platform not in social_media_platforms for platform in x) else 0)

    for platform in social_media_platforms:
        dataframe[f'HAVE_{platform}_ACCOUNT'] = dataframe["ORDINAL_Q2"].apply(
            lambda x: 1 if platform in x else 0)


def social_media_active_enc(dataframe):
    social_media_platforms = ['WhatsApp', 'Youtube', 'Instagram', 'Facebook', 'Twitter', 'Telegram', 'LinkedIn']

    # Active Other User Column
    dataframe['ACTIVE_OTHER_USER'] = dataframe['ORDINAL_Q3'].apply(
        lambda x: 1 if any(platform not in social_media_platforms for platform in x) else 0)

    # Active Platform User Columns
    for platform in social_media_platforms:
        dataframe[f'ACTIVE_{platform}_USER'] = dataframe['ORDINAL_Q3'].apply(
            lambda x: 1 if platform in x else 0)


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


def ordinal_enc(dataframe):
    columns_to_encode_addiction = ["ADDICTION_Q1", "ADDICTION_Q2", "ADDICTION_Q3", "ADDICTION_Q4", "ADDICTION_Q5",
                                   "ADDICTION_Q6"]
    columns_to_encode_categories_selfesteem_pos = ["SELFESTEEM_Q1_POS", "SELFESTEEM_Q3_POS", "SELFESTEEM_Q4_POS",
                                                   "SELFESTEEM_Q7_POS", "SELFESTEEM_Q10_POS"]
    columns_to_encode_categories_selfesteem_neg = ["SELFESTEEM_Q2_NEG", "SELFESTEEM_Q5_NEG", "SELFESTEEM_Q6_NEG",
                                                   "SELFESTEEM_Q8_NEG", "SELFESTEEM_Q9_NEG"]

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

    for column in columns_to_encode_ORDINAL_Q5:
        encoder = OrdinalEncoder(categories=categories_ORDINAL_Q5)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])

    for column in columns_to_encode_ORDINAL_Q6:
        encoder = OrdinalEncoder(categories=categories_ORDINAL_Q6, handle_unknown='use_encoded_value', unknown_value=-1)
        dataframe[column] = encoder.fit_transform(dataframe[[column]])


def data_fixer(dataframe):
    numeric_columns = dataframe.select_dtypes(include='number').columns
    dataframe[numeric_columns] = dataframe[numeric_columns].astype(int)
    dataframe.rename(str.upper, axis='columns', inplace=True)


def label_enc(dataframe):
    columns_to_encode_addiction = ["AGE", "GENDER", "ORDINAL_Q1", "ORDINAL_Q4", "ORDINAL_Q5"]

    for column in columns_to_encode_addiction:
        label_encoder = LabelEncoder()
        dataframe[column] = label_encoder.fit_transform(dataframe[[column]])


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


def categorize_selfesteem_score(row):
    if 20 < row <= 30:
        return 0
    elif 16 < row <= 19:
        return 1
    else:
        return 2


def calculate_selfesteem_score(dataframe):
    selfesteem_columns = ["SELFESTEEM_Q1_POS", "SELFESTEEM_Q2_NEG", "SELFESTEEM_Q3_POS",
                          "SELFESTEEM_Q4_POS", "SELFESTEEM_Q5_NEG", "SELFESTEEM_Q6_NEG",
                          "SELFESTEEM_Q7_POS", "SELFESTEEM_Q8_NEG", "SELFESTEEM_Q9_NEG", "SELFESTEEM_Q10_POS"]
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


def data_prep(dataframe):
    dataframe.columns = ["".join(char if char.isalnum() else "_" for char in col) for col in dataframe.columns]
    ordinal_adiction_question_enumeration(dataframe)
    selfesteem_question_enumeration(dataframe)
    social_media_enc(dataframe)
    social_media_active_enc(dataframe)

    social_media_calculator(dataframe, col_names=["ORDINAL_Q2", "ORDINAL_Q3"])
    data_fixer(dataframe)

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
    dataframe['ADDICTION_CATEGORY'] = [1 if value == 2 else value for value in dataframe['ADDICTION_CATEGORY']]

    dataframe = dataframe.drop(columns=[col for col in dataframe.columns if 'ADDICTION' in col])
    cat_variables_df_profile_analysis = ["SELFESTEEM_Q1_POS", "SELFESTEEM_Q2_NEG", "SELFESTEEM_Q3_POS",
                                         "SELFESTEEM_Q4_POS", "SELFESTEEM_Q5_NEG", "SELFESTEEM_Q6_NEG",
                                         "SELFESTEEM_Q7_POS", "SELFESTEEM_Q8_NEG", "SELFESTEEM_Q9_NEG",
                                         "SELFESTEEM_Q10_POS",
                                         "GENDER", "ORDINAL_Q1", "ORDINAL_Q4", "ORDINAL_Q5",
                                         "ORDINAL_Q6", "HAVE_WHATSAPP_ACCOUNT", "HAVE_YOUTUBE_ACCOUNT",
                                         "HAVE_INSTAGRAM_ACCOUNT", "HAVE_FACEBOOK_ACCOUNT", "HAVE_TWITTER_ACCOUNT",
                                         "HAVE_TELEGRAM_ACCOUNT", "HAVE_LINKEDIN_ACCOUNT", "ACTIVE_WHATSAPP_USER",
                                         "ACTIVE_YOUTUBE_USER", "ACTIVE_INSTAGRAM_USER", "ACTIVE_FACEBOOK_USER",
                                         "ACTIVE_TWITTER_USER", "ACTIVE_TELEGRAM_USER", "ACTIVE_LINKEDIN_USER",
                                         "SELFESTEEM_CATEGORY"]

    X = dataframe
    return X, cat_variables_df_profile_analysis


def main():
    model = pickle.load(open("model/catboost.pkl", 'rb'))
    data = pd.read_csv("survey_data.csv")

    PAGE_TITLE = "Sosyal Medya Bağımlılık"
    PAGE_ICON = ":bar_chart:"

    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 400px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        language_picker = st.selectbox('Change Language', options=['🇹🇷 Türkçe', '🇬🇧 English'])

        if language_picker == '🇹🇷 Türkçe':
            translation = config.TR
        else:
            translation = config.ENG

        options = [translation["PAGE_OPTIONS_SOCIAL_TITLE"], translation["PAGE_OPTIONS_ADDICTON_TEST"],
                   translation["PAGE_OPTIONS_BLOG"],
                   translation["PAGE_OPTIONS_OUR_GLOAL"], translation["PAGE_OPTIONS_CONTACT_US"],
                   translation["PAGE_OPTIONS_DEVELOPERS"]]
        selected_option = option_menu(translation["SIDEBAR_TITLE"], options,
                                      icons=['bi bi-house',
                                             'bi bi-bar-chart-fill',
                                             'bi bi-magic',
                                             'bi bi-projector',
                                             'bi bi-calculator-fill',
                                             'wrench',
                                             'thermometer-snow'],
                                      # icons = https://icons.getbootstrap.com/
                                      menu_icon="cast",
                                      default_index=0,
                                      styles={
                                          "container": {"padding": "5!important"},
                                          "background-color": "#fafafa",
                                          "icon": {"font-size": "25px"},
                                          "color": "orange",

                                          "nav-link": {"font-size": "16px",
                                                       "text-align": "left",
                                                       "margin": "0px"},

                                          "--hover-color": "#eee",
                                          "nav-link-selected": {"background-color": "#ed145b"},
                                      }
                                      )

    if selected_option == translation["PAGE_OPTIONS_SOCIAL_TITLE"]:

        st.title(translation["PAGE_OPTIONS_SOCIAL_TITLE"])

        columns = [
            "Age",
            "Gender",
            "Education_Occupation__currently_doing_",
            "Social_media_apps_in_which_you_have_accounts",
            "Apps_that_you_are_frequently_using",
            "you_are_using_social_media_since_",
            "Average_time_you_spent_on_social_media_everyday",
            "How_often_do_you_respond_to_the_notification_of_social_media_",
            "You_spend_a_lot_of_time_thinking_about_social_media_or_planning_how_to_use_it_",
            "You_feel_an_urge_to_use_social_media_more_and_more_",
            "You_use_social_media_in_order_to_forget_about_personal_problems_",
            "You_have_tried_to_cut_down_on_the_use_of_social_media_without_success_",
            "You_become_restless_or_troubled_if_you_are_prohibited_from_using_social_media_",
            "You_use_social_media_in_a_way_that_achieving_your_goals_academic_score_becomes_tough_stressful_",
            "I_am_satisfied_with_work_what_I_do_",
            "At_times_I_think_I_am_no_good_at_all_",
            "_I_feel_that_I_have_a_good_qualities_",
            "I_am_able_to_do_things_as_well_as_most_other_people__",
            "I_do_not_feel_much_proud_of_my_ability_",
            "Sometimes__I__feel_my_expertise_has_no_use_",
            "I_feel_that_I_m_here_with_purpose_as_others_",
            "I_wish_I_could_have_more_respect_for_myself__",
            "I_feel_that_I_can_not_achieve_goals_dreams__as_others_do_",
            "I_take_a_positive_attitude_toward_myself_"
        ]

        df = pd.DataFrame(columns=columns)

        # Anket soruları ve cevap türleri

        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                age_options = ['13-18', '19-25', '26-35', '>35']
                st.session_state.age_options = age_options
                age = age_options.index(st.selectbox(translation["SELECTBOX_AGE"], age_options, key='age_select'))
                st.session_state.age = age
            with col2:
                gender_options = ['Female', 'Male']
                st.session_state.gender_options = gender_options
                gender = gender_options.index(
                    st.selectbox(translation["SELECTBOX_GENDER"], gender_options, key='gender_select'))
                st.session_state.gender = gender
            with col3:
                ordinal_q1_options = ['Employed', 'Student']
                ORDINAL_Q1 = ordinal_q1_options.index(st.selectbox(translation["SELECTBOX_Q1"], ordinal_q1_options))

        with st.container(border=True):
            col1, col2 = st.columns(2)

            with col1:
                ORDINAL_Q2 = st.multiselect(translation["SELECTBOX_Q2"],
                                            ['WhatsApp', 'Youtube', 'Instagram', 'Facebook', 'Telegram', 'LinkedIn'])
                st.session_state.ordinal_q2 = ORDINAL_Q2

            with col2:
                ORDINAL_Q3 = st.multiselect(translation["SELECTBOX_Q3"],
                                            ['WhatsApp', 'Youtube', 'Instagram', 'Facebook', 'Telegram', 'LinkedIn'])
                st.session_state.ordinal_q3 = ORDINAL_Q3

            with col1:
                ordinal_q4_options = ["less than a year", "1 year", "2 years", "3 years", "4 years", "5 years",
                                      "more than 5 years"]
                ORDINAL_Q4 = ordinal_q4_options.index(st.selectbox(translation["SELECTBOX_Q4"], ordinal_q4_options))

            with col2:
                ordinal_q5_options = ["less than a hour", "1 - 2hours", "2 - 3 hours", "3 - 4 hours", "> 4 hours"]
                ORDINAL_Q5 = ordinal_q5_options.index(st.selectbox(translation["SELECTBOX_Q5"], ordinal_q5_options))

            with col1:
                ordinal_q6_options = ["I put off notifications", "I Don't respond if it's not important",
                                      "Whenever took a phone", "As soon as possible",
                                      "Immediately only if it's important", "Immediately"]

                ORDINAL_Q6 = ordinal_q6_options.index(st.selectbox(translation["SELECTBOX_Q6"], ordinal_q6_options))

        with st.container(border=True):
            col1, col2 = st.columns(2)

            addicton_options = ['Very rarely', 'rarely', 'sometimes', 'often', 'Very often']
            ADDICTION_Q1 = addicton_options.index(st.selectbox(translation["SELECTBOX_ADDICTION_Q1"], addicton_options))
            ADDICTION_Q2 = addicton_options.index(st.selectbox(translation["SELECTBOX_ADDICTION_Q2"], addicton_options))
            ADDICTION_Q3 = addicton_options.index(st.selectbox(translation["SELECTBOX_ADDICTION_Q3"], addicton_options))
            ADDICTION_Q4 = addicton_options.index(st.selectbox(translation["SELECTBOX_ADDICTION_Q4"], addicton_options))
            ADDICTION_Q5 = addicton_options.index(st.selectbox(translation["SELECTBOX_ADDICTION_Q5"], addicton_options))
            ADDICTION_Q6 = addicton_options.index(st.selectbox(translation["SELECTBOX_ADDICTION_Q6"], addicton_options))

            categories_selfesteem_options = ['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree']
            categories_selfesteem_pos_ordering = ['Strongly disagree', 'Disagree', 'Agree', 'Strongly agree']
            categories_selfesteem_neg_ordering = ['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree']

            SELFESTEEM_Q1_POS = categories_selfesteem_pos_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q1"], categories_selfesteem_options))
            SELFESTEEM_Q2_NEG = categories_selfesteem_neg_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q2"], categories_selfesteem_options))
            SELFESTEEM_Q3_POS = categories_selfesteem_pos_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q3"], categories_selfesteem_options))
            SELFESTEEM_Q4_POS = categories_selfesteem_pos_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q4"], categories_selfesteem_options))
            SELFESTEEM_Q5_NEG = categories_selfesteem_neg_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q5"], categories_selfesteem_options))
            SELFESTEEM_Q6_NEG = categories_selfesteem_neg_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q6"], categories_selfesteem_options,
                             key="SELECTBOX_SELFESTEEM_Q6"))
            SELFESTEEM_Q7_POS = categories_selfesteem_pos_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q7"], categories_selfesteem_options))
            SELFESTEEM_Q8_NEG = categories_selfesteem_neg_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q8"], categories_selfesteem_options))
            SELFESTEEM_Q9_NEG = categories_selfesteem_neg_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q9"], categories_selfesteem_options))
            SELFESTEEM_Q10_POS = categories_selfesteem_pos_ordering.index(
                st.selectbox(translation["SELECTBOX_SELFESTEEM_Q10"], categories_selfesteem_options))

        # Cevapları veri çerçevesine ekle
        df.loc[0] = [
            age,
            gender,

            ORDINAL_Q1,
            ORDINAL_Q2,
            ORDINAL_Q3,
            ORDINAL_Q4,
            ORDINAL_Q5,
            ORDINAL_Q6,

            ADDICTION_Q1,
            ADDICTION_Q2,
            ADDICTION_Q3,
            ADDICTION_Q4,
            ADDICTION_Q5,
            ADDICTION_Q6,

            SELFESTEEM_Q1_POS,
            SELFESTEEM_Q2_NEG,
            SELFESTEEM_Q3_POS,
            SELFESTEEM_Q4_POS,
            SELFESTEEM_Q5_NEG,
            SELFESTEEM_Q6_NEG,
            SELFESTEEM_Q7_POS,
            SELFESTEEM_Q8_NEG,
            SELFESTEEM_Q9_NEG,
            SELFESTEEM_Q10_POS
        ]
        # Anketi gönderme butonu
        if st.button(translation["SUBMIT_BUTTON_ADDICTION"]):
            st.title(translation["ADDICTION_RESULT_TITLE"])

            X, cat_variables_df_profile_analysis = data_prep(df)

            X = X.drop("ACTIVE_OTHER_USER", axis=1)
            X = X.drop("HAVE_OTHER_ACCOUNT", axis=1)
            X = X.drop("SELFESTEEM_SCORE", axis=1)
            X = X.drop("SELFESTEEM_SCORE_WEIGHTED", axis=1)

            if (df["ADDICTION_CATEGORY"] == 1).any():

                st.session_state["prediction_value"] = 2
                st.session_state["prediction_proba"] = 100

                st.error(translation["ADDICTION_WARNING_DANGER_W1"])
                st.write(translation["ADDICTION_WARNING_DANGER_W2"])

                tips_dict = {
                    "t1": translation["ADDICTION_TIP_T1"],
                    "t2": translation["ADDICTION_TIP_T2"],
                    "t3": translation["ADDICTION_TIP_T3"],
                    "t4": translation["ADDICTION_TIP_T4"],
                    "t5": translation["ADDICTION_TIP_T5"],
                    "t6": translation["ADDICTION_TIP_T6"],
                    "t7": translation["ADDICTION_TIP_T7"],
                    "t8": translation["ADDICTION_TIP_T8"],
                    "t9": translation["ADDICTION_TIP_T9"],
                    "t10": translation["ADDICTION_TIP_T10"],
                    "t11": translation["ADDICTION_TIP_T11"],
                    "t12": translation["ADDICTION_TIP_T12"],
                    "t13": translation["ADDICTION_TIP_T13"],
                    "t14": translation["ADDICTION_TIP_T14"],
                    "t15": translation["ADDICTION_TIP_T15"],
                    "t16": translation["ADDICTION_TIP_T16"],
                    "t17": translation["ADDICTION_TIP_T17"]
                }

                tip_list = ["t5", "t7", "t8", "t15", "t17"]
                tips_for_q5_0 = ["t3", "t8", "t9", "t13", "t17"]
                tips_for_q5_1 = ["t13", "t15"]
                tips_for_q5_2 = ["t2", "t3", "t5", "t15"]
                tips_for_q5_3 = ["t5", "t6", "t8", "t10", "t13"]
                tips_for_q5_4 = ["t17"]
                tips_for_q6_0 = ["t13", "t14", "t15"]
                tips_for_q6_1 = ["t2", "t13", "t15"]
                tips_for_q6_2 = ["t5"]
                tips_for_q6_3 = ["t3"]
                tips_for_q6_4 = []
                tips_for_q6_5 = ["t9", "t15"]

                if (df["ORDINAL_Q5"] == 0).any():
                    tip_list.extend(tips_for_q5_0)
                elif (df["ORDINAL_Q5"] == 1).any():
                    tip_list.extend(tips_for_q5_1)
                elif (df["ORDINAL_Q5"] == 2).any():
                    tip_list.extend(tips_for_q5_2)
                elif (df["ORDINAL_Q5"] == 3).any():
                    tip_list.extend(tips_for_q5_3)
                elif (df["ORDINAL_Q5"] == 4).any():
                    tip_list.extend(tips_for_q5_4)

                if (df["ORDINAL_Q6"] == 0).any():
                    tip_list.extend(tips_for_q6_0)
                elif (df["ORDINAL_Q6"] == 1).any():
                    tip_list.extend(tips_for_q6_1)
                elif (df["ORDINAL_Q6"] == 2).any():
                    tip_list.extend(tips_for_q6_2)
                elif (df["ORDINAL_Q6"] == 3).any():
                    tip_list.extend(tips_for_q6_3)
                elif (df["ORDINAL_Q6"] == 4).any():
                    tip_list.extend(tips_for_q6_4)
                elif (df["ORDINAL_Q6"] == 5).any():
                    tip_list.extend(tips_for_q6_5)

                tip_list = list(set(tip_list))
                values_list = [tips_dict.get(key) for key in tip_list]

                for value in values_list:
                    st.info(f"Tip: {value}")

                st.success(translation["WARNING_GENERAL"])


            else:
                predictions = model.predict(X)
                proba = model.predict_proba(X)[0, 1]
                st.session_state["prediction_value"] = predictions
                st.session_state["prediction_proba"] = proba

                if predictions == 1:

                    st.info(translation["ADDICTION_WARNING_ADDICTED"])
                    st.success(translation["WARNING_GENERAL"])
                else:
                    st.success(translation["ADDICTION_WARNING_NOT_ADDICTED"])
                    st.success(translation["WARNING_GENERAL"])


    elif selected_option == translation["PAGE_OPTIONS_ADDICTON_TEST"]:
        st.title(translation["CHART_PAGE_TITLE"])

        if "prediction_value" in st.session_state:
            with st.container(border=True):
                if st.session_state.prediction_value == 2:
                    st.metric(label="Tahmin", value="Bağımlısın")
                elif st.session_state.prediction_value == 1:
                    st.metric(label="Tahmin", value="Bağımlığı Olabilirsin")
                elif st.session_state.prediction_value == 0:
                    st.metric(label="Tahmin", value="Bağımlığı Değilsin")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                if not data.empty:
                    if "age" in st.session_state:
                        st.markdown('#### Senin Yaşındakilerin Bağımlılık Durumu')
                        age_position = st.session_state.age
                        age_option = st.session_state.age_options[age_position]

                        bar_chart_data = data[data["Age"] == age_option]
                        bar_chart_data = bar_chart_data["ADDICTION_CATEGORY"].value_counts(normalize=True) * 100
                        bar_chart_data = bar_chart_data.rename({1: "Bağımlı", 0: "Bağımlı Değil"}).reset_index()
                        bar_chart_data = bar_chart_data.rename(
                            {"ADDICTION_CATEGORY": "Bağılımlılık Durumu", "proportion": "Bağımlılık Oranı"}, axis=1)

                        c = (
                            alt.Chart(bar_chart_data)
                            .mark_bar(color='#ed145b', width=200)
                            .encode(x="Bağılımlılık Durumu", y="Bağımlılık Oranı")
                            .configure_axis(grid=False)
                        )

                        st.altair_chart(c, use_container_width=True)
        with col2:
            with st.container(border=True):
                if not data.empty:
                    if "gender" in st.session_state:
                        st.markdown('#### Senin Cinsiyetindekilerin Bağımlılık Durumu')
                        gender_position = st.session_state.gender
                        gender_option = st.session_state.gender_options[gender_position]

                        bar_chart_gender = data[data["Gender"] == gender_option]
                        bar_chart_gender = bar_chart_gender["ADDICTION_CATEGORY"].value_counts(normalize=True) * 100
                        bar_chart_gender = bar_chart_gender.rename({1: "Bağımlı", 0: "Bağımlı Değil"}).reset_index()
                        bar_chart_gender = bar_chart_gender.rename(
                            {"ADDICTION_CATEGORY": "Bağılımlılık Durumu", "proportion": "Bağımlılık Oranı"}, axis=1)

                        c = (
                            alt.Chart(bar_chart_gender)
                            .mark_bar(color='#ed145b', width=200)
                            .encode(x="Bağılımlılık Durumu", y="Bağımlılık Oranı")
                            .configure_axis(grid=False)
                        )
                        st.altair_chart(c, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown('### Sosyal Medya Kullanım İlişkisi Grafiği')

                if not data.empty:
                    if "prediction_value" in st.session_state:

                        data['Bağımlılık Durumu'] = data['ADDICTION_CATEGORY'].apply(
                            lambda x: 'Bağımlı' if x == 1 else 'Bağımlı Değil')
                        data = data.rename({'SOCIAL_MEDIA_COUNT_COMBINED_ORDINAL_Q2': 'Sosyal Medya Hesap Sayısı',
                                            "SOCIAL_MEDIA_COUNT_COMBINED_ORDINAL_Q3": "Aktif Kullanılan Sosyal Medya"},
                                           axis=1)
                        scatter_plot = (
                            alt.Chart(data)
                            .mark_circle()
                            .encode(
                                x='Sosyal Medya Hesap Sayısı',
                                y='Aktif Kullanılan Sosyal Medya',
                                color=alt.Color('Bağımlılık Durumu',
                                                scale=alt.Scale(domain=['Bağımlı', 'Bağımlı Değil'],
                                                                range=['#ed145b', '#1f77b4']),
                                                legend=alt.Legend(title="Bağımlılık Durumu"))
                            )
                        )
                        ordinal_q2 = len(st.session_state.ordinal_q2)
                        ordinal_q3 = len(st.session_state.ordinal_q3)
                        prediction = st.session_state.prediction_value

                        if prediction == 1 or prediction == 2:
                            result = 'Bağımlı'
                        else:
                            result = 'Bağımlı Değil'
                        highlight_point = pd.DataFrame(
                            {"Sosyal Medya Hesap Sayısı": ordinal_q2, "Aktif Kullanılan Sosyal Medya": ordinal_q3,
                             "Bağımlılık Durumu": result}, index=[0])

                        highlight_layer = (
                            alt.Chart(highlight_point)
                            .mark_circle(color='red', size=200)  # You can customize the color and size
                            .encode(
                                x='Sosyal Medya Hesap Sayısı',
                                y='Aktif Kullanılan Sosyal Medya',
                                tooltip=['Bağımlılık Durumu']
                            )
                        )

                        text_layer = (
                            alt.Chart(highlight_point)
                            .mark_text(align='right', baseline='bottom', dx=5, dy=-5, fontSize=12,
                                       color='black')  # Adjust the position and style
                            .encode(
                                x='Sosyal Medya Hesap Sayısı',
                                y='Aktif Kullanılan Sosyal Medya',
                                text='Bağımlılık Durumu'
                            )
                        )

                        combined_chart = scatter_plot + highlight_layer + text_layer

                        st.altair_chart(combined_chart, use_container_width=True)
        with col2:
            with st.container(border=True):
                st.markdown('### Sosyal Medya Kullanım Süresi Grafiği')
                if not data.empty:
                    data['Bağımlılık Durumu'] = data['ADDICTION_CATEGORY'].apply(
                        lambda x: 'Bağımlı' if x == 1 else 'Bağımlı Değil')

                    pie_chart_data = data["Average_time_you_spent_on_social_media_everyday"].value_counts(
                        normalize=True) * 100
                    pie_chart_data = pie_chart_data.reset_index().rename(
                        {"Average_time_you_spent_on_social_media_everyday": "Sosyal Medyada Kullanımı",
                         "proportion": "Sosyal Medyada Geçirilen Süre"}, axis=1)

                    pie_chart = alt.Chart(pie_chart_data).mark_arc(innerRadius=50).encode(
                        theta="Sosyal Medyada Geçirilen Süre",
                        color="Sosyal Medyada Kullanımı",
                    )
                    st.altair_chart(pie_chart, use_container_width=True)

    elif selected_option == translation["PAGE_OPTIONS_BLOG"]:
        with st.container(border=True):
            st.markdown(translation["BLOG_POST"])
            with st.expander(translation["SEE REFERENCES"]):
                st.markdown(translation["BLOG_REFERENCE"])

    elif selected_option == translation["PAGE_OPTIONS_OUR_GLOAL"]:
        def generate_goal_textbox(title, explanation):
            html_code = f"""
                <div style="text-align: justify-left; width: 900px; border-radius: 15px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25); overflow: hidden;">
                    <div style="background: linear-gradient(#f60089, #ff1744); padding: 1rem;">
                        <div style="font-size: 2rem; font-weight: 600; color: #fff;">{title}</div>
                    </div>
                    <div style="padding: 1.5rem;">
                        <p>{explanation}</p>
                    </div>
                </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)

        generate_goal_textbox(translation["OUR_GOAL_TITLE"], translation["OUR_GOAL_TEXT"])


    elif selected_option == translation["PAGE_OPTIONS_CONTACT_US"]:

        # Burada ikinci sayfa içeriğini oluşturabilirsiniz.
        # İletişim sayfasının içeriğini doldurabilirsiniz: formlar, iletişim bilgileri vb.

        st.subheader(translation["BLOG_PAGE_TITLE"])
        name = st.text_input(translation["CONTACT_PAGE_NAME"])
        email = st.text_input(translation["CONTACT_PAGE_MAIL"])
        message = st.text_area(translation["CONTACT_PAGE_MESSAGE"])

        if st.button(translation["CONTACT_PAGE_SUBMIT"]):
            # Kullanıcıdan alınan bilgilerle bir DataFrame oluştur.
            contact_data = {
                "Ad": [name],
                "E-posta": [email],
                "Mesaj": [message]
            }
            contact_df = pd.DataFrame(contact_data)

            # Teşekkür mesajını göster.
            st.success(translation["SURVEY_PLEASING"])

    elif selected_option == translation["PAGE_OPTIONS_DEVELOPERS"]:
        st.title(translation["DEVELOPERS_PAGE_TITLE"])

        def profile_card(name, username, bio, bio2, image_url, social_links):
            st.markdown(
                f"""
                <div style="text-align: center; width: 300px; border-radius: 15px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25); overflow: hidden;">
                    <div style="background: linear-gradient(#f60089, #ff1744); padding: 1rem;">
                        <img src="{image_url}" style="width: 150px; height: 150px; border-radius: 50%;" alt="Profile Picture" />
                    </div>
                    <div style="padding: 1.5rem;">
                        <p style="font-size: 1.5rem; font-weight: 600; text-transform: uppercase;">{name}</p>
                        <p style="opacity: 0.75; font-size:1.2rem;">{username}</p>
                        <div style="display: flex; justify-content: center;">
                            {"".join([f'<a href="{link}" style="padding: 0.5rem; font-size: 1.25rem; color: #000; text-decoration: none;"><img width="48" height="48" src="https://img.icons8.com/color/48/linkedin.png" alt="linkedin"/></a>' for icon, link in social_links.items()])}
                        </div>
                        <p>{bio}</p>
                        <p>{bio2}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container():
            col1, col2, col3 = st.columns(3, gap="large")
            with col1:
                profile_card(**translation["PROJECT_DEVELOPER_AYLIN"])
            with col2:
                profile_card(**translation["PROJECT_DEVELOPER_BUSE"])
            with col3:
                profile_card(**translation["PROJECT_DEVELOPER_CANMERT"])

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: #FFFFFF;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    

    /* Tüm widget'ların metin boyutlarını büyüt ve renklerini beyaz yap */
    .stTextInput, .stSelectbox, .stMultiselect, .stRadio, .stCheckbox, .stSlider, .stTextarea {{
        font-size: 20px !important; /* Font boyutunu daha büyük yap */
        color: white !important; /* Metin rengini beyaz yap */
    }}

    /* Widget başlıklarını büyüt ve renklerini beyaz yap */
    .stTextInput label, .stSelectbox label, .stMultiselect label, .stRadio label, .stCheckbox label, .stSlider label, .stTextarea label {{
        font-size: 20px !important; /* Font boyutunu daha büyük yap */
        color: black !important; /* Metin rengini beyaz yap */
        font-weight: bold !important; /* Yazı tipini kalın yap */
    }}

    /* Buton stilini ayarla */
    .stButton > button {{
        font-size: 20px !important; /* Font boyutunu daha büyük yap */
        font-weight: bold !important; /* Yazı tipini kalın yap */
        color: white !important; /* Metin rengini beyaz yap */
        background-color: #ed145b; /* Buton arka plan rengini değiştir */
    }}
    .st-emotion-cache-7ym5gk{{
        background-color: #ed145b; /* Buton arka plan rengini değiştir */
    }}
    
    /* Buton stilini ayarla */
    .st-dp  {{
        background-color: #ed145b; /* Buton arka plan rengini değiştir */
    }}
    
    

    /* Başlık rengini beyaz yap */
    h1 {{
        color: black !important; /* Metin rengini beyaz yap */
        /*font-size: 34px !important; /* Başlık font boyutunu büyüt */*/
    }}
    
    h2#b778660f {{
        color: #3EB489 !important; /* Metin rengini beyaz yap */
    }}

    /* Multiselect widget'larının metin renklerini beyaz yap */
    .css-1s2u09g-control, .css-1hwfws3 {{
        color: white;
    }}

    /* Seçeneklerin ve seçilen öğelerin metin renklerini beyaz yap */
    .css-12jo7m5, .css-14el2xx {{
        color: white !important;
    }}

    /* Multiselect widget'larının placeholder metin rengini beyaz yap */
    .css-1wa3eu0-placeholder {{
        color: white !important;
    }}

    /* Multiselect dropdown ok rengini beyaz yap */
    .css-1okebmr-indicatorSeparator, .css-tlfecz-indicatorContainer {{
        color: white !important;
    }}

    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
