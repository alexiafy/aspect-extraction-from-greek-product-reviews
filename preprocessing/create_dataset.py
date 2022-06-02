from preprocess import *
from equivalent_aspect_detection import *
from aspect_tagging import *


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 500)
# pd.set_option('display.expand_frame_repr', False)


FILENAME1 = 'target_sentiment_parts_1_2_3_GK_annotated'
FILENAME2 = 'target_sentiment_parts_1_2_3_GR_annotated'


def create_dataset(sentiment_extracted, punctuation_removed, stopwords_removed, username_replaced):
    """

    :param sentiment_extracted:
    :param punctuation_removed:
    :param stopwords_removed:
    :param username_replaced:
    :return:
    """


    # 1. Load data
    # --------------------------------
    annotator1 = pd.read_csv("../data/PALO/Final/" + FILENAME1 + ".csv")  # .head(500)
    annotator2 = pd.read_csv("../data/PALO/Final/" + FILENAME2 + ".csv")  # .head(500)

    df = annotator1.rename(columns={'text': 'text1', 'targets': 'targets1', 'sentiments': 'sentiments1'})
    df['text2'] = pd.Series(annotator2['text'])
    df['targets2'] = pd.Series(annotator2['targets'])
    df['sentiments2'] = pd.Series(annotator2['sentiments'])
    print('Initial number of rows:', len(df))



    # 2. Drop rows with NaNs or with different texts
    # --------------------------------
    # Drop rows with NaN values on target
    df = df.dropna(subset=['targets1',
                           'targets2'])  # drop NaN from only target columns todo change it if we want to consider sentiment too

    if sentiment_extracted:
        df = df.dropna(subset=['sentiments1', 'sentiments2'])
    print('Number of rows after removing NaN values', len(df))

    # Drop rows with different text between annotators
    df['text'] = [text1 if text1 == text2 else '' for text1, text2 in zip(df['text1'], df['text2'])]
    df = df[df['text'] != '']
    print('Number of rows after removing rows with different text between annotators', len(df))



    # 3. Preprocess text
    # --------------------------------
    df['text_preprocessed'] = df['text'].apply(basic_text_preprocess)

    if punctuation_removed:
        df['text_preprocessed'] = df['text_preprocessed'].apply(remove_punctuation)

    if stopwords_removed:
        df['text_preprocessed'] = df['text_preprocessed'].apply(remove_stopwords)



    # Targets -------------------------------------------
    # 4. Remove greek accents just in case one annotator includes them while the other doesn't
    # --------------------------------
    # Apply the replacement of greek accents in order to catch possible differences in accents between the annotators
    df['targets1'] = df['targets1'].apply(replace_greek_accents)
    df['targets2'] = df['targets2'].apply(replace_greek_accents)



    # 5. Keep only same targets
    # --------------------------------
    if sentiment_extracted:
        # extract sentiments and targets
        df = df.apply(get_targets_sentiments, axis=1)
    else:
        # create new column with the same targets from the two annotators
        df['targets'] = df.apply(lambda row: find_equivalent_targets(row['targets1'], row['targets2'], row['text']),
                                 axis=1)

    # drop rows with empty list on final 'targets' column, meaning with no same targets from the two annotators
    df = df[df['targets'].map(lambda d: len(d)) > 0]

    print('Number of rows after keeping only same targets and dropping rows that have null same targets:', len(df))



    # 6. Preprocess targets
    # --------------------------------
    df['targets_preprocessed'] = df['targets'].apply(preprocess_targets)

    if punctuation_removed:
        df['targets_preprocessed'] = df['targets_preprocessed'].apply(remove_punctuation_from_targets)

    if stopwords_removed:
        df['targets_preprocessed'] = df['targets_preprocessed'].apply(remove_stopwords_from_targets)

    # for index, row in df.iterrows():
    #    print(row['targets'], row['targets_preprocessed'])



    # 7. Replace username on text and targets
    # --------------------------------
    #if replace_mentions:
    #    # !! remove mentions from text if it does not exist in targets
    #    df['text_preprocessed'] = df.apply(lambda row: replace_mentions(row['text_preprocessed'], row['targets_preprocessed']),
    #                                   axis=1)

    if username_replaced:
        # !! remove usrnames from mentions in text if it does not exist in targets
            df['text_preprocessed'] = df.apply(lambda row: replace_usernames(row['text_preprocessed'], row['targets_preprocessed']), axis = 1)



    # 8.IOB & BIOES tagging
    # --------------------------------
    df['IOB1_tagging'] = df.apply(lambda row: get_IOB1_tagging(row['text_preprocessed'], row['targets_preprocessed']), axis=1)
    df['IOB2_tagging'] = df.apply(lambda row: get_IOB2_tagging(row['text_preprocessed'], row['targets_preprocessed']), axis=1)
    df['BIOES_tagging'] = df.apply(lambda row: get_BIOES_tagging(row['text_preprocessed'], row['targets_preprocessed']), axis=1)

    if sentiment_extracted:
        df['IOB1_sentiment_c3'] = df.apply(
            lambda row: get_IOB1_with_3_classes_sent(row['text_preprocessed'], row['targets_preprocessed'], row['sentiment']), axis=1)

        df['IOB1_sentiment_c5'] = df.apply(
            lambda row: get_IOB1_with_3_classes_sent(row['text_preprocessed'], row['targets_preprocessed'], row['sentiment']), axis=1)

        df['IOB2_sentiment_c3'] = df.apply(
            lambda row: get_IOB2_with_3_classes_sent(row['text_preprocessed'], row['targets_preprocessed'],
                                                           row['sentiment']), axis=1)

        df['IOB2_sentiment_c5'] = df.apply(
            lambda row: get_IOB2_with_3_classes_sent(row['text_preprocessed'], row['targets_preprocessed'],
                                                           row['sentiment']), axis=1)

        df['BIOES_sentiment_c3'] = df.apply(
            lambda row: get_BIOES_with_3_classes_sent(row['text_preprocessed'], row['targets_preprocessed'],
                                                           row['sentiment']), axis=1)



    # 9.POS tagging
    # --------------------------------
    df['POS_tagging'] = df.apply(lambda row: create_POS_tagging(row['text_preprocessed']), axis=1)




    # 10. Keep only targets with same sentiments
    # --------------------------------
    # Drop rows with different sentiment
    # different_sentiment_rows_counter = len(df[df['setniments1_tokenized'] != df['setniments2_tokenized']])
    # df = df[df['setniments1_tokenized'] == df['setniments2_tokenized']]
    # print('Number of rows after removing rows with different sentiments:', len(df))



    '''
    for idx, row in df.iterrows():
        print('\n')
        print(row['text'])
        print(row['text_preprocessed'])
        print(row['targets'])
        print(row['targets_preprocessed'])
    '''


    # FINALLY Write to csv
    # --------------------------------
    performed_actions = ['_punct_removed' if punctuation_removed else '',
                         '_stopwords_removed' if stopwords_removed else '',
                         '_mentions_replaced' if replace_mentions else '',
                         '_usrnames_replaced' if username_replaced else '']


    df.to_csv('../data/datasets/aspect_extraction_datasets/parts_1_2_3/ae_parts_1_2_3'
               + ''.join(performed_actions) + '.csv')


if __name__ == '__main__':

    #create_dataset(False, False, False, False)
    #create_dataset(False, False, False, True)

    create_dataset(False, False, True, False)
    create_dataset(False, False, True, True)

    create_dataset(False, True, False, False)
    create_dataset(False, True, False, True)

    create_dataset(False, True, True, False)
    create_dataset(False, True, True, True)

    """
    remove_stopwords_flag = False
    remove_punctuation_flag = False

    replace_username_flag = True
    replace_mentions_flag = False

    extract_sentiment = True


    # 1. Load data
    # --------------------------------
    annotator1 = pd.read_csv("../datasets/PALO/Final/" + FILENAME1 + ".csv")  # .head(500)
    annotator2 = pd.read_csv("../datasets/PALO/Final/" + FILENAME2 + ".csv")  # .head(500)

    df = annotator1.rename(columns={'text': 'text1', 'targets': 'targets1', 'sentiments': 'sentiments1'})
    df['text2'] = pd.Series(annotator2['text'])
    df['targets2'] = pd.Series(annotator2['targets'])
    df['sentiments2'] = pd.Series(annotator2['sentiments'])
    print('Initial number of rows:', len(df))



    # 2. Drop rows with NaNs or with different texts
    # --------------------------------
    # Drop rows with NaN values on target
    df = df.dropna(subset=['targets1',
                           'targets2'])  # drop NaN from only target columns todo change it if we want to consider sentiment too

    if extract_sentiment:
        df = df.dropna(subset=['sentiments1', 'sentiments2'])
    print('Number of rows after removing NaN values', len(df))

    # Drop rows with different text between annotators
    df['text'] = [text1 if text1 == text2 else '' for text1, text2 in zip(df['text1'], df['text2'])]
    df = df[df['text'] != '']
    print('Number of rows after removing rows with different text between annotators', len(df))




    # 3. Preprocess text
    # --------------------------------
    df['text_preprocessed'] = df['text'].apply(preprocess_text)

    if remove_punctuation_flag:
        df['text_preprocessed'] = df['text_preprocessed'].apply(remove_punctuation)

    if remove_stopwords_flag:
        df['text_preprocessed'] = df['text_preprocessed'].apply(remove_stopwords)




    # Targets -------------------------------------------
    # 4. Remove greek accents just in case one annotator includes them while the other doesn't
    # --------------------------------
    # Apply the replacement of greek accents in order to catch possible differences in accents between the annotators
    df['targets1'] = df['targets1'].apply(replace_greek_accents)
    df['targets2'] = df['targets2'].apply(replace_greek_accents)




    # 5. Keep only same targets
    # --------------------------------

    if extract_sentiment:
        # extract sentiments and targets
        df = df.apply(get_targets_sentiments, axis=1)
    else:
        # create new column with the same targets from the two annotators
        df['targets'] = df.apply(lambda row: find_equivalent_targets(row['targets1'], row['targets2'], row['text']), axis=1)


    # drop rows with empty list on final 'targets' column, meaning with no same targets from the two annotators
    df = df[df['targets'].map(lambda d: len(d)) > 0]

    print('Number of rows after keeping only same targets and dropping rows that have null same targets:', len(df))




    # 6. Preprocess targets
    # --------------------------------
    df['targets_preprocessed'] = df['targets'].apply(preprocess_targets)



    # !! separate numbers from chars only if the word does not appear in targets
    # df['text_preprocessed'] = df.apply(
        # lambda row: separate_numbers_from_chars(row['text_preprocessed'], row['targets_preprocessed']),
        # axis=1)


    if remove_punctuation_flag:
        df['targets_preprocessed'] = df['targets_preprocessed'].apply(remove_punctuation_targets)

    if remove_stopwords_flag:
        df['targets_preprocessed'] = df['targets_preprocessed'].apply(remove_stopwords_from_targets)

    # for index, row in df.iterrows():
    #    print(row['targets'], row['targets_preprocessed'])



    if replace_mentions_flag:
        # !! remove mentions from text if it does not exist in targets
        df['text_preprocessed'] = df.apply(lambda row: replace_mentions(row['text_preprocessed'], row['targets_preprocessed']),
                                       axis=1)
    elif replace_username_flag:
        # !! remove usrnames from mentions in text if it does not exist in targets
        df['text_preprocessed'] = df.apply(
            lambda row: replace_usernames(row['text_preprocessed'], row['targets_preprocessed']),
            axis=1)


    # !! replace greelish (in comment because the number of greeklish words is too small, maybe later...)
    # df = df.apply(replace_greeklish, axis=1)





    # 8.IOB & BIOES tagging
    # --------------------------------
    df['IOB1_tagging'] = df.apply(lambda row: get_IOB1_tagging(row['text_preprocessed'], row['targets_preprocessed']), axis=1)
    df['IOB2_tagging'] = df.apply(lambda row: get_IOB2_tagging(row['text_preprocessed'], row['targets_preprocessed']), axis=1)
    df['BIOES_tagging'] = df.apply(lambda row: get_BIOES_tagging(row['text_preprocessed'], row['targets_preprocessed']), axis=1)

    if extract_sentiment:
        df['IOB1_sentiment_c3'] = df.apply(
            lambda row: get_IOB1_with_sentiments_3_classes(row['text_preprocessed'], row['targets_preprocessed'], row['sentiment']), axis=1)

        df['IOB1_sentiment_c5'] = df.apply(
            lambda row: get_IOB1_with_sentiments_5_classes(row['text_preprocessed'], row['targets_preprocessed'], row['sentiment']), axis=1)

        df['IOB2_sentiment_c3'] = df.apply(
            lambda row: get_IOB2_with_sentiments_3_classes(row['text_preprocessed'], row['targets_preprocessed'],
                                                           row['sentiment']), axis=1)

        df['IOB2_sentiment_c5'] = df.apply(
            lambda row: get_IOB2_with_sentiments_5_classes(row['text_preprocessed'], row['targets_preprocessed'],
                                                           row['sentiment']), axis=1)

        df['BIOES_sentiment_c3'] = df.apply(
            lambda row: get_BIOES_with_sentiments_3_classes(row['text_preprocessed'], row['targets_preprocessed'],
                                                           row['sentiment']), axis=1)



    # 9.POS tagging
    # --------------------------------
    df['POS_tagging'] = df.apply(lambda row: create_POS_tagging(row['text_preprocessed']), axis=1)





    # 10. Keep only targets with same sentiments
    # --------------------------------
    # Drop rows with different sentiment
    # different_sentiment_rows_counter = len(df[df['setniments1_tokenized'] != df['setniments2_tokenized']])
    # df = df[df['setniments1_tokenized'] == df['setniments2_tokenized']]
    # print('Number of rows after removing rows with different sentiments:', len(df))




    '''
    for idx, row in df.iterrows():
        print('\n')
        print(row['text'])
        print(row['text_preprocessed'])
        print(row['targets'])
        print(row['targets_preprocessed'])
    '''


    # FINALLY Write to csv
    # --------------------------------
    performed_actions = ['_punct_removed' if remove_punctuation_flag else '',
                         '_stopwords_removed' if remove_stopwords_flag else '',
                         '_mentions_replaced' if replace_mentions_flag else '',
                         '_usrnames_replaced' if replace_username_flag else '']


    df.to_csv('../datasets/aspect_extraction_datasets/parts_1_2_3/absa_parts_1_2_3_withBIOES'
               + ''.join(performed_actions) + '.csv')

    """



