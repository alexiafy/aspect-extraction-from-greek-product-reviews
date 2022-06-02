import regex as re
from nltk.corpus import stopwords
import spacy
import itertools
import emoji
from nltk.corpus import words
import string


nlp_spacy = spacy.load("el_core_news_lg")


def replace_greek_accents(text):
    """
    Removes the accents/tones from greek language

    :param text: The original text
    :return: The text with removed accents
    """

    text = text.replace("ά", "α")
    text = text.replace("έ", "ε")
    text = text.replace("ύ", "υ")
    text = text.replace("ή", "η")
    text = text.replace("ώ", "ω")
    text = text.replace("ί", "ι")
    text = text.replace("ό", "ο")

    text = text.replace("Ά", "Α")
    text = text.replace("Έ", "Ε")
    text = text.replace("Ύ", "Υ")
    text = text.replace("Ή", "Η")
    text = text.replace("Ώ", "Ω")
    text = text.replace("Ί", "Ι")
    text = text.replace("Ό", "Ο")

    text = text.replace("ϊ", "ι")
    text = text.replace("ΐ", "ι")
    text = text.replace("ϋ", "υ")
    text = text.replace("ΰ", "υ")

    text = text.replace("Ϊ", "Ι")
    text = text.replace("Ϋ", "υ")

    return text


def replace_mentions(text, aspects):
    """
    Replace retweet mentions with MENTION
    e.g. RT @xstefanou -> RT MENTION
         RT @caftis    -> RT MENTION
         @Valkyrie_Gr   -> MENTION

    :param text: The original text
    :param aspects: The aspects detected in this text
    :return: The text with replaced retweet mentions
    """

    text = ' '.join(text)  # join strings to apply the following preprocessing steps
    result = re.findall("@ ([A-Za-z]+[A-Za-z0-9-_]+)", text)  # ((?<=@ )(\w){1,15})  ((?<=@ )([\w\_\.]+))  ((?<=@ )([\w]+))

    for res in result:

        mentions_from_targets_qty = 0  # number of mentions that appear in aspects

        for target in aspects:
            if ('@ ' + res) in target or res in target:
                mentions_from_targets_qty += 1

        if mentions_from_targets_qty == 0:
            text = text.replace('@ ' + res, 'MENTION')

    text = [x.strip() for x in text.split()]
    return text


def replace_usernames(text, targets):
    """
    Replace retweet mentions with @ USERNAME
    E.g. RT @ xstefanou -> RT @ USERNAME
         RT @ caftis    -> RT @ USERNAME
         @Valkyrie_Gr   -> @ USERNAME

    :param text: The original text
    :param targets: The aspects detected in this text
    :return: The text with replaced mentions
    """

    text = ' '.join(text)  # join strings to apply the following preprocessing steps
    result = re.findall("@ ([A-Za-z]+[A-Za-z0-9-_]+)", text)  # ((?<=@ )(\w){1,15})  ((?<=@ )([\w\_\.]+))  ((?<=@ )([\w]+))

    for res in result:

        usernames_on_aspects_qty = 0  # number of usernames that appear in aspects

        for target in targets:
            if ('@ ' + res) in target or res in target:
                usernames_on_aspects_qty += 1

        if usernames_on_aspects_qty == 0:
            text = text.replace('@ ' + res, '@ USERNAME')

    text = [x.strip() for x in text.split()]
    return text


def remove_punctuation(text):
    """
    Removes punctuation from text

    :param text: The original text
    :return: The text with removed punctuation
    """

    text = ' '.join(text)  # join strings to apply the following preprocessing steps


    # NOT NEEDED ANYMORE
    # replace "." with space if it appears between characters
    # e.g. ρε...μονο -> ρε μονο
    #      days....τωρα -> days τωρα
    # regex_dots_between_chars_str = '(?<=[a-zA-Zα-ωΑ-Ω])\.+(?=[a-zA-Zα-ωΑ-Ω])'
    # text = re.sub(regex_dots_between_chars_str, ', ', text)


    # remove all special characters except "," "'", "&", "%" and "." because they exist in targets
    # e.g. l'Oreal, Bailey's, what's up
    #      box.gr
    #      e-food, Coca-Cola HBC
    #      TOTAL 0 %
    #      Head & Shoulders
    regex_str = '[^,&%.\'a-zA-Z0-9α-ωΑ-Ωάέύήώίόϊΐϋΰ \n]'
    text = re.sub(regex_str, '', text)

    # remove "," and "." only when they do not appear between numbers
    # e.g. 5,8, 6.2
    regex_comma_on_digits_only_str = '(?<=\D)[.,]|[.,](?=\D)'
    text = re.sub(regex_comma_on_digits_only_str, '', text)

    # tokenize text
    text = [x.strip() for x in text.split()]

    return text


def remove_emoji(text):
    """
    Remove emojis

    :param text: The original text
    :return: The text with removed emojis
    """
    return emoji.get_emoji_regexp().sub(u' ', text)  # text.encode('ascii', 'ignore').decode('ascii')  #emoji.get_emoji_regexp().sub(u'', text)


def clean_and_tokenize(text):
    """
    Remove accents/tons and tokenize

    :param text: The original text
    :return: The tokenized text with removed tons
    """
    text = replace_greek_accents(text)  # remove greek accents/tons
    text_tokenized = [x.strip() for x in text.split(",")]  # tokenize

    return text_tokenized


def get_stopwords():
    """
    Returns a concatanated list of greek and english stopwords

    :return: List of stopwords
    """

    gr_stopwords = set(stopwords.words('greek'))
    en_stopwords = set(stopwords.words('english'))

    with open("stopword_files/greek_stopwords.txt", 'r',
              encoding='UTF-8') as file:  # read file which contains additional greek stopword_files
        additional_gr_stopwords = file.readlines()
        additional_gr_stopwords = [line.rstrip() for line in additional_gr_stopwords]

    gr_stopwords = [replace_greek_accents(stopword) for stopword in gr_stopwords]  # remove greek accents/tons
    additional_gr_stopwords = [replace_greek_accents(stopword) for stopword in
                               additional_gr_stopwords]  # remove greek accents/tons

    all_stopwords = list(itertools.chain(gr_stopwords, additional_gr_stopwords, en_stopwords))

    stopwords_contained_in_targets = ['and', 'up', 'by', 'κατ', 'αν',
                                      'my', 'what', 's', 'i', 'with', 'me',
                                      'της', 'it', 'τι', 'm', 'του']  # stopword_files that are contained in targets and should be excluded from the list
    all_stopwords = [stopword for stopword in all_stopwords if stopword not in stopwords_contained_in_targets]

    return all_stopwords


def remove_stopwords(text):
    """
    Removes the stopwords from the text

    :param text: The original text
    :return: The text with removed stopwords
    """
    stopwords_list = get_stopwords()
    return [word for word in text if word.lower() not in stopwords_list]


def convert_to_lower(text):
    """
    Converts text to lower cased

    :param text: The original text
    :return: The lower-cased text
    """
    return [word.lower() for word in text]


def basic_text_preprocess(text):
    """
    Applies the basic preprocessing steps including:
    Accent removal
    Url replacement
    Emoji removal
    Char separation
    Whitespace removal

    :param text: The original text
    :return: The preprocessed text
    """

    # print('\n', text)


    # remove greek accents/tons
    text = replace_greek_accents(text)


    # convert URLs to 'hprlnk'
    regex_url_str = '(https?://[^\s]+)'
    text = re.sub(regex_url_str, 'hprlnk', text)


    # remove emojis
    text = remove_emoji(text)


    # Tokenize in all punctuation marks.
    # (When symbols such as -,?, _, / are removed, the words that appear before and after it are concatenated.
    # Float numbers (e.g. 0,24) are not affected
    #      initial       unwanted   wanted
    # e.g. nova?η    ->  novaη  ->  nova η
    #      οτε/δεη ->  οτεδεη ->  οτε / δεη

    # add space between comma (,) and dot (.) only if
    # they do not appear in a float/decimal
    text = re.sub(r'([,.])(?![0-9])', r' \1 ', text)

    # add space between character and dot or comma
    # e.g. ΑΤΤΙΚΗ ΟΔΟΣ,1992 -> ΑΤΤΙΚΗ ΟΔΟΣ , 1992
    text = re.sub(r'([a-zA-Zα-ωΑ-Ω])([,.])', r'\1 \2 ', text)
    text = re.sub(r'([!?;`¨“”\-%@#&\*\’΄:\"\'\(\)\[\]/»«<>\+\$])', r' \1 ', text)  # add space between punctuation


    # replace … with ...
    # e.g. πες…Ολυμπιακός
    #      Ολυμπιακή…
    text = text.replace('…', ' . . . ')


    # remove whitespace
    text = text.strip().rstrip()  # remove whitespace from start and end
    text = re.sub('\s+', ' ', text)  # remove duplicate spaces in the text
    # print(text)


    # NOT NEEDED ANYMORE
    # not needed after adding the "first_tokenization" step
    # replace "," with ", " if it appears between characters or character from one side and number from the other side
    # e.g. καλτες,παπουτσια,φανελα -> καλτες παπουτσια φανελα
    #      Πιπη,αστα -> Πιπη αστα
    # regex_comma_between_chars_str = '(?<=[a-zA-Zα-ωΑ-Ω]),(?=[a-zA-Zα-ωΑ-Ω])'
    # text = re.sub(regex_comma_between_chars_str, ', ', text)
    # print(text)


    # convert to lower is commented because CRF has feature the converts the words to lower
    # text_lowered = convert_to_lower(text_tokenized)


    # tokenize text
    text = [x.strip() for x in text.split()]

    # print(text)
    return text


def create_POS_tagging(text):
    """
    Creates the POS tagging

    :param text: The original text
    :return: The POS tagging
    """
    text = ' '.join(text)
    pos_tag = []

    doc = nlp_spacy(text)

    for token in doc:
        # print(token.text, token.pos_, token.lemma_)
        pos_tag.append((str(token), token.pos_))

    return pos_tag


def preprocess_targets(targets):
    # print('\n', targets)
    targets_preprocessed_list = []
    # targets = [x.strip() for x in targets.split(',')]  # split on comma ","

    for target in targets:
        target_preprocessed = basic_text_preprocess(target)
        target_preprocessed = ' '.join(target_preprocessed)  # join strings
        targets_preprocessed_list.append(target_preprocessed)

    # print(targets_preprocessed_list)
    return targets_preprocessed_list



def remove_stopwords_from_targets(targets):
    # print('\n', targets)
    targets_stopwords_removed_list = []

    for target in targets:
        targets_splitted = [x.strip() for x in target.split(' ')]
        target_stopwords_removed = remove_stopwords(targets_splitted)
        target_stopwords_removed = ' '.join(target_stopwords_removed)  # join strings
        targets_stopwords_removed_list.append(target_stopwords_removed)

    # print(targets_preprocessed_list)
    return targets_stopwords_removed_list


def remove_punctuation_from_targets(targets):
    # print('\n', targets)
    targets_extensive_preprocessing_list = []

    for target in targets:
        targets_splitted = [x.strip() for x in target.split(' ')]
        target_extensive_preprocessing = remove_punctuation(targets_splitted)
        target_extensive_preprocessing = ' '.join(target_extensive_preprocessing)  # join strings
        targets_extensive_preprocessing_list.append(target_extensive_preprocessing)

    # print(targets_preprocessed_list)
    return targets_extensive_preprocessing_list




if __name__ == '__main__':
    print("hello world")

    preprocess_targets('@COSMOTE')
    preprocess_targets('APPALOOSA RESTAURANT - BAR ')
    preprocess_targets('@PlaisioOfficial')
    preprocess_targets('@COSMOTE, Lynne')
    preprocess_targets('')





