import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 800)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)


def get_IOB1_tagging(text, targets):
    """
    This function extracts the IOB1 tags based on the following assumptions.
    1st Assumption: the order of the targets corresponds to the order of their appearance in the text
    2nd Assumption: if the same word is detected as target n times in the text, then it should
                    be included in the list of targets n times.

    In IOB1, B- is only used to separate two adjacent entities of the same type.
    The I- prefix before a tag indicates that the tag is inside a chunk.
    An O tag indicates that a token belongs to no chunk.
    The B- prefix before a tag indicates that the tag is the beginning of a chunk that immediately
    follows another chunk without O tags between them.

    :param text: The given text
    :param targets: The targets that should be detected in the text
    :return: IOB tags
    """

    IOB_tag = []
    target_index = 0
    idx = 0

    while idx < len(text):  # while loop that iterates through the words of the text

        word_of_text = text[idx]
        # print(target_index, len(targets))

        # if there are no other targets, then tag the rest of the words as 'O'
        # without checking anything in the rest function
        if target_index >= len(targets):
            IOB_tag.append((word_of_text, 'O'))
            idx += 1
            continue

        # get the targets of the current iteration
        current_targets = targets[target_index]

        # split on space removing whitespaces in order to capture the multi-word targets
        current_target_words = [x.strip() for x in current_targets.split(' ')]
        # print(word_of_text, current_targets)

        # if target is constituted by one word only, then it is tagged as 'B'
        if len(current_target_words) == 1:
            if word_of_text == current_target_words[0]\
                    or word_of_text == '@' + current_target_words[0]\
                    or word_of_text == '#' + current_target_words[0]:

                # check if the previous word was an aspect,
                # ensuring that a previous word exists (this is not the first word of the text)
                if IOB_tag and (IOB_tag[len(IOB_tag)-1][1] == 'I' or IOB_tag[len(IOB_tag)-1][1] == 'B'):
                    IOB_tag.append((word_of_text, 'B'))
                else:
                    IOB_tag.append((word_of_text, 'I'))

                target_index += 1  # continue with the next target
                idx += 1  # continue with the next word of the text
                continue
            else:
                IOB_tag.append((word_of_text, 'O'))
                idx += 1  # continue with the next word of the text
                continue

        else:  # if we have a multi-word target
            # print("multi word")
            sequence_of_found_target_words = []  # list of the sequence of the words found that are targets
            entire_sequence_of_words_same = True
            text_word_index = idx

            for target_word in current_target_words:
                if target_word == text[text_word_index]:
                    sequence_of_found_target_words.append(text[text_word_index])
                    text_word_index += 1
                else:
                    entire_sequence_of_words_same = False
                    break

            if entire_sequence_of_words_same:
                # check if the previous word was an aspect,
                # ensuring that a previous word exists (this is not the first word of the text)
                if IOB_tag and (IOB_tag[len(IOB_tag) - 1][1] == 'I' or IOB_tag[len(IOB_tag) - 1][1] == 'B'):
                    IOB_tag.append((word_of_text, 'B'))
                else:
                    IOB_tag.append((word_of_text, 'I'))

                # each word from the one one to the last one is tagged with 'I'
                for sequence_index in range(1, len(sequence_of_found_target_words)):
                    IOB_tag.append((sequence_of_found_target_words[sequence_index], 'I'))

                # continue with the word that follows the last word that was detected as target word
                idx += len(sequence_of_found_target_words)
                target_index += 1  # continue with the next target

            else:
                # if the target word is not the current word of the text, continue with the next word of the text
                IOB_tag.append((word_of_text, 'O'))
                idx += 1

    # print(IOB_tag)
    return IOB_tag


def get_IOB2_tagging(text, targets):
    """
    This function extracts the IOB2 tags based on the following assumptions.
    1st Assumption: the order of the targets in the "targets" column corresponds to the order of their appearance in the text
    2nd Assumption: if the same word is detected as target n times in the text, then it should
                    be included in the list of targets n times.
    :param text: the given text
    :param targets: the targets that should be detected in the text
    :return: IOB tags
    """

    IOB_tag = []
    target_index = 0
    idx = 0

    while idx < len(text):  # while loop that iterates through the words of the text

        word_of_text = text[idx]

        # if there are no other targets,
        # then tag the rest of the words as 'O' without checking anything in the rest function
        if target_index >= len(targets):
            IOB_tag.append((word_of_text, 'O'))
            idx += 1
            continue

        current_targets = targets[target_index]  # get the targets of the current iteration

        # split on space removing whitespaces in order to capture the multi-word targets
        current_target_words = [x.strip() for x in current_targets.split(' ')]
        # print(word_of_text, current_targets)

        if len(current_target_words) == 1:  # if target is constituted by one word only, then it is tagged as 'B'
            if word_of_text == current_target_words[0] \
                    or word_of_text == '@' + current_target_words[0]\
                    or word_of_text == '#' + current_target_words[0]:

                IOB_tag.append((word_of_text, 'B'))
                target_index += 1  # continue with the next target
                idx += 1  # continue with the next word of the text
                continue

            else:
                IOB_tag.append((word_of_text, 'O'))
                idx += 1  # continue with the next word of the text
                continue

        else:  # if we have a multi-word target

            sequence_of_found_target_words = []  # list of the sequence of the words found that are targets
            entire_sequence_of_words_same = True
            text_word_index = idx

            for target_word in current_target_words:
                if target_word == text[text_word_index]:
                    sequence_of_found_target_words.append(text[text_word_index])
                    text_word_index += 1
                else:
                    entire_sequence_of_words_same = False
                    break

            if entire_sequence_of_words_same:

                IOB_tag.append((sequence_of_found_target_words[0], 'B'))  # tag the first word with 'B'

                # each word from the second one to the last one is tagged with 'I'
                for sequence_index in range(1, len(sequence_of_found_target_words)):
                    IOB_tag.append((sequence_of_found_target_words[sequence_index], 'I'))

                # continue with the word that follows the last word that was detected as target word
                idx += len(sequence_of_found_target_words)
                target_index += 1  # continue with the next target

            else:  # if the target word is not the current word of the text, continue with the next word of the text
                IOB_tag.append((word_of_text, 'O'))
                idx += 1

    # print(IOB_tag)
    return IOB_tag


def get_BIOES_tagging(text, targets):
    """
    This function extracts the BIOES tags based on the following assumptions.
    1st Assumption: the order of the targets corresponds to the order of their appearance in the text
    2nd Assumption: if the same word is detected as target n times in the text, then it should
                    be included in the list of targets n times.

    In IOB2 format the B- tag is used in the beginning of every chunk

    :param text: the given text
    :param targets: the targets that should be detected in the text
    :return: BIOES tags
    """

    BIOES_tag = []
    target_index = 0
    idx = 0

    while idx < len(text):  # while loop that iterates through the words of the text

        word_of_text = text[idx]

        # if there are no other targets,
        # then tag the rest of the words as 'O' without checking anything in the rest function
        if target_index >= len(targets):
            BIOES_tag.append((word_of_text, 'O'))
            idx += 1
            continue

        current_targets = targets[target_index]  # get the targets of the current iteration

        # split on space removing whitespaces in order to capture the multi-word targets
        current_target_words = [x.strip() for x in current_targets.split(' ')]
        # print(word_of_text, current_targets)

        if len(current_target_words) == 1:  # if target is constituted by one word only, then it is tagged as 'B'
            if word_of_text == current_target_words[0]\
                    or word_of_text == '@' + current_target_words[0]\
                    or word_of_text == '#' + current_target_words[0]:

                BIOES_tag.append((word_of_text, 'S'))
                target_index += 1  # continue with the next target
                idx += 1  # continue with the next word of the text
                continue

            else:
                BIOES_tag.append((word_of_text, 'O'))
                idx += 1  # continue with the next word of the text
                continue

        else:  # if we have a multi-word target

            sequence_of_found_target_words = []  # list of the sequence of the words found that are targets
            entire_sequence_of_words_same = True
            text_word_index = idx

            for target_word in current_target_words:
                # if the target word is found in the text,
                # append it to the list and continue with the next word of the target
                if target_word == text[text_word_index]:
                    sequence_of_found_target_words.append(text[text_word_index])
                    text_word_index += 1
                else:
                    entire_sequence_of_words_same = False
                    break
            # print(sequence_of_found_targets)

            if entire_sequence_of_words_same:

                BIOES_tag.append((sequence_of_found_target_words[0], 'B'))  # tag the first word with 'B'

                # each word from the second one to the last one is tagged with 'I'
                for sequence_index in range(1, len(sequence_of_found_target_words)-1):
                    BIOES_tag.append((sequence_of_found_target_words[sequence_index], 'I'))

                BIOES_tag.append((sequence_of_found_target_words[len(sequence_of_found_target_words)-1], 'E'))

                # continue with the word that follows the last word that was detected as target word
                idx += len(sequence_of_found_target_words)
                target_index += 1  # continue with the next target

            else:  # if the target word is not the current word of the text, continue with the next word of the text
                BIOES_tag.append((word_of_text, 'O'))
                idx += 1

    # print(IOB_tag)
    return BIOES_tag


def get_IOB1_with_5_classes_sent(text, targets, sentiments):
    """
    This function extracts the IOB1 tags combined with sentiment tags
    with 5 classes ({VNEG, NEG, NEU, POS, VPOS})

    :param text: The given text
    :param targets: The targets that should be detected in the text
    :return: IOB tags with 5 class sentiment tags
    """
    # print('\n')
    # print(targets, sentiments)

    IOB_tag = []
    target_index = 0
    idx = 0

    while idx < len(text):  # while loop that iterates through the words of the text

        word_of_text = text[idx]
        # print(target_index, len(targets))

        # if there are no other targets, then tag the rest of the words as 'O'
        # without checking anything in the rest function
        if target_index >= len(targets):
            IOB_tag.append((word_of_text, 'O'))
            idx += 1
            continue

        # get the targets of the current iteration
        current_targets = targets[target_index]
        current_sentiment = sentiments[target_index]

        if current_sentiment == '-2':
            sent_tag = '-VNEG'
        elif current_sentiment == '-1':
            sent_tag = '-NEG'
        elif current_sentiment == '0':
            sent_tag = '-NEU'
        elif current_sentiment == '1':
            sent_tag = '-POS'
        elif current_sentiment == '2':
            sent_tag = '-VPOS'
        else:
            sent_tag = ''


        # split on space removing whitespaces in order to capture the multi-word targets
        current_target_words = [x.strip() for x in current_targets.split(' ')]
        # print(word_of_text, current_targets)

        # if target is constituted by one word only, then it is tagged as 'B'
        if len(current_target_words) == 1:
            if word_of_text == current_target_words[0]\
                    or word_of_text == '@' + current_target_words[0]\
                    or word_of_text == '#' + current_target_words[0]:

                # check if the previous word was an aspect,
                # ensuring that a previous word exists (this is not the first word of the text)
                if IOB_tag and (IOB_tag[len(IOB_tag)-1][1].startswith('I') or IOB_tag[len(IOB_tag)-1][1].startswith('B')):
                    IOB_tag.append((word_of_text, 'B' + sent_tag))
                else:
                    IOB_tag.append((word_of_text, 'I' + sent_tag))

                target_index += 1  # continue with the next target
                idx += 1  # continue with the next word of the text
                continue
            else:
                IOB_tag.append((word_of_text, 'O'))
                idx += 1  # continue with the next word of the text
                continue

        else:  # if we have a multi-word target
            # print("multi word")
            sequence_of_found_target_words = []  # list of the sequence of the words found that are targets
            entire_sequence_of_words_same = True
            text_word_index = idx

            for target_word in current_target_words:
                if target_word == text[text_word_index]:
                    sequence_of_found_target_words.append(text[text_word_index])
                    text_word_index += 1
                else:
                    entire_sequence_of_words_same = False
                    break

            if entire_sequence_of_words_same:
                # check if the previous word was an aspect,
                # ensuring that a previous word exists (this is not the first word of the text)
                if IOB_tag and (IOB_tag[len(IOB_tag) - 1][1].startswith('I') or IOB_tag[len(IOB_tag) - 1][1].startswith('B')):
                    IOB_tag.append((word_of_text, 'B' + sent_tag))
                else:
                    IOB_tag.append((word_of_text, 'I' + sent_tag))

                # each word from the one one to the last one is tagged with 'I'
                for sequence_index in range(1, len(sequence_of_found_target_words)):
                    IOB_tag.append((sequence_of_found_target_words[sequence_index], 'I' + sent_tag))

                # continue with the word that follows the last word that was detected as target word
                idx += len(sequence_of_found_target_words)
                target_index += 1  # continue with the next target

            else:
                # if the target word is not the current word of the text, continue with the next word of the text
                IOB_tag.append((word_of_text, 'O'))
                idx += 1

    # print(IOB_tag)
    return IOB_tag


def get_IOB1_with_3_classes_sent(text, targets, sentiments):
    """
    This function extracts the IOB1 tags combined with sentiment tags
    with 3 classes ({NEG, NEU, POS})

    :param text: The given text
    :param targets: The targets that should be detected in the text
    :return: IOB tags with 3 class sentiment tags
    """
    # print('\n')
    # print(targets, sentiments)

    IOB_tag = []
    target_index = 0
    idx = 0

    while idx < len(text):  # while loop that iterates through the words of the text

        word_of_text = text[idx]
        # print(target_index, len(targets))

        # if there are no other targets, then tag the rest of the words as 'O'
        # without checking anything in the rest function
        if target_index >= len(targets):
            IOB_tag.append((word_of_text, 'O'))
            idx += 1
            continue

        # get the targets of the current iteration
        current_targets = targets[target_index]
        current_sentiment = sentiments[target_index]

        if current_sentiment == '-2' or current_sentiment == '-1':
            sent_tag = '-NEG'
        elif current_sentiment == '0':
            sent_tag = '-NEU'
        elif current_sentiment == '1' or current_sentiment == '2':
            sent_tag = '-POS'
        else:
            sent_tag = ''


        # split on space removing whitespaces in order to capture the multi-word targets
        current_target_words = [x.strip() for x in current_targets.split(' ')]
        # print(word_of_text, current_targets)

        # if target is constituted by one word only, then it is tagged as 'B'
        if len(current_target_words) == 1:
            if word_of_text == current_target_words[0]\
                    or word_of_text == '@' + current_target_words[0]\
                    or word_of_text == '#' + current_target_words[0]:

                # check if the previous word was an aspect,
                # ensuring that a previous word exists (this is not the first word of the text)
                if IOB_tag and (IOB_tag[len(IOB_tag)-1][1].startswith('I') or IOB_tag[len(IOB_tag)-1][1].startswith('B')):
                    IOB_tag.append((word_of_text, 'B' + sent_tag))
                else:
                    IOB_tag.append((word_of_text, 'I' + sent_tag))

                target_index += 1  # continue with the next target
                idx += 1  # continue with the next word of the text
                continue
            else:
                IOB_tag.append((word_of_text, 'O'))
                idx += 1  # continue with the next word of the text
                continue

        else:  # if we have a multi-word target
            # print("multi word")
            sequence_of_found_target_words = []  # list of the sequence of the words found that are targets
            entire_sequence_of_words_same = True
            text_word_index = idx

            for target_word in current_target_words:
                if target_word == text[text_word_index]:
                    sequence_of_found_target_words.append(text[text_word_index])
                    text_word_index += 1
                else:
                    entire_sequence_of_words_same = False
                    break

            if entire_sequence_of_words_same:
                # check if the previous word was an aspect,
                # ensuring that a previous word exists (this is not the first word of the text)
                if IOB_tag and (IOB_tag[len(IOB_tag) - 1][1].startswith('I') or IOB_tag[len(IOB_tag) - 1][1].startswith('B')):
                    IOB_tag.append((word_of_text, 'B' + sent_tag))
                else:
                    IOB_tag.append((word_of_text, 'I' + sent_tag))

                # each word from the one one to the last one is tagged with 'I'
                for sequence_index in range(1, len(sequence_of_found_target_words)):
                    IOB_tag.append((sequence_of_found_target_words[sequence_index], 'I' + sent_tag))

                # continue with the word that follows the last word that was detected as target word
                idx += len(sequence_of_found_target_words)
                target_index += 1  # continue with the next target

            else:
                # if the target word is not the current word of the text, continue with the next word of the text
                IOB_tag.append((word_of_text, 'O'))
                idx += 1

    # print(IOB_tag)
    return IOB_tag


def get_IOB2_with_5_classes_sent(text, targets, sentiments):
    """
    This function extracts the IOB2 tags combined with sentiment tags
    with 5 classes ({VNEG, NEG, NEU, POS, VPOS})

    :param text: The given text
    :param targets: The targets that should be detected in the text
    :return: IOB tags with 5 class sentiment tags
    """
    # print('\n')
    # print(targets, sentiments)

    IOB_tag = []
    target_index = 0
    idx = 0

    while idx < len(text):  # while loop that iterates through the words of the text

        word_of_text = text[idx]
        # print(target_index, len(targets))

        # if there are no other targets, then tag the rest of the words as 'O'
        # without checking anything in the rest function
        if target_index >= len(targets):
            IOB_tag.append((word_of_text, 'O'))
            idx += 1
            continue

        # get the targets of the current iteration
        current_targets = targets[target_index]
        current_sentiment = sentiments[target_index]

        if current_sentiment == '-2':
            sent_tag = '-VNEG'
        elif current_sentiment == '-1':
            sent_tag = '-NEG'
        elif current_sentiment == '0':
            sent_tag = '-NEU'
        elif current_sentiment == '1':
            sent_tag = '-POS'
        elif current_sentiment == '2':
            sent_tag = '-VPOS'
        else:
            sent_tag = ''


        # split on space removing whitespaces in order to capture the multi-word targets
        current_target_words = [x.strip() for x in current_targets.split(' ')]
        # print(word_of_text, current_targets)

        if len(current_target_words) == 1:  # if target is constituted by one word only, then it is tagged as 'B'
            if word_of_text == current_target_words[0] \
                    or word_of_text == '@' + current_target_words[0] \
                    or word_of_text == '#' + current_target_words[0]:

                IOB_tag.append((word_of_text, 'B' + sent_tag))
                target_index += 1  # continue with the next target
                idx += 1  # continue with the next word of the text
                continue

            else:
                IOB_tag.append((word_of_text, 'O'))
                idx += 1  # continue with the next word of the text
                continue

        else:  # if we have a multi-word target

            sequence_of_found_target_words = []  # list of the sequence of the words found that are targets
            entire_sequence_of_words_same = True
            text_word_index = idx

            for target_word in current_target_words:
                if target_word == text[text_word_index]:
                    sequence_of_found_target_words.append(text[text_word_index])
                    text_word_index += 1
                else:
                    entire_sequence_of_words_same = False
                    break

            if entire_sequence_of_words_same:

                IOB_tag.append((sequence_of_found_target_words[0], 'B' + sent_tag))  # tag the first word with 'B'

                # each word from the second one to the last one is tagged with 'I'
                for sequence_index in range(1, len(sequence_of_found_target_words)):
                    IOB_tag.append((sequence_of_found_target_words[sequence_index], 'I' + sent_tag))

                # continue with the word that follows the last word that was detected as target word
                idx += len(sequence_of_found_target_words)
                target_index += 1  # continue with the next target

            else:  # if the target word is not the current word of the text, continue with the next word of the text
                IOB_tag.append((word_of_text, 'O'))
                idx += 1

    # print(IOB_tag)
    return IOB_tag


def get_IOB2_with_3_classes_sent(text, targets, sentiments):
    """
    This function extracts the IOB2 tags combined with sentiment tags
    with 3 classes ({NEG, NEU, POS})

    :param text: The given text
    :param targets: The targets that should be detected in the text
    :return: IOB tags with 3 class sentiment tags
    """
    # print('\n')
    # print(targets, sentiments)

    IOB_tag = []
    target_index = 0
    idx = 0

    while idx < len(text):  # while loop that iterates through the words of the text

        word_of_text = text[idx]
        # print(target_index, len(targets))

        # if there are no other targets, then tag the rest of the words as 'O'
        # without checking anything in the rest function
        if target_index >= len(targets):
            IOB_tag.append((word_of_text, 'O'))
            idx += 1
            continue

        # get the targets of the current iteration
        current_targets = targets[target_index]
        current_sentiment = sentiments[target_index]

        if current_sentiment == '-2' or current_sentiment == '-1':
            sent_tag = '-NEG'
        elif current_sentiment == '0':
            sent_tag = '-NEU'
        elif current_sentiment == '1' or current_sentiment == '2':
            sent_tag = '-POS'
        else:
            sent_tag = ''


        # split on space removing whitespaces in order to capture the multi-word targets
        current_target_words = [x.strip() for x in current_targets.split(' ')]
        # print(word_of_text, current_targets)

        if len(current_target_words) == 1:  # if target is constituted by one word only, then it is tagged as 'B'
            if word_of_text == current_target_words[0] \
                    or word_of_text == '@' + current_target_words[0] \
                    or word_of_text == '#' + current_target_words[0]:

                IOB_tag.append((word_of_text, 'B' + sent_tag))
                target_index += 1  # continue with the next target
                idx += 1  # continue with the next word of the text
                continue

            else:
                IOB_tag.append((word_of_text, 'O'))
                idx += 1  # continue with the next word of the text
                continue

        else:  # if we have a multi-word target

            sequence_of_found_target_words = []  # list of the sequence of the words found that are targets
            entire_sequence_of_words_same = True
            text_word_index = idx

            for target_word in current_target_words:
                if target_word == text[text_word_index]:
                    sequence_of_found_target_words.append(text[text_word_index])
                    text_word_index += 1
                else:
                    entire_sequence_of_words_same = False
                    break

            if entire_sequence_of_words_same:

                IOB_tag.append((sequence_of_found_target_words[0], 'B' + sent_tag))  # tag the first word with 'B'

                # each word from the second one to the last one is tagged with 'I'
                for sequence_index in range(1, len(sequence_of_found_target_words)):
                    IOB_tag.append((sequence_of_found_target_words[sequence_index], 'I' + sent_tag))

                # continue with the word that follows the last word that was detected as target word
                idx += len(sequence_of_found_target_words)
                target_index += 1  # continue with the next target

            else:  # if the target word is not the current word of the text, continue with the next word of the text
                IOB_tag.append((word_of_text, 'O'))
                idx += 1

    # print(IOB_tag)
    return IOB_tag


def get_BIOES_with_3_classes_sent(text, targets, sentiments):
    """
    This function extracts the BIOES tags combined with sentiment tags
    with 3 classes ({NEG, NEU, POS})

    :param text: The given text
    :param targets: The targets that should be detected in the text
    :return: IOB tags with 3 class sentiment tags
    """

    BIOES_tag = []
    target_index = 0
    idx = 0

    while idx < len(text):  # while loop that iterates through the words of the text

        word_of_text = text[idx]

        # if there are no other targets,
        # then tag the rest of the words as 'O' without checking anything in the rest function
        if target_index >= len(targets):
            BIOES_tag.append((word_of_text, 'O'))
            idx += 1
            continue


        current_targets = targets[target_index]  # get the targets of the current iteration
        current_sentiment = sentiments[target_index]
        if current_sentiment == '-2' or current_sentiment == '-1':
            sent_tag = '-NEG'
        elif current_sentiment == '0':
            sent_tag = '-NEU'
        elif current_sentiment == '1' or current_sentiment == '2':
            sent_tag = '-POS'
        else:
            sent_tag = ''


        # split on space removing whitespaces in order to capture the multi-word targets
        current_target_words = [x.strip() for x in current_targets.split(' ')]
        # print(word_of_text, current_targets)

        if len(current_target_words) == 1:  # if target is constituted by one word only, then it is tagged as 'B'
            if word_of_text == current_target_words[0]\
                    or word_of_text == '@' + current_target_words[0]\
                    or word_of_text == '#' + current_target_words[0]:

                BIOES_tag.append((word_of_text, 'S' + sent_tag))
                target_index += 1  # continue with the next target
                idx += 1  # continue with the next word of the text
                continue

            else:
                BIOES_tag.append((word_of_text, 'O'))
                idx += 1  # continue with the next word of the text
                continue

        else:  # if we have a multi-word target

            sequence_of_found_target_words = []  # list of the sequence of the words found that are targets
            entire_sequence_of_words_same = True
            text_word_index = idx

            for target_word in current_target_words:
                # if the target word is found in the text,
                # append it to the list and continue with the next word of the target
                if target_word == text[text_word_index]:
                    sequence_of_found_target_words.append(text[text_word_index])
                    text_word_index += 1
                else:
                    entire_sequence_of_words_same = False
                    break
            # print(sequence_of_found_targets)

            if entire_sequence_of_words_same:

                BIOES_tag.append((sequence_of_found_target_words[0], 'B' + sent_tag))  # tag the first word with 'B'

                # each word from the second one to the last one is tagged with 'I'
                for sequence_index in range(1, len(sequence_of_found_target_words)-1):
                    BIOES_tag.append((sequence_of_found_target_words[sequence_index], 'I' + sent_tag))

                BIOES_tag.append((sequence_of_found_target_words[len(sequence_of_found_target_words)-1], 'E' + sent_tag))

                # continue with the word that follows the last word that was detected as target word
                idx += len(sequence_of_found_target_words)
                target_index += 1  # continue with the next target

            else:  # if the target word is not the current word of the text, continue with the next word of the text
                BIOES_tag.append((word_of_text, 'O'))
                idx += 1

    # print(IOB_tag)
    return BIOES_tag






