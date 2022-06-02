
def targets_are_equivalent(target1, target2, text):
    """
    Checks whether two targets are equivalent.
    Two targets are equivalent if they are exactly the same, or if they are different
    only on starting from # or @ AND they appear only once in the text.
    We set the extra restriction of appearing only once in the text in the form
    @word or #word, because the same aspect may appear more than once in a text,
    expressing different opinion about it.

    :param target1:
    :param target2:
    :param text:
    :return:
    """

    if target1 == target2:
        return True
    elif ('#' + target1) == target2 and text.count(target2):
        # print('\n', target1)
        # print(target2)
        # print(text)
        return True
    elif ('@' + target1) == target2 and text.count(target2):
        # print('\n',target1)
        # print(target2)
        # print(text)
        return True
    elif target1 == ('#' + target2) and text.count(target1):
        # print('\n',target1)
        # print(target2)
        # print(text)
        return True
    elif target1 == ('@' + target2) and text.count(target1):
        # print('\n',target1)
        # print(target2)
        # print(text)
        return True
    else:
        return False


def find_equivalent_targets(targets_annotator1, targets_annotator2, text):
    """
    If one of the annotators has included "@" and "#" in the target, while
    the other one hasn't, then the targets are regarded as equal and "@" and "#"
    are included in the final target column. ! They are regarded as equal only if
    the exact target word does not have any other occurence in the text !

    This is important because in a great number of rows the one annotator includes
    those marks while the other doesn't
    e.g. ann1: @COSMOTE, ann2: COSMOTE
         ann1: FlocafeEspressoRoom, ann2: #FlocafeEspressoRoom
         ann1: chrisochoidis, ann2: @chrisochoidis
         ann1: survivorGR, ann2: #survivorGR
         ann1: netflix, ann2: @netflix
    :param text: the text in which the targets appear in
    :param targets_annotator1: targets from annotator 1
    :param targets_annotator2: targets from annotator 2
    :return: the targets that are equivalent among the two annotators
    """

    # print('\n', targets_annotator1, ' -------- ', targets_annotator2)
    targets_annotator1 = [x.strip() for x in targets_annotator1.split(',')]
    targets_annotator2 = [x.strip() for x in targets_annotator2.split(',')]

    # find the target set with the maximum number of targets
    if len(targets_annotator2) > len(targets_annotator1):
        temp_targets = targets_annotator1
        targets_annotator1 = targets_annotator2
        targets_annotator2 = temp_targets

    equal_targets = []
    target1_index = 0
    target2_index = 0
    previous_same_aspect2_position = 0

    while target1_index < len(targets_annotator1):
        same_aspect_found = False

        while target2_index < len(targets_annotator2) and target1_index < len(targets_annotator1):

            target1 = targets_annotator1[target1_index]
            target2 = targets_annotator2[target2_index]

            # if the targets are same or if they are different because the one has @ or #, while the other hasn't
            if targets_are_equivalent(target1, target2, text):

                if target1 == target2:
                    equal_targets.append(target1)
                elif ('#' + target1) == target2 or ('@' + target1) == target2:
                    equal_targets.append(target2)
                elif target1 == ('#' + target2) or target1 == ('@' + target2):
                    equal_targets.append(target1)

                target1_index += 1
                target2_index += 1
                previous_same_aspect2_position = target2_index
                same_aspect_found = True
                # print(equal_targets)
                continue
            else:
                target2_index += 1
                same_aspect_found = False

        if not same_aspect_found:
            target2_index = previous_same_aspect2_position
            target1_index += 1

    # print(equal_targets)
    return equal_targets


def find_exactly_same_targets(targets_annotator1, targets_annotator2):
    """
    Returns the targets that are appear exactly same in the text

    :param targets_annotator1: targets from annotator 1
    :param targets_annotator2: targets from annotator 2
    :return: the targets that are exactly the same between the two annotators
    """

    # print('\n', targets_annotator1, ' -------- ', targets_annotator2)
    targets_annotator1 = [x.strip() for x in targets_annotator1.split(',')]
    targets_annotator2 = [x.strip() for x in targets_annotator2.split(',')]

    # find the target set with the maximum number of targets
    if len(targets_annotator2) > len(targets_annotator1):
        temp_targets = targets_annotator1
        targets_annotator1 = targets_annotator2
        targets_annotator2 = temp_targets

    equal_targets = []
    target1_index = 0
    target2_index = 0
    previous_same_aspect2_position = 0

    while target1_index < len(targets_annotator1):
        same_aspect_found = False

        while target2_index < len(targets_annotator2) and target1_index < len(targets_annotator1):
            # print(target1_index, " ", target2_index, " ")
            # print(targets_annotator1[target1_index], "  ", targets_annotator2[target2_index])

            if targets_annotator1[target1_index] == targets_annotator2[target2_index]:
                equal_targets.append(targets_annotator1[target1_index])
                target1_index += 1
                target2_index += 1
                previous_same_aspect2_position = target2_index
                same_aspect_found = True
                continue
            else:
                target2_index += 1
                same_aspect_found = False

        if not same_aspect_found:
            target2_index = previous_same_aspect2_position
            target1_index += 1

    # print(equal_targets)
    return equal_targets


def get_targets_sentiments(row):
    """
    Finds the targets that appear exactly the same in the text or that differ
    by a @ or # and their corresponding sentiment

    :param row: Row of the dataframe
    :return: Row of the dataframe with the detected targets and sentiments
    """

    equal_targets = []
    equal_sents = []

    # print('\n', targets_annotator1, ' -------- ', targets_annotator2)
    targets_annotator1 = [x.strip() for x in row['targets1'].split(',')]
    targets_annotator2 = [x.strip() for x in row['targets2'].split(',')]

    sents_annotator1 = [x.strip() for x in row['sentiments1'].split(',')]
    sents_annotator2 = [x.strip() for x in row['sentiments2'].split(',')]

    # if the number of targets and the number of sentiments is not equal
    # then we cant be sure which sentiment value is missing, thus we cant
    # assign sentiment values to targets
    if len(targets_annotator1) != len(sents_annotator1):
        row['targets'] = []
        row['sentiment'] = []
        return row

    if len(targets_annotator2) != len(sents_annotator2):
        row['targets'] = []
        row['sentiment'] = []
        return row

    # find the target set with the maximum number of targets (I cant remember why.. xmm)
    if len(targets_annotator2) > len(targets_annotator1):
        temp_targets = targets_annotator1
        targets_annotator1 = targets_annotator2
        targets_annotator2 = temp_targets

        temp_sents = sents_annotator1
        sents_annotator1 = sents_annotator2
        sents_annotator2 = temp_sents

    target1_index = 0
    target2_index = 0
    previous_same_aspect2_position = 0

    while target1_index < len(targets_annotator1):
        same_aspect_found = False

        while target2_index < len(targets_annotator2) and target1_index < len(targets_annotator1):

            target1 = targets_annotator1[target1_index]
            target2 = targets_annotator2[target2_index]

            sent1 = sents_annotator1[target1_index]
            sent2 = sents_annotator2[target2_index]

            # if the targets are same or if they are different because the one has @ or #, while the other hasn't
            if targets_are_equivalent(target1, target2, row['text']):

                if target1 == target2 and sent1 == sent2:
                    equal_targets.append(target1)
                    equal_sents.append(sent1)
                elif (('#' + target1) == target2 or ('@' + target1) == target2) and sent1 == sent2:
                    equal_targets.append(target2)
                    equal_sents.append(sent1)
                elif (target1 == ('#' + target2) or target1 == ('@' + target2)) and sent1 == sent2:
                    equal_targets.append(target1)
                    equal_sents.append(sent1)

                target1_index += 1
                target2_index += 1
                previous_same_aspect2_position = target2_index
                same_aspect_found = True
                # print(equal_targets)
                continue
            else:
                target2_index += 1
                same_aspect_found = False

        if not same_aspect_found:
            target2_index = previous_same_aspect2_position
            target1_index += 1

    row['targets'] = equal_targets
    row['sentiment'] = equal_sents
    # print(equal_targets, equal_sents)
    return row


def get_equal_sent_targets_for_testing(targets1, targets2, sentiments1, sentiments2, text):
    """
    Finds the targets that appear exactly the same in the text or that differ
    by a @ or # and their corresponding sentiment.
    ! Used for testing in this py file!

    :param targets1: Targets from annotator 1
    :param targets2: Targets from annotator 2
    :param sentiments1: Sentiments from annotator 1
    :param sentiments2: Sentiments from annotator 2
    :param text: The text
    :return: The detected targets and sentiments
    """


    """
    Finds the targets that appear exactly the same in the text or that differ
    by a @ or # and their corresponding sentiment.
    ! Used for testing in this py file!

    :param row: Row of the dataframe
    :return: Row of the dataframe with the detected targets and sentiments
    """

    equal_targets = []
    equal_sents = []

    # print('\n', targets_annotator1, ' -------- ', targets_annotator2)
    targets_annotator1 = [x.strip() for x in targets1.split(',')]
    targets_annotator2 = [x.strip() for x in targets2.split(',')]

    sents_annotator1 = [x.strip() for x in sentiments1.split(',')]
    sents_annotator2 = [x.strip() for x in sentiments2.split(',')]

    # print("\n")
    # print(targets_annotator1, sents_annotator1)
    # print(targets_annotator2, sents_annotator2)

    # find the target set with the maximum number of targets (I cant remember why.. xmm)
    if len(targets_annotator2) > len(targets_annotator1):
        temp_targets = targets_annotator1
        targets_annotator1 = targets_annotator2
        targets_annotator2 = temp_targets

        temp_sents = sents_annotator1
        sents_annotator1 = sents_annotator2
        sents_annotator2 = temp_sents

    target1_index = 0
    target2_index = 0
    previous_same_aspect2_position = 0

    while target1_index < len(targets_annotator1):
        same_aspect_found = False

        while target2_index < len(targets_annotator2) and target1_index < len(targets_annotator1):

            target1 = targets_annotator1[target1_index]
            target2 = targets_annotator2[target2_index]

            sent1 = sents_annotator1[target1_index]
            sent2 = sents_annotator2[target2_index]

            # if the targets are same or if they are different because the one has @ or #, while the other hasn't
            if targets_are_equivalent(target1, target2, text):

                if target1 == target2 and sent1 == sent2:
                    equal_targets.append(target1)
                    equal_sents.append(sent1)
                elif (('#' + target1) == target2 or ('@' + target1) == target2) and sent1 == sent2:
                    equal_targets.append(target2)
                    equal_sents.append(sent1)
                elif (target1 == ('#' + target2) or target1 == ('@' + target2)) and sent1 == sent2:
                    equal_targets.append(target1)
                    equal_sents.append(sent1)

                target1_index += 1
                target2_index += 1
                previous_same_aspect2_position = target2_index
                same_aspect_found = True
                # print(equal_targets)
                continue
            else:
                target2_index += 1
                same_aspect_found = False

        if not same_aspect_found:
            target2_index = previous_same_aspect2_position
            target1_index += 1

    # print(equal_targets, equal_sents)
    return equal_targets, equal_sents


if __name__ == '__main__':
    # find_equivalent_targets('Κυριακο Μητσοτακη,κυβερνηση, OTE-Cosmote, Wind, Vodafone ', 'Κυριακο Μητσοτακη, OTE-Cosmote, Wind, Vodafone', 'Κυριάκο Μητσοτάκη και όλη την κυβέρνηση έχουν υποκλαπεί ανεξάρτητα του δικτύου που χρησιμοποιούσαν με δεδομένο ότι μέσω του OTE-Cosmote «βγαίνουν» όλες οι εταιρείες τηλεφωνικής εταιρείες (δηλαδή και Wind και Vodafone) δηλαδή υποκλάπησαν και τα στοιχεία τηλεφωνίας')
    # find_equivalent_targets('Vodafone_GR', '@Vodafone_GR', '@Zed_Ryder @Vodafone_GR Τι να πω...  βλεπω πως είμαστε πολλοί που την @Vodafone_GR έχουμε πατήσει')

    targets1 = 'Νικος Λεπενιωτης, Βασιλη Σκουντη, Cosmote TV, Ολυμπιακου.'
    tragets2 = 'ΚΑΕ Ολυμπιακος, Νικος Λεπενιωτης, Βασιλη Σκουντη, Cosmote TV, ΟΑΚΑ'
    sents1 = '0, 0, 0, 0'
    sents2 = '0, 2, 0, 0, 0'
    text = 'Ο Γενικός Διευθυντής της ΚΑΕ Ολυμπιακός, Νίκος Λεπενιώτης, μίλησε στην εκπομπή του Βασίλη Σκουντή στην Cosmote TV για τις αποφάσεις των ερυθρολεύκων σε διοικητικό επίπεδο, μίλησε για δικαίωση περί της αποχώρησης από το ΟΑΚΑ, ενώ απάντησε στο θέμα της ανέγερσης του νέου γηπέδου του Ολυμπιακού'
    get_equal_sent_targets_for_testing(targets1, tragets2, sents1, sents2, text)

    '''
    find_equivalent_targets('Τραμπ, Μερκελ, cosmote, Πακης', 'Τραμπ, Μερκελ, cosmote, Πακης')
    find_equivalent_targets('FlocafeEspressoRoom', '#FlocafeEspressoRoom')
    find_equivalent_targets('netflix, BigBrotherGR', 'netflix, #bbxeftiles')
    '''

    '''
    get_exactly_same_targets('Κυριακο Μητσοτακη,κυβερνηση, OTE-Cosmote, Wind, Vodafone ', 'Κυριακο Μητσοτακη, OTE-Cosmote, Wind, Vodafone')
    get_exactly_same_targets('Τραμπ, Μερκελ, cosmote, Πακης', 'Τραμπ, Μερκελ, cosmote, Πακης')
    get_exactly_same_targets('Vodafone_GR', '@Vodafone_GR')
    get_exactly_same_targets('FlocafeEspressoRoom', '#FlocafeEspressoRoom')
    get_exactly_same_targets('BigBrotherGR, netflix', 'netflix, #bbxeftiles')
    '''



