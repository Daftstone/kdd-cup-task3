import collections
import math
import numpy as np


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def last_predictions(locale, sess_test_locale, products, test_locale_names, graph=None, graph1=None):
    predictions = []
    for _, row in sess_test_locale.iterrows():
        his_list = [s.strip("'") for s in row['prev_items'][1:-1].split(" ")][::-1]
        flag = False
        for i in range(len(his_list)):
            cur_id = his_list[i]
            try:
                predictions.append(products[locale][cur_id])
                flag = True
                break
            except:
                for local in test_locale_names:
                    try:
                        predictions.append(products[local][cur_id])
                        flag = True
                        break
                    except:
                        pass
                if (flag == True):
                    break
        if (flag == False):
            predictions.append(list(products[locale].values())[0])
        if (isinstance(predictions[-1], str) == False):
            predictions[-1] = list(products[locale].values())[0]
    sess_test_locale['next_item_prediction'] = predictions
    sess_test_locale.drop('prev_items', inplace=True, axis=1)
    return sess_test_locale


def title_predictions(locale, sess_test_locale, products, graph, graph1):
    predictions = []
    candidate_count = 0
    for _, row in sess_test_locale.iterrows():
        his_list = [s.strip("'") for s in row['prev_items'][1:-1].split(" ")][::-1]
        his_titles = []
        his_titles_id = []

        # obtain the latest record of the corresponding language in the history record, and we predict the next title based on this record
        for i in range(len(his_list)):
            try:
                title = products[locale][his_list[i]]
                if (isinstance(title, str)):
                    his_titles.append(title)
                    his_titles_id.append(his_list[i])
            except:
                pass

        # we only use the last interacted item
        his_titles = his_titles[:1]

        # get the last item's co-visiting items
        titles_map = graph[his_titles_id[0]]
        titles_map = sorted(titles_map.items(), key=lambda x: x[1], reverse=True)
        titles_map = {v[0]: v[1] for v in titles_map}
        titles_array = np.array(list(titles_map.values()))
        threshold = np.mean(titles_array) + 6. * np.std(titles_array)

        candidate = []
        for key, value in titles_map.items():
            if (
                    value > threshold):  # if the maximum item's weight in co-visiting items is larger than a certain threshold (here is 6-sigma as shwon in Line 145)
                try:
                    candidate.append(" ".join(products[locale][key].split()).replace("  ", " "))
                    break
                except:
                    pass
            else:
                break

        # if the number of tokens in the title is too small, to avoid the risk of low BLEU due to wrong prediciton, we directly use the title of the last item as the predicted title
        if (len(his_titles[0].split()) <= 2 and locale != 'JP' and len(candidate) > 0):  # if
            predictions.append(candidate[0])
            continue

        # if no co-visiting item satisfies 6-sigma, we use a richer co-visiting graph (add task1 and task 2 test data) to find out whether any item satisfies 7-sigma (Line 166-179)
        if (len(candidate) == 0):
            titles_map = graph1[his_titles_id[0]]
            titles_map = sorted(titles_map.items(), key=lambda x: x[1], reverse=True)
            titles_map = {v[0]: v[1] for v in titles_map}
            titles_array = np.array(list(titles_map.values()))
            threshold = np.mean(titles_array) + 7. * np.std(titles_array)
            for key, value in titles_map.items():
                if (value > threshold):
                    try:
                        candidate.append(" ".join(products[locale][key].split()).replace("  ", " "))
                        break
                    except:
                        pass
                else:
                    break
            # if still no co-visiting item satisfies 7-sigma, we directly use hte last item's title as the predicted title
            if (len(candidate) == 0):
                answer = his_titles[0]
            # else, we take the intersection of the title of the last product and the predicted title. If the number of words in the intersection is greater than a certain threshold,
            # which is 50% here, we choose the intersection as the last predicted title, otherwise, we still choose the last item's title as prediction title
            else:
                list1 = split_notation(his_titles[0].lower().split())
                set1 = set(list1)
                list2 = split_notation(candidate[0].lower().split())
                list22 = split_notation(candidate[0].split())
                set2 = set(list2)
                set3 = set1 & set2
                list3 = []
                list4 = []
                for iii in range(len(list2)):
                    if (list2[iii] in set3):
                        list3.append(list22[iii])
                        if (iii > 0):
                            if ("%s %s" % (list22[iii - 1], list22[iii]) in candidate[0]):
                                if (len(list4) > 0 and list4[-1] != " "):
                                    list4.append(" ")
                                list4.append(list22[iii])
                            elif ("%s%s" % (list22[iii - 1], list22[iii]) in candidate[0]):
                                list4.append(list22[iii])
                        else:
                            list4.append(list22[iii])
                    else:
                        if (iii > 0):
                            if ("%s %s" % (list22[iii - 1], list22[iii]) in candidate[0] and len(list4) > 0 and
                                    list4[
                                        -1] != " "):
                                list4.append(" ")
                if (locale == 'JP' and len("".join(list3)) > len("".join(list2)) * 0.8):
                    list4 = finetune(list4)
                    answer = "".join(list4)
                    candidate_count += 1
                elif (locale != 'JP' and len(list3) > len(list2) * 0.8):
                    list4 = finetune(list4)
                    answer = "".join(list4)
                    candidate_count += 1
                else:
                    answer = his_titles[0]
        # similar to Line 185-226
        else:
            list1 = split_notation(his_titles[0].lower().split())
            set1 = set(list1)
            list2 = split_notation(candidate[0].lower().split())
            list22 = split_notation(candidate[0].split())
            set2 = set(list2)
            set3 = set1 & set2
            list3 = []
            list4 = []
            for iii in range(len(list2)):
                if (list2[iii] in set3):
                    list3.append(list22[iii])
                    if (iii > 0):
                        if ("%s %s" % (list22[iii - 1], list22[iii]) in candidate[0]):
                            if (len(list4) > 0 and list4[-1] != " "):
                                list4.append(" ")
                            list4.append(list22[iii])
                        elif ("%s%s" % (list22[iii - 1], list22[iii]) in candidate[0]):
                            list4.append(list22[iii])
                    else:
                        list4.append(list22[iii])
                else:
                    if (iii > 0):
                        if ("%s %s" % (list22[iii - 1], list22[iii]) in candidate[0] and len(list4) > 0 and list4[
                            -1] != " "):
                            list4.append(" ")
            if (locale == 'JP' and len("".join(list3)) > len("".join(list2)) * 0.5):
                list4 = finetune(list4)
                answer = "".join(list4)
                candidate_count += 1
            elif (locale != 'JP' and len(list3) > len(list2) * 0.5 and len(list3) > 2):
                list4 = finetune(list4)
                answer = "".join(list4)
                candidate_count += 1
            else:
                answer = his_titles[0]
        predictions.append(answer)
    sess_test_locale['next_item_prediction'] = predictions
    sess_test_locale.drop('prev_items', inplace=True, axis=1)
    return sess_test_locale


def split_notation(ll):
    new_ll = []
    for l in ll:
        temp = l.strip(".,!:';\"!@#$%^&*()-+【】")
        index = l.find(temp)
        # tt = list(l[:index]) + [temp] + list(l[index + len(temp):])
        tt = list(l[:index]) + [temp] + list(l[index + len(temp):])
        if (tt[0] == ''):
            tt = tt[1:]
        if (tt[-1] == ''):
            tt = tt[:-1]
        new_ll += tt
    return new_ll


def finetune(ll):
    new_ll = []
    for i in range(len(ll)):
        if (i > 0 and len(ll[i]) == 1 and ll[i] in ",.;:!&" and ll[i] == ll[i - 1]):
            print("test")
            pass
        elif (i > 1 and len(ll[i]) == 1 and ll[i - 1] == " " and ll[i] in ",.;:!&" and ll[i] == ll[i - 2]):
            pass
        else:
            new_ll.append(ll[i])
    return new_ll
