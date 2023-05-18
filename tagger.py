import os
import sys

# Initial Probability
# {tag: probability to be first word}
initial_prob = {}

# Transition Probability
# {(tag at t - 1, tag at t): probability of transition}
trans_prob = {}

# Emission Probability
# { word: {tag1: prob, tag2: prob}}
emission_prob = {}

all_tags = [
    "AJ0",
    "AJC",
    "AJS",
    "AT0",
    "AV0",
    "AVP",
    "AVQ",
    "CJC",
    "CJS",
    "CJT",
    "CRD",
    "DPS",
    "DT0",
    "DTQ",
    "EX0",
    "ITJ",
    "NN0",
    "NN1",
    "NN2",
    "NP0",
    "ORD",
    "PNI",
    "PNP",
    "PNQ",
    "PNX",
    "POS",
    "PRF",
    "PRP",
    "PUL",
    "PUN",
    "PUQ",
    "PUR",
    "TO0",
    "UNC",
    "VBB",
    "VBD",
    "VBG",
    "VBI",
    "VBN",
    "VBZ",
    "VDB",
    "VDD",
    "VDG",
    "VDI",
    "VDN",
    "VDZ",
    "VHB",
    "VHD",
    "VHG",
    "VHI",
    "VHN",
    "VHZ",
    "VM0",
    "VVB",
    "VVD",
    "VVG",
    "VVI",
    "VVN",
    "VVZ",
    "XX0",
    "ZZ0",
]

ambiguity_tags = [
    "AJ0-AV0",
    "AJ0-VVN",
    "AJ0-VVD",
    "AJ0-NN1",
    "AJ0-VVG",
    "AVP-PRP",
    "AVQ-CJS",
    "CJS-PRP",
    "CJT-DT0",
    "CRD-PNI",
    "NN1-NP0",
    "NN1-VVB",
    "NN1-VVG",
    "NN2-VVZ",
    "VVD-VVN",
]


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.

    training(training_list)

    sentences = read_test_file(test_file)

    print("Tagging the file.")

    tagging(sentences, output_file)


def tagging(sentences, output_file):
    """Tag each sentence and write to the output file."""
    result = []
    for sentence in sentences:
        tags = viterbi(sentence)
        for i in range(len(sentence)):
            result.append((sentence[i], tags[i]))

    # Write to output file
    f = open(output_file, "w")
    for word_tag in result:
        if type(word_tag[1]) == frozenset:
            word = word_tag[0]
            two_tag = list(word_tag[1])
            if "-".join([two_tag[0], two_tag[1]]) in ambiguity_tags:
                tag = "-".join([two_tag[0], two_tag[1]])
            else:
                tag = "-".join([two_tag[1], two_tag[0]])

        else:
            word, tag = word_tag
        f.write(word + " : " + tag + "\n")


def fill_untrained_word(word):
    """
    Fill probability table of an untrained word with the most common tag
    for the word format.
    Else fill it with all tags
    """
    emission_prob[word] = {}
    possible_tags = []

    # Get possible tags based on the word format
    if word[0].isupper():
        possible_tags.append("NP0")
        possible_tags.append(frozenset(["NP0", "NN1"]))
    if "-" in word:
        possible_tags.append("AJ0")
    if word[-1] == "s" or (len(word) > 2 and word[-2] == "es"):
        possible_tags.append("NN2")
    else:
        possible_tags.append("NN1")
        possible_tags.append("NN0")
    if len(word) > 3 and word[-3:] == "ing":
        possible_tags.append("VVG")
        possible_tags.append("NN1")
        possible_tags.append("AJ0")
        possible_tags.append(frozenset(["AJ0", "NN1"]))
    if len(word) > 2 and word[-2:] == "ly":
        possible_tags.append("AV0")
    if len(word) > 2 and word[-2:] == "ed":
        possible_tags.append("AJ0")
        possible_tags.append("VVD")
        possible_tags.append("VVN")
        possible_tags.append(frozenset(["VVN", "VVD"]))

    if possible_tags == []:
        for tag in all_tags:
            emission_prob[word][tag] = sys.float_info.epsilon

        for tag in ambiguity_tags:
            emission_prob[word][frozenset(tag.split("-"))] = sys.float_info.epsilon
    else:
        for tag in possible_tags:
            emission_prob[word][tag] = sys.float_info.epsilon


def viterbi(sentence):
    """Viterbi algorithm to find the most likely tag sequence for a sentence."""
    prob = {}
    prev = {}

    # Base case (first word)
    first_word = sentence[0]

    # If the first word is never seen in training file
    if first_word not in emission_prob:
        fill_untrained_word(first_word)
    all_possible_tag = emission_prob[first_word]

    prob[0] = {}
    prev[0] = {}
    for tag in all_possible_tag:
        if tag == "__total":
            continue
        inital = initial_prob[tag] if tag in initial_prob else sys.float_info.epsilon
        emission = emission_prob[first_word][tag]
        prob[0][tag] = inital * emission
        prev[0][tag] = None

    # Recursive case
    for i in range(1, len(sentence)):
        word = sentence[i]
        prob[i] = {}
        prev[i] = {}
        # If the word is never seen in training file
        if word not in emission_prob:
            fill_untrained_word(word)
        all_possible_tag = emission_prob[word]
        total_prob = 0
        for tag in all_possible_tag:
            max_prob = -1
            max_prev_tag = None
            for prev_tag in prob[i - 1]:
                trans = (
                    trans_prob[(prev_tag, tag)]
                    if (prev_tag, tag) in trans_prob
                    else sys.float_info.epsilon
                )

                curr_prob = prob[i - 1][prev_tag] * trans * emission_prob[word][tag]
                if curr_prob > max_prob:
                    max_prob = curr_prob
                    max_prev_tag = prev_tag

            # Record maximum probability for each possible tag
            prob[i][tag] = max_prob
            total_prob += max_prob
            # Record previus tag to generate maximum probability
            prev[i][tag] = max_prev_tag

        # Normalization
        for tag in prob[i]:
            prob[i][tag] = prob[i][tag] / total_prob

    max_prob_tag = prob[len(sentence) - 1]
    target_tag = max(max_prob_tag, key=max_prob_tag.get)

    # Trace from the end to get the sequence of tags with max probability
    tag_seq = [target_tag]
    for i in range(len(sentence) - 1, 0, -1):
        tag_seq.append(prev[i][target_tag])
        target_tag = prev[i][target_tag]

    tag_seq.reverse()
    return tag_seq


def read_test_file(test_file):
    """Read each sentence from the test file"""
    sentences = []
    with open(test_file) as f:
        new_sentence = []
        open_tag = False
        for line in f:
            word = line.strip()
            new_sentence.append(word)

            if open_tag:
                if word == '"':
                    open_tag = False
                    sentences.append(new_sentence)
                    new_sentence = []
            else:
                if word == '"':
                    open_tag = True
                elif word == "." or word == "?" or word == "!":
                    sentences.append(new_sentence)
                    new_sentence = []

        if len(new_sentence) != 0:
            sentences.append(new_sentence)

    return sentences


def training(training_list):
    """Reads the traning file and genreate probabilities."""
    sentence_num = 0
    trans_count = {}
    emission_count = {}

    for training in training_list:
        with open(training) as f:
            first_word = True
            prev_tag = None
            open_tag = False

            for line in f:
                word, _, tag = [x.strip() for x in line.split()]

                # Treats different order ambiguity tag the same
                if "-" in tag:
                    tag = frozenset(tag.split("-"))

                if tag not in trans_count:
                    trans_count[tag] = {"__total": 0}

                if tag not in emission_count:
                    emission_count[tag] = {"__total": 0}

                if word not in emission_count[tag]:
                    emission_count[tag][word] = 0
                emission_count[tag][word] += 1
                emission_count[tag]["__total"] += 1

                # Get number of times tag appears at the start of a sentence
                if first_word == True:
                    if tag not in initial_prob:
                        initial_prob[tag] = 0
                    initial_prob[tag] += 1
                    first_word = False
                    if word == '"':
                        open_tag = True
                    prev_tag = tag

                else:

                    if tag not in trans_count[prev_tag]:
                        trans_count[prev_tag][tag] = 0
                    trans_count[prev_tag][tag] += 1
                    trans_count[prev_tag]["__total"] += 1
                    prev_tag = tag

                    # If there was a quote open before
                    if open_tag == True:
                        # The sentence ends when the quote is closed
                        # Otherwise, the sentence continues
                        if word == '"':
                            sentence_num += 1
                            open_tag = False
                            first_word = True
                    else:
                        if word == '"':
                            open_tag = True
                        elif word == "." or word == "?" or word == "!":
                            sentence_num += 1
                            first_word = True

    # Get initial probability for tag
    for tag in initial_prob:
        initial_prob[tag] = initial_prob[tag] / sentence_num

    # Get transition probability which how likely one tag follow by another tag
    for prev_tag in trans_count:
        if trans_count[prev_tag]["__total"] == 0:
            continue
        for next_tag in trans_count[prev_tag]:
            if next_tag != "__total":
                trans_prob[(prev_tag, next_tag)] = (
                    trans_count[prev_tag][next_tag] / trans_count[prev_tag]["__total"]
                )

    # Get emission probability which how likely a word is tagged with a tag
    for tag in emission_count:
        for word in emission_count[tag]:
            if word not in emission_prob:
                emission_prob[word] = {}
            if word != "__total":
                emission_prob[word][tag] = (
                    emission_count[tag][word] / emission_count[tag]["__total"]
                )

    return None


if __name__ == "__main__":
    # Run the tagger function.
    print("Starting the tagging process.")

    print("length of all tags:", len(all_tags) + len(ambiguity_tags))

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d") + 1 : parameters.index("-t")]
    test_file = parameters[parameters.index("-t") + 1]
    output_file = parameters[parameters.index("-o") + 1]
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
