# HMM-POS-tagging

POS tagging alogorihm using Hidden Markov Model.


## Getting Started

### Prerequisites

```
Python 3
```

## Running the tagger

To run the tagger, use the following command:

```
$ python3 tagger.py -d <training files> -t <test file> -o <output file>
```

You can train the tagger on multiple training files:

```
$ python3 tagger.py -d data/training1.txt data/training2.txt -t data/test1.txt -o data/output1.txt
```

This will create an output file (data/output1.txt) that contains the tagged results.

## Running the validation

To validate the accuracy of the tagging results, use the tagger_validate.py script:

```
$ python3 tagger_validate.py
```

This will generate a file called results.txt, which records the test results with incorrectly matched tags and the accuracy. 
You can check the data/results1.txt file for an example.
