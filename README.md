topic-stability
===============

### Summary
Despite the many [topic modeling algorithms](http://en.wikipedia.org/wiki/Topic_model) that have been proposed for text mining, a common challenge is selecting an appropriate number of topics for a particular data set. Choosing too few topics will produce results that are overly broad, while choosing too many will result in the over-clustering of a data into many redundant, highly-similar topics. We have developed a *stability analysis* approach to address this problem, the idea being that a model with an appropriate number of topics will be more robust to perturbations in the data. Details of this approach are described in the following paper:

	How Many Topics? Stability Analysis for Topic Models (2014)
	Derek Greene, Derek O'Callaghan, Pádraig Cunningham
	Proc. European Conference on Machine Learning (ECML'14)
	http://arxiv.org/abs/1404.4606	
	
This repository contains a Python reference implementation of the above approach.

### Dependencies
Tested with Python 2.7.x and requiring the following packages, which are available via PIP:

* Required: numpy >= 1.8.0 (http://www.numpy.org/)
* Required: scikit-learn >= 0.14 (http://scikit-learn.org/stable/)
* Required for LDA: scipy >= 0.13 (http://www.scipy.org/)
* Required for NMF: nimfa >= 1.0 (http://nimfa.biolab.si/)
* Required for utility tools: prettytable >= 0.7.2 (https://code.google.com/p/prettytable/)

The following dependency is bundled with this project:
- hungarian-algorithm 2013-11-03 (https://github.com/tdedecko/hungarian-algorithm)
 
To run the LDA tools, an installation of Mallet 2.0 is required, which is available [here](http://mallet.cs.umass.edu/).

### Basic Usage
Before apply topic modeling to a corpus, the first step is to pre-process the corpus and store it in a suitable format. The script 'parse-text.py' can be used to parse a directory of plain text documents. Here, we parse all .txt files in the directory or sub-directories of 'sample'. The output files will also have the prefix 'sample'.

	python parse-text.py sample/ -o sample

The output will be a number of binary files, with the main corpus file being named 'sample.pkl'.

If we are interested in applying topic modelling based on Non-negative Matrix Factorization (NMF), we next generate a *reference* set of topics on the pre-processed corpus by using the script 'reference-nmf.py'.  Our initial estimate for a range for the number of topics (*k*) for our corpus is between 2 and 8.

	python reference-nmf.py sample.pkl --kmin 2 --kmax 8 -o reference-nmf/

The output of the process above will be 7 sub-directories of 'reference-nmf', each containing a reference topic model for a different value of *k*.

Next, we generate a set of 50 "standard" topic models for each value of *k* using 'generate-nmf.py'. Again we specify the range of candidates values for *k* to be [2,8].
	
	python generate-nmf.py sample.pkl --kmin 2 --kmax 8 -r 50 -o topic-nmf/
