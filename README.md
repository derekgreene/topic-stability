topic-stability
===============

### Summary
Despite the many [topic modeling algorithms](http://en.wikipedia.org/wiki/Topic_model) that have been proposed for text mining, a common challenge is selecting an appropriate number of topics for a particular data set. Choosing too few topics will produce results that are overly broad, while choosing too many will result in the over-clustering of a data into many redundant, highly-similar topics. We have developed a *stability analysis* approach to address this problem, the idea being that a model with an appropriate number of topics will be more robust to perturbations in the data. Details of this approach are described in the paper:

	How Many Topics? Stability Analysis for Topic Models (2014)
	Derek Greene, Derek O'Callaghan, Pádraig Cunningham
	arXiv pre-print 1404.4606
	http://arxiv.org/abs/1404.4606
	
This repository contains a Python reference implementation of the above approach.

### Dependencies
Tested with Python 2.7.x and requiring the following packages, which are available via PIP:
 - Required: numpy >= 1.8.0 (http://www.numpy.org/)
 - Required: scikit-learn >= 0.14 (http://scikit-learn.org/stable/)
 - Optional: nimfa >= 1.0 (http://nimfa.biolab.si/)
 - Optional: prettytable >= 0.7.2 (https://code.google.com/p/prettytable/)

The following dependency is bundled with this project:
- hungarian-algorithm 2013-11-03 (https://github.com/tdedecko/hungarian-algorithm)
 
To run the LDA tools, an installation of Mallet 2.0 is required, which is available [here](http://mallet.cs.umass.edu/).
