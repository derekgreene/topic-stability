#!/usr/bin/env python
import os, sys
import logging as log
from optparse import OptionParser
import numpy as np
import unsupervised.util
import unsupervised.rankings

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] reference_rank_file test_rank_file1 test_rank_file2 ...")
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to use", default=20)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)

	(options, args) = parser.parse_args()
	if( len(args) < 2 ):
		parser.error( "Must specify at least two ranking sets" )
	log.basicConfig(level=max(50 - (options.debug * 10), 10), format='%(asctime)-18s %(levelname)-10s %(message)s', datefmt='%d/%m/%Y %H:%M',)

	# Load cached ranking sets
	log.info( "Reading %d term ranking sets (top=%d) ..." % ( len(args), options.top ) )
	all_term_rankings = []
	for rank_path in args:
		# first set is the reference set
		if len(all_term_rankings) == 0:
			log.debug( "Loading reference term ranking set from %s ..." % rank_path )
		else:
			log.debug( "Loading test term ranking set from %s ..." % rank_path )
		(term_rankings,labels) = unsupervised.util.load_term_rankings( rank_path )
		log.debug( "Set has %d rankings covering %d terms" % ( len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
		# do we need to truncate the number of terms in the ranking?
		if options.top > 1:
			term_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, options.top )
			log.debug( "Truncated to %d -> set now has %d rankings covering %d terms" % ( options.top, len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
		all_term_rankings.append( term_rankings )

	# First argument was the reference term ranking
	reference_term_ranking = all_term_rankings[0]
	all_term_rankings = all_term_rankings[1:]
	r = len(all_term_rankings)
	log.info( "Loaded %d non-reference term rankings" % r )

	# Perform the evaluation
	metric = unsupervised.rankings.AverageJaccard()
	matcher = unsupervised.rankings.RankingSetAgreement( metric )	
	log.info( "Performing reference comparisons with %s ..." % str(metric) )
	all_scores = []
	for i in range(r):
		score = matcher.similarity( reference_term_ranking, all_term_rankings[i] )
		all_scores.append( score )
	
	# Get overall score across all candidates
	all_scores = np.array( all_scores )
	log.info( "Stability=%.4f [%.4f,%.4f]" % ( all_scores.mean(), all_scores.min(), all_scores.max() ) )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
