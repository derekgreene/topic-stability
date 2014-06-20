#!/usr/bin/env python
import os, os.path, sys, random
from optparse import OptionParser
import numpy as np
import text.util, unsupervised.lda, unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] corpus_file")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=1000)
	parser.add_option("--kmin", action="store", type="int", dest="kmin", help="minimum number of topics", default=5)
	parser.add_option("--kmax", action="store", type="int", dest="kmax", help="maximum number of topics", default=5)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to display", default=10)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=200)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	parser.add_option("-p", "--path", action="store", type="string", dest="mallet_path", help="path to Mallet 2 binary (required)", default=None)
	parser.add_option("--reweight", action="store_true", dest="reweight_terms", help="reweight terms after apply LDA")
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one corpus file" )
	# Verify that we can find the Mallet binary.
	if options.mallet_path is None:
		parser.error( "Must specify path to Mallet 2 binary file using the option -p <file_path>" )	
	if not os.path.exists( options.mallet_path ):
		parser.error( "Cannot find specified Mallet 2 binary" )

	# Set random state
	np.random.seed( options.seed )
	random.seed( options.seed )	

	if options.dir_out is None:
		dir_out_base = os.getcwd()
	else:
		dir_out_base = options.dir_out	

	# Load the cached corpus
	corpus_path = args[0]
	(X,terms,doc_ids,classes) = text.util.load_corpus( corpus_path )

	# Create implementation
	impl = unsupervised.lda.MalletLDA( options.mallet_path, top = min(200,len(terms)), max_iters = options.maxiter, reweight_terms = options.reweight_terms )

	# Generate reference LDA topic models for the specified numbers of topics
	print "* Running reference experiments in range k=[%d,%d] max_iters=%d" % ( options.kmin, options.kmax, options.maxiter )
	for k in range(options.kmin, options.kmax+1):
		print "* Applying LDA k=%d (%s) using Mallet in %s" % ( k, impl.__class__.__name__, options.mallet_path )
		dir_out_k = os.path.join( dir_out_base, "lda_k%02d" % k )
		if not os.path.exists(dir_out_k):
			os.makedirs(dir_out_k)		
		print "Results will be written to %s" % ( dir_out_k )
		impl.seed = options.seed
		try:
			impl.apply( X, k )
		except Exception, error:
			print str(error)
			print "Skipping LDA for k=%d" % k
			continue
		# Get term rankings for each topic
		term_rankings = []
		for topic_index in range(k):		
			ranked_term_indices = impl.rank_terms( topic_index )
			term_ranking = [terms[i] for i in ranked_term_indices]
			term_rankings.append(term_ranking)
		print "Writing %d rankings covering up to %d terms" % ( len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) 
		# Print out the top terms
		if options.top > 0:
			print unsupervised.rankings.format_term_rankings( term_rankings, top = options.top )
		# Write term rankings
		ranks_out_path = os.path.join( dir_out_k, "ranks_reference.pkl" )
		print "Writing term ranking set to %s" % ranks_out_path
		unsupervised.util.save_term_rankings( ranks_out_path, term_rankings )
		# Write document partition
		partition = impl.generate_partition()
		partition_out_path = os.path.join( dir_out_k, "partition_reference.pkl" )
		print "Writing document partition to %s" % partition_out_path
		unsupervised.util.save_partition( partition_out_path, partition, doc_ids )

	print "* Done"	  

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
 
