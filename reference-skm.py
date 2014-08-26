#!/usr/bin/env python
import os, os.path, sys, random
import logging as log
from optparse import OptionParser
import numpy as np
from unsupervised.skm import SphericalKMeans, SpectralSphericalKMeans
import text.util, unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] corpus_file")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=1000)
	parser.add_option("--kmin", action="store", type="int", dest="kmin", help="minimum number of topics", default=5)
	parser.add_option("--kmax", action="store", type="int", dest="kmax", help="maximum number of topics", default=5)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to display", default=10)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=100)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one corpus file" )
	log_level = max(50 - (options.debug * 10), 10)
	log.basicConfig(level=log_level, format='%(asctime)-18s %(levelname)-10s %(message)s', datefmt='%d/%m/%Y %H:%M',)
	init_spectral = True

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

	# Implementation of the algorithm
	if init_spectral:
		impl = SpectralSphericalKMeans( max_iters = options.maxiter )
	else:
		impl = SphericalKMeans( max_iters = options.maxiter )

	# Generate reference clusterings for the specified numbers of clusters
	log.info( "Running reference experiments in range k=[%d,%d]" % ( options.kmin, options.kmax ) )
	for k in range(options.kmin, options.kmax+1):
		log.info( "Applying spherical k-means (k=%d, max_iters=%d, init_spectral=%s, seed=%s)" % ( k, options.maxiter, init_spectral, options.seed ) )
		dir_out_k = os.path.join( dir_out_base, "skm_k%02d" % k )
		if not os.path.exists(dir_out_k):
			os.makedirs(dir_out_k)		
		impl.apply( X, k )
		# Get term rankings for each topic
		term_rankings = []
		for topic_index in range(k):		
			ranked_term_indices = impl.rank_terms( topic_index )
			term_ranking = [terms[i] for i in ranked_term_indices]
			term_rankings.append(term_ranking)
		log.info( "Generated %d rankings covering up to %d terms" % ( len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
		# Print out the top terms, if we want verbose output
		if log_level <= 10 and options.top > 0:
			print unsupervised.rankings.format_term_rankings( term_rankings, top = options.top )

		log.info( "Writing results to %s" % ( dir_out_k ) )
		# Write term rankings
		ranks_out_path = os.path.join( dir_out_k, "ranks_reference.pkl" )
		log.debug( "Writing term ranking set to %s" % ranks_out_path )
		unsupervised.util.save_term_rankings( ranks_out_path, term_rankings )
		# Write document partition
		partition = impl.generate_partition()
		partition_out_path = os.path.join( dir_out_k, "partition_reference.pkl" )
		log.debug( "Writing document partition to %s" % partition_out_path )
		unsupervised.util.save_partition( partition_out_path, partition, doc_ids )


# --------------------------------------------------------------

if __name__ == "__main__":
	main()
 
