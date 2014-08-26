#!/usr/bin/env python
import os, sys, random
import logging as log
from optparse import OptionParser
import numpy as np
import text.util, unsupervised.nmf, unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] corpus_file")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=1000)
	parser.add_option("--kmin", action="store", type="int", dest="kmin", help="minimum number of topics", default=5)
	parser.add_option("--kmax", action="store", type="int", dest="kmax", help="maximum number of topics", default=5)
	parser.add_option("-r","--runs", action="store", type="int", dest="runs", help="number of runs", default=1)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=10)
	parser.add_option("-s", "--sample", action="store", type="float", dest="sample_ratio", help="sampling ratio of documents to include in each run (range is 0 to 1)", default=0.8)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	parser.add_option("-w","--writefactors", action="store_true", dest="write_factors", help="write complete factorization results")
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	(options, args) = parser.parse_args()
	if len(args) < 1:
		parser.error( "Must specify at least one corpus file" )	
	log.basicConfig(level=max(50 - (options.debug * 10), 10), format='%(asctime)-18s %(levelname)-10s %(message)s', datefmt='%d/%m/%Y %H:%M',)
	# use nimfa instead of sklearn?
	use_nimfa = True

	if options.dir_out is None:
		dir_out_base = os.getcwd()
	else:
		dir_out_base = options.dir_out

	# Load the cached corpus
	corpus_path = args[0]
	(X,terms,doc_ids,classes) = text.util.load_corpus( corpus_path )

	# Choose implementation
	if use_nimfa:
		impl = unsupervised.nmf.NimfaNMF( max_iters = options.maxiter, init_strategy = "random", update = "euclidean" )
	else:
		impl = unsupervised.nmf.SklNMF( max_iters = options.maxiter, init_strategy = "random" )

	n_documents = X.shape[0]
	n_sample = int( options.sample_ratio * n_documents )
	indices = np.arange(n_documents)

	# Generate all NMF topic models for the specified numbers of topics
	log.info( "Testing models in range k=[%d,%d]" % ( options.kmin, options.kmax ) )
	log.info( "Sampling ratio = %.2f - %d/%d documents per run" % ( options.sample_ratio, n_sample, n_documents ) )
	for k in range(options.kmin, options.kmax+1):
		# Set random state
		np.random.seed( options.seed )
		random.seed( options.seed )			
		log.info( "Applying NMF (k=%d, runs=%d, seed=%s - %s) ..." % ( k, options.runs, options.seed, impl.__class__.__name__ ) )
		dir_out_k = os.path.join( dir_out_base, "nmf_k%02d" % k )
		if not os.path.exists(dir_out_k):
			os.makedirs(dir_out_k)		
		log.debug( "Results will be written to %s" % dir_out_k )
		# Run NMF
		for r in range(options.runs):
			log.info( "NMF run %d/%d (k=%d, max_iters=%d)" % (r+1, options.runs, k, options.maxiter ) )
			file_suffix = "%s_%03d" % ( options.seed, r+1 )
			# sub-sample data
			np.random.shuffle(indices)
			sample_indices = indices[0:n_sample]
			S = X[sample_indices,:]
			sample_doc_ids = []
			for doc_index in sample_indices:
				sample_doc_ids.append( doc_ids[doc_index] )
			# apply NMF
			impl.apply( S, k )
			# Get term rankings for each topic
			term_rankings = []
			for topic_index in range(k):		
				ranked_term_indices = impl.rank_terms( topic_index )
				term_ranking = [terms[i] for i in ranked_term_indices]
				term_rankings.append(term_ranking)
			log.debug( "Generated ranking set with %d topics covering up to %d terms" % ( len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
			# Write term rankings
			ranks_out_path = os.path.join( dir_out_k, "ranks_%s.pkl" % file_suffix )
			log.debug( "Writing term ranking set to %s" % ranks_out_path )
			unsupervised.util.save_term_rankings( ranks_out_path, term_rankings )
			# Write document partition
			partition = impl.generate_partition()
			partition_out_path = os.path.join( dir_out_k, "partition_%s.pkl" % file_suffix )
			log.debug( "Writing document partition to %s" % partition_out_path )
			unsupervised.util.save_partition( partition_out_path, partition, sample_doc_ids )			
			# Write the complete factorization?
			if options.write_factors:
				factor_out_path = os.path.join( dir_out_k, "factors_%s.pkl" % file_suffix )
				# NB: need to make a copy of the factors
				log.debug( "Writing factorization to %s" % factor_out_path )
				unsupervised.util.save_nmf_factors( factor_out_path, np.array( impl.W ), np.array( impl.H ), sample_doc_ids )
		  

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
