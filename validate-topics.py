#!/usr/bin/env python
import os, os.path, sys
import logging as log
from optparse import OptionParser
import text.util, unsupervised.util, unsupervised.validation

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] corpus_file input_directory1 input_directory2 ...")
	parser.add_option("-p", "--precision", action="store", type="int", dest="precision", help="precision for results", default=2)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	(options, args) = parser.parse_args()
	if( len(args) < 2 ):
		parser.error( "Must specify at least a corpus and one input direct containing topic modeling results" )	
	log.basicConfig(level=max(50 - (options.debug * 10), 10), format='%(asctime)-18s %(levelname)-10s %(message)s', datefmt='%d/%m/%Y %H:%M',)

	# Read the corpus
	corpus_path = args[0]
	print "* Reading %s ..." % corpus_path
	(X,terms,doc_ids,classes) = text.util.load_corpus( corpus_path )
	if classes is None or len(classes) < 2:
		print "Error: No ground truth class information for this corpus"
		sys.exit(1)
	class_partition = unsupervised.util.clustermap_to_partition( classes, doc_ids )

	partition_validator = unsupervised.validation.PartitionValidator( classes, doc_ids )
	term_validator = unsupervised.validation.TermValidator( X, terms, class_partition )
	term_top_values = [ 10, 20, 50, 100 ]
	
	# Process each directory
	mean_collection = unsupervised.validation.ScoreCollection()
	for result_dir_path in args[1:]:
		result_dir_path = result_dir_path.rstrip(os.sep)
		print "* Processing results in directory", result_dir_path
		fils = os.listdir( result_dir_path )
		path_pairs = []
		for fname in fils:
			if fname.startswith( "ranks_" ) and fname.endswith( ".pkl"):
				rank_file_path = os.path.join( result_dir_path, fname )	
				# do we have a partition file for this rank file?
				partition_file_path = os.path.join( result_dir_path, fname.replace( "ranks_", "partition_" ) )
				if not os.path.exists(partition_file_path):
					partition_file_path = None
				path_pairs.append( (rank_file_path,partition_file_path) )
		if len(path_pairs) == 0:
			print "Warning: No ranking sets found in directory ", result_dir_path
			continue
		print "Validating %d topic ranking sets" % len(path_pairs)
		collection = unsupervised.validation.ScoreCollection()
		for path_pair in path_pairs:
			# evaluate partition
			if path_pair[1] is None:
				print "Warning: no partition available for", path_pair[0]
				partition_results = {}
			else:
				partition, doc_ids = unsupervised.util.load_partition( path_pair[1] )
				partition_results = partition_validator.evaluate( partition, doc_ids )
			# evaluate topic terms
			(term_rankings,labels) = unsupervised.util.load_term_rankings( path_pair[0] )
			term_results = term_validator.evaluate( term_rankings, term_top_values )
			# add all results to the collection
			experiment_key = os.path.splitext( os.path.basename( path_pair[0] ) )[0]
			collection.add( experiment_key, dict( partition_results.items() + term_results.items() )  )
		# finished this directory, so print results for it
		print collection.create_table( precision = options.precision )
		mean_collection.add( os.path.basename(result_dir_path), collection.aggregate_scores()[0] )

	# Display mean scores across all experiments
	print "* Summary - Mean Scores"
	print mean_collection.create_table( precision = options.precision )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
