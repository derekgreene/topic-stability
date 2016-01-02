import codecs, os, os.path, re
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess( docs, stopwords, min_df = 3, min_term_length = 2, ngram_range = (1,1), apply_tfidf = True, apply_norm = True ):
	"""
	Preprocess a list containing text documents stored as strings.
	"""
	token_pattern=r"\b\w\w+\b"
	token_pattern = re.compile(token_pattern, re.U)

	def custom_tokenizer( s ):
		return [x.lower() for x in token_pattern.findall(s) if (len(x) >= min_term_length and x[0].isalpha() ) ]

	# Build the Vector Space Model, apply TF-IDF and normalize lines to unit length all in one call
	if apply_norm:
		norm_function = "l2"
	else:
		norm_function = None
	tfidf = TfidfVectorizer(stop_words=stopwords, lowercase=True, strip_accents="unicode", tokenizer=custom_tokenizer, use_idf=apply_tfidf, norm=norm_function, min_df = min_df, ngram_range = ngram_range) 
	X = tfidf.fit_transform(docs)
	terms = []
	# store the vocabulary map
	v = tfidf.vocabulary_
	for i in range(len(v)):
		terms.append("")
	for term in v.keys():
		terms[ v[term] ] = term
	return (X,terms)

def load_stopwords( inpath = "text/stopwords.txt"):
	"""
	Load stopwords from a file into a set.
	"""
	stopwords = set()
	with open(inpath) as f:
		lines = f.readlines()
		for l in lines:
			l = l.strip().lower()
			if len(l) > 0:
				stopwords.add(l)
	return stopwords

def save_corpus( out_prefix, X, terms, doc_ids, classes = None ):
	"""
	Save a pre-processed scikit-learn corpus and associated metadata using Joblib.
	"""
	matrix_outpath = "%s.pkl" % out_prefix 
	joblib.dump((X,terms,doc_ids,classes), matrix_outpath ) 

def load_corpus( in_path ):
	"""
	Load a pre-processed scikit-learn corpus and associated metadata using Joblib.
	"""
	(X,terms,doc_ids,classes) = joblib.load( in_path )
	return (X, terms, doc_ids, classes)


