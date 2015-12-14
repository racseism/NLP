import logging
import math
import os.path
import time

"""
This class generate words from given corpus file or given sentence. This function is used while 
generating unigram, bigram, trigram and vocabulary generation
"""
class TokenGenerator:
	corporafilename = ""
	tokenDelimeter = ' '
	sentences = []
	corpusSize = 0

	def __init__(self, corporafilename, sentences = []):
		self.corpusSize = 0
		if (len(sentences) == 0):
			fileObject = open(corporafilename)
			self.sentences = fileObject.readlines()
		else:
			self.sentences = sentences

	def GetNextToken(self):
		# Depending on default delimiter of python.
		for sentence in self.GetNextSentence():
			tokens = sentence.replace('\n', '').split()
			for token in tokens:
				yield token

	def SplitSentence(self, sentence):
		tokens = sentence.split()
		for token in tokens:
			yield token


	def GetNextSentence(self):
		for sentence in self.sentences:
			newSentence = sentence.replace('\n', '')
			if sentence.find('STOP') == -1:
				newSentence = newSentence + ' STOP'
			yield newSentence

	def GetTotalSentenceCount(self):
		return len(self.sentences)

	def GetCorpusSize(self):
		if self.corpusSize != 0:
			return self.corpusSize
		size = 0;
		for sentence in self.sentences:
			size += len(sentence.split())

		self.corpusSize = size
		return size;

"""
This class generates vocabulary/unique words from any given corpora file.
We use to generate this to generate vocabulary on training set.
It also generates, How much percentage of UNK words are present in validation set corrsponding 
to training set. It then generates the map of word which is map to UNK in training set.
"""
class VocabularySets:
	#Generates vocabulary sets from validation set.
	def __init__(self):
		self.vocabularySets = []
		self.UNKWords = []
		self.fractionOfUNKWord = 0.0
	"""
	This generates UNK word map.
	1. Calculate the percentage of UNK word in validation set, let x%.
	2. Count the number of words with count=1 in training set.
	3. Map x% of count=1 word of training set to UNK
	"""
	def GenerateUNKWord(self, tokenGenerator):
		startTime = time.clock()
		corpusSize = tokenGenerator.GetCorpusSize();

		#Consider 20% of validation set for finding the UNK word. The word whose count will be 1 or less will be givne UNK world.
		#Use training corpus and validation corpus to find the vocabulary size
		tokenTraversedCount = 0
		vocabularyForUNK = {}
		for token in tokenGenerator.GetNextToken():
			if vocabularyForUNK.has_key(token):
				vocabularyForUNK[token] += 1
			else:
				vocabularyForUNK[token] = 1

		#Extract the word whose count would be 1. This Extracted would be MAP to UNK word in training corpus.
		wordsWithCount1 = 0
		singleWordCount = []
		for key, value in vocabularyForUNK.iteritems():
			if value == 1:
				wordsWithCount1 += 1
				singleWordCount.append(key)
				self.UNKWords.append(key)
		print("Single word count = %d" %wordsWithCount1)
		unknownWordCorpus = int(wordsWithCount1 * self.fractionOfUNKWord)
		print("Unkow word corpus count = %d" %unknownWordCorpus)
		self.UNKWords = singleWordCount[0:unknownWordCorpus]
		print("Length of UNK words count = %d" %len(self.UNKWords))
		#Logging Words map to UNK
		logging.debug("Words map to UNK")
		logging.debug("UNKword Count = %d", len(self.UNKWords))
		endTime = time.clock();
		logging.debug("UNK map generation = %f ", (endTime - startTime))
	"""
	This function generats the percentage of UNK word in validatoin set correponding training set.
	"""
	def GeneratePercentageOfUnkWordInValidationSet(self, validationSetTokenGenerator):
		wordCountMissingInTrainingSet = 0
		for token in validationSetTokenGenerator.GetNextToken():
			if not (self.IsWordInTrainingSetVocabulary(token)):
				wordCountMissingInTrainingSet += 1

		self.fractionOfUNKWord = float(wordCountMissingInTrainingSet)/validationSetTokenGenerator.GetCorpusSize()
		print("Fracton of UNK word %f" %self.fractionOfUNKWord)

	def IsWordMapToUNK(self, token):
		if token in self.UNKWords:
			return True
		return False

	def IsWordInTrainingSetVocabulary(self, token):
		if token in self.vocabularySets:
			return True
		return False

	def GetMappedWordIfSatisfyUNK(self, token):
		if self.IsWordMapToUNK(token):
			return 'UNK'
		return token

	def GetMappedWordIfSatisfyUNKForMLE(self, token):
		if self.IsWordMapToUNK(token) or (not self.IsWordInTrainingSetVocabulary(token)):
			return 'UNK'
		return token

	"""
	This functi generates the dictionary from training set file.
	"""
	def GenerateVocubularyFromTrainingSet(self, trainingSetTokenGenerator, regenerate = False):
		startTime = time.clock()
		if (len(self.vocabularySets) > 0 and (not regenerate)):
			#vocabulary set is already generated
			return
		for token in trainingSetTokenGenerator.GetNextToken():
			if token not in self.vocabularySets:
				if not self.IsWordMapToUNK(token):
					self.vocabularySets.append(token)
		self.vocabularySets.append('UNK')
		endTime = time.clock()
		logging.debug("VocabularySets count = %d", len(self.vocabularySets))
		logging.debug("Vocabulary set generation Time  = %f", (endTime - startTime))

"""
This class generates and store bigram of given corpus.
"""
class Bigram :
	bigrams = {}
	totalCount = 0
	# This will have sum of count(v, w), where w is elment of vocabulary sets.
	# This is used in Katz back-off model
	bigramCountForGivenV = []

	def __init__(self, vocabularySets, backOffModel = False):
		self.bigrams = {}
		self.totalCount = 0
		self.vocabularySets = vocabularySets
		self.katzBackOff = {}

	def GenerateBigram(self, tokenGenerator, enableKatzBackOff = False):
		self.bigrams[('*','*')] = 0
		self.bigrams[('UNK', 'UNK')] = 1
		startTime = time.clock()
		for sentence in tokenGenerator.GetNextSentence():
			xi_1 = '*'
			xi = '*'
			#To handle the start symbol - describe in details
			#Apply UNK word mapping only with few sets of training set. For time being applying on complete sets.
			self.bigrams[('*', '*')] += 1
			for token in tokenGenerator.SplitSentence(sentence):
				xi = self.vocabularySets.GetMappedWordIfSatisfyUNK(token)
				bigram = (xi_1, xi)
				if self.bigrams.has_key(bigram):
					self.bigrams[bigram] += 1
				else:
					self.bigrams[bigram] = 1

				if enableKatzBackOff:
					v = (bigram[0])
					if (self.katzBackOff.has_key(v)):
						if bigram[1] not in self.katzBackOff[v]:
							self.katzBackOff[v].append(bigram[1])
					else:
						self.katzBackOff[v] = [bigram[1]]

				xi_1 = xi
				self.totalCount += 1
		endTime = time.clock()
		logging.debug("Bigram generation time = %f", (endTime - startTime))

	def GetWordsPrefixWithV(self, v):
		if (self.katzBackOff.has_key(v)):
			return self.katzBackOff[v]
		return []
	def GetCount(self, bigram) :
		if len(bigram) == 2 and self.bigrams.has_key(bigram) :
			return self.bigrams[bigram]

		return 0

	def GetBigramCount_Prefix_With_Unigram(self, unigram):
		if self.katzBackOff.has_key(unigram):
			return len(self.katzBackOff[unigram])
		return 0

class Unigram :
	unigrams = {}
	totalCount = 0

	def __init__(self, vocabularySets):
		self.unigrams = {}
		self.totalCount = 0
		self.vocabularySets = vocabularySets

	def GenerateUigrams(self, tokenGenerator):
		startTime = time.clock()
		xi = '*'
		self.unigrams[('*')] = 0
		self.unigrams[('UNK')] = 1
		for sentence in tokenGenerator.GetNextSentence():
			#Handling start word
			self.unigrams[('*')] += 1
			for token in tokenGenerator.SplitSentence(sentence):
				xi = self.vocabularySets.GetMappedWordIfSatisfyUNK(token)
				unigram = (xi)
				if self.unigrams.has_key(unigram):
					self.unigrams[unigram] += 1
				else:
					self.unigrams[unigram] = 1

				self.totalCount += 1
		endTime = time.clock()
		logging.debug("Unigram generation time = %f", (endTime - startTime))

	def GetCount(self, unigram):
		if (self.unigrams.has_key(unigram)):
			return self.unigrams[unigram]
		return 0

	def GetTotalCount(self):
		return self.totalCount

class Trigram :
	trigrams = {}
	totalCount = 0

	def __init__(self, vocabularySets):
		self.trigrams = {}
		self.trigramCount = 0
		self.vocabularySets = vocabularySets
		self.katzBackOff_u_v = {}

	def GenerateTrigram(self, tokenGenerator,  enableKatzBackOff = False):
		startTime = time.clock()
		self.trigrams[('UNK', 'UNK', 'UNK')] = 1
		for sentence in tokenGenerator.GetNextSentence():
			xi_2 = "*"
			xi_1 = "*"
			xi = "*"
			for token in tokenGenerator.SplitSentence(sentence):
				xi = self.vocabularySets.GetMappedWordIfSatisfyUNK(token)
				trigram = (xi_2, xi_1, xi)
				if (self.trigrams.has_key(trigram)):
					self.trigrams[trigram] = self.trigrams[trigram] + 1
				else:
					self.trigrams[trigram] = 1

				if (enableKatzBackOff):
					u_v = (trigram[0], trigram[1])
					if self.katzBackOff_u_v.has_key(u_v):
						if not (trigram[2] in self.katzBackOff_u_v[u_v]):
							self.katzBackOff_u_v[u_v].append(trigram[2])
					else:
						self.katzBackOff_u_v[u_v] = [trigram[2]]

				#update the trigram triplets
				xi_2 = xi_1
				xi_1 = xi
				xi = 'STOP'
				self.totalCount += 1
		endTime = time.clock()
		logging.debug("Trigram Generation Time = %f", (endTime - startTime))
	
	def GetAllWFromPrefixBigram(self, bigram):
		if self.katzBackOff_u_v.has_key(bigram):
			return self.katzBackOff_u_v[bigram]
		return []

	def GetCount(self,trigram):
		if (len(trigram) == 3) and self.trigrams.has_key(trigram):
			return self.trigrams[trigram]
		return 0;

	def GetTrigramCount_PrefixWith_Bigram(self, bigram):
		if (self.katzBackOff_u_v.has_key(bigram)):
			return len(self.katzBackOff_u_v[bigram])	
		return 0
"""
This class Estimates MLE of unigram, bigram and trigram. If the word is un-seen, it first map to UNK, then generate the MLE
"""
class MLEEstimator:
	def __init__(self, unigramObj, bigramObj, trigramObj, vocabularySets):
		self.unigram = unigramObj
		self.bigram = bigramObj
		self.trigram = trigramObj
		self.vocabularySets = vocabularySets

	def GetMLEOfTrigram(self, trigram):
		if (len(trigram) != 3):
			print("Invalid Trigram")
			return 0.0

		newTrigramWithUNKMapping = ['UNK', 'UNK', 'UNK']

		newTrigramWithUNKMapping[0] = self.vocabularySets.GetMappedWordIfSatisfyUNKForMLE(trigram[0])
		newTrigramWithUNKMapping[1] = self.vocabularySets.GetMappedWordIfSatisfyUNKForMLE(trigram[1])
		newTrigramWithUNKMapping[2] = self.vocabularySets.GetMappedWordIfSatisfyUNKForMLE(trigram[2])
		trigramCount = self.trigram.GetCount((newTrigramWithUNKMapping[0], newTrigramWithUNKMapping[1], newTrigramWithUNKMapping[2]))

		if (trigramCount == 0):
			return 0.0;

		newBigramWithUNKMapping = ['UNK', 'UNK']
		newBigramWithUNKMapping[0] = self.vocabularySets.GetMappedWordIfSatisfyUNKForMLE(trigram[0])
		newBigramWithUNKMapping[1] = self.vocabularySets.GetMappedWordIfSatisfyUNKForMLE(trigram[1])
		bigramCount = self.bigram.GetCount((newBigramWithUNKMapping[0], newBigramWithUNKMapping[1]))

		if (bigramCount == 0):
			return 0.0;

		return float(trigramCount)/bigramCount

	def GetMLEOfBigram(self, bigram):
		if (len(bigram) != 2):
			print("Invalid Bigram")
			return 0.0

		newBigramWithUNKMapping = ['UNK', 'UNK']
		newBigramWithUNKMapping[0] = self.vocabularySets.GetMappedWordIfSatisfyUNKForMLE(bigram[0])
		newBigramWithUNKMapping[1] = self.vocabularySets.GetMappedWordIfSatisfyUNKForMLE(bigram[1])
		bigramCount = self.bigram.GetCount((newBigramWithUNKMapping[0], newBigramWithUNKMapping[1]))

		if (bigramCount == 0):
			return 0.0;

		unigramCount = self.unigram.GetCount(self.vocabularySets.GetMappedWordIfSatisfyUNKForMLE(bigram[0]))

		return float(bigramCount)/unigramCount

	def GetMLEOfUnigram(self, unigram):
		unigramCount = self.unigram.GetCount(unigram)
		unigramWithUNKMapping = ('UNK')
		if (unigramCount == 0):
			unigramCount = self.unigram.GetCount(unigramWithUNKMapping)

		return float(unigramCount)/self.unigram.GetTotalCount()
"""
This is class for training, and calculating perplexity of linear interpolation method.
"""
class LinearInterpolationLanguageModel:
	"""docstring for LinearInterpolationLanguageModel"""
	def __init__(self, trainingSetFile, validationSetFile, testSetFile):
		startTime = time.clock()
		self.trainingSetTokenGenerator = TokenGenerator(trainingSetFile)
		self.validationSetTokenGenerator = TokenGenerator(validationSetFile)
		self.testSetTokenGenerator = TokenGenerator(testSetFile)
		self.vocabularySets = VocabularySets()
		self.vocabularySets.GenerateVocubularyFromTrainingSet(self.trainingSetTokenGenerator)
		self.vocabularySets.GeneratePercentageOfUnkWordInValidationSet(self.validationSetTokenGenerator)
		self.vocabularySets.GenerateUNKWord(self.validationSetTokenGenerator)
		self.vocabularySets.GenerateVocubularyFromTrainingSet(self.trainingSetTokenGenerator, True)
		self.trigramMLA = Trigram(self.vocabularySets)
		self.bigramMLA = Bigram(self.vocabularySets)
		self.unigramMLA = Unigram(self.vocabularySets)
		self.mleEstimator = MLEEstimator(self.unigramMLA, self.bigramMLA, self.trigramMLA, self.vocabularySets)
		self.lambda1 = 0.5
		self.lambda2 = 0.3
		self.lambda3 = 0.2
		endTime = time.clock()
		logging.debug("Initialization Time = %f", (endTime - startTime))

	"""
	This is function for estimating lambda using grid search.
	"""
	def EstimateLambdasParamter(self):
		#For time being. Chooosing the random value
		#Use grid search algorithm to estimate lambadas parameter.
		self.lambda1 = 0.5
		self.lambda2 = 0.3
		self.lambda3 = 0.2
		minimumPerplexity = 50000.0
		interval = [0.8, 0.6, 0.4, 0.2]
		for lambda1 in interval:
			for lambda2 in interval:
				if (lambda1 + lambda2 < 0.9):
					lambda3 = 1 - lambda1 - lambda2
					perplexity = self.CalculatePerplexity(self.validationSetTokenGenerator)
					print("lambda1 = %f, lambda2 = %f lambda3 = %f perplexity = %f" %(lambda1, lambda2, lambda3, perplexity))
					if minimumPerplexity > perplexity:
						minimumPerplexity = perplexity
						self.lambda1 = lambda1
						self.lambda2 = lambda2
						self.lambda3 = lambda3
		print("Optimized perplexity = (%f, %f, %f" %(self.lambda1, self.lambda2, self.lambda3))

	"""
	This function train the model by generating Unigram, bigram and trigram
	"""
	def TrainModelUsingTrainingSet(self):
		#Generate the Maximum likelihood of unigram, bigram and trigram parameter.
		startTime = time.clock()
		logging.debug("Generating trigram using training set")
		self.trigramMLA.GenerateTrigram(self.trainingSetTokenGenerator)
		logging.debug("Generating bigram using training Set")
		self.bigramMLA.GenerateBigram(self.trainingSetTokenGenerator)
		logging.debug("Generating unigram using training")
		self.unigramMLA.GenerateUigrams(self.trainingSetTokenGenerator)
		endTime = time.clock()
		#Next step estimate the parameter of lambda1, lambda2, lambda3
		logging.debug("Training model time = %f", (endTime - startTime))
	"""
	This calculates the ML probabiltiy of each trigram
	"""
	def EstimateLikeliHoodOfTrigramUsingInterpolation(self, trigram):
		startTime = time.clock()
		trigramMLEEstimate = self.mleEstimator.GetMLEOfTrigram(trigram)
		bigramMLEEstimate = self.mleEstimator.GetMLEOfBigram((trigram[1], trigram[2]))
		unigramMLEEstimate = self.mleEstimator.GetMLEOfUnigram((trigram[2]))
		qEstimate = self.lambda1 *	trigramMLEEstimate + \
		self.lambda2 * bigramMLEEstimate+ \
		self.lambda3 * unigramMLEEstimate
		endTime = time.clock()
		logging.debug("MLE Eastiagte of Trigram  = %f  Bigram = %f Unigram = %f, Sentence = %f and Time = %f", trigramMLEEstimate, bigramMLEEstimate, unigramMLEEstimate, qEstimate, (endTime - startTime))
		return math.log(qEstimate, 2)

	"""
	This calculates the perplexity of corpora on trained model.
	"""
	def CalculatePerplexity(self, tokenGenerator):	
		l = 0.0
		startTime = time.clock()
		sentenceCount = 1
		for sentence in tokenGenerator.GetNextSentence():
			sentenceProbalityDistribution = self.EstimateTheLikelieHoodOfSentence(sentence)
			logging.debug("Sentence = %d probability distribution = %f", sentenceCount, sentenceProbalityDistribution)
			l += sentenceProbalityDistribution
			sentenceCount += 1
		l = l/ tokenGenerator.GetCorpusSize()

		perplexity = math.pow(2, -l)
		endTime = time.clock()
		logging.debug("Perplexity computation time = %f", (endTime - startTime))
		return perplexity		
	"""
	This estimates the probability of sentence using interpolation method
	"""
	def EstimateTheLikelieHoodOfSentence(self, sentence):
		#broke it into tokes
		#Calculate the probability estimation of this model using linear interpolation.
		tokenGenerator = TokenGenerator('', [sentence])
		logging.debug("MLE of sentence %s", sentence)
		xi_2 = '*'
		xi_1 = '*'
		xi = '*'
		sentenceProbalityDistribution = 1.0
		for token in tokenGenerator.GetNextToken():
			xi = token
			logging.debug("Estimate MLE of Trigram %s,%s,%s", xi_2, xi_1, xi)
			sentenceProbalityDistribution += self.EstimateLikeliHoodOfTrigramUsingInterpolation((xi_2, xi_1, xi))
			xi_2 = xi_1
			xi_1 = xi
		return sentenceProbalityDistribution

	"""
	This trin and generate the model
	"""
	def TrainAndTestLanguageModel(self):
		self.TrainModelUsingTrainingSet()
		#self.EstimateLambdasParamter();
		perplexity = self.CalculatePerplexity(self.testSetTokenGenerator)
		print("Perplexity %f" %perplexity)

"""
This class trains and calculate perplexity of KatzBackOffModel
"""
class KatzBackOffModel:
	"""
	This function calculates missing probability on given trigram(u,v,w)
	This is used for KatzBackOffModel for trigram probability calculation when B(u,v,w) = 0
	"""
	def calculate_Alpha_u_v(self, u_v):
		startTime = time.clock()
		value = float(self.katzBackOffDiscount * self.trigramMLA.GetTrigramCount_PrefixWith_Bigram(u_v))
		unkBigram  = ('UNK', 'UNK')
		if value == 0:
			unkBigramCount = self.bigramMLA.GetCount(unkBigram)
			if unkBigramCount == 0:
				return 0
			value = float(self.katzBackOffDiscount * self.trigramMLA.GetTrigramCount_PrefixWith_Bigram(unkBigram))/self.bigramMLA.GetCount(unkBigram)
		else:
			value = value/self.bigramMLA.GetCount(u_v)
		endTime = time.clock()
		logging.debug("alpha v computation time = %f", (endTime - startTime))
		return value
	"""
	This function calculates the missing mass of Bigram Katz model
	"""
	def calculate_Alpha_v(self, v):
		countv = self.unigramMLA.GetCount(v)
		if countv == 0:
			countv = self.unigramMLA.GetCount('UNK')
		return float(self.katzBackOffDiscount * self.bigramMLA.GetBigramCount_Prefix_With_Unigram(v))/countv

	"""
	This estimates the probability of given bigram using Katz back off model
	"""
	def EstimateKatzBackOffForBigram(self, bigram):
		startTime = time.clock()
		if self.bigramKatzBackOffCache.has_key(bigram):
			return self.bigramKatzBackOffCache[bigram]

		v_w_BigramCount = self.bigramMLA.GetCount(bigram)
		bigramKatz = 0.0
		if v_w_BigramCount > 0:
			bigramKatz = float(v_w_BigramCount)/self.unigramMLA.GetCount((bigram[0]))
		else:
			alpha_v = self.calculate_Alpha_v((bigram[0]))
			pwValue = 1
			count_v = self.unigramMLA.GetCount((bigram[0]))
			for w in self.bigramMLA.GetWordsPrefixWithV(bigram[0]):
				pwValue = 1- float(self.unigramMLA.GetCount((w)))/count_v
			#todo - debug why we are getting 0 here.
			if pwValue == 0:
				pwValue = 1
			wprobility = float(self.unigramMLA.GetCount(bigram[1]))/count_v
			bigramKatz = alpha_v * (wprobility/pwValue)
		self.bigramKatzBackOffCache[bigram] = bigramKatz
		logging.debug("Bigram Katz probability = %f", bigramKatz)
		endTime = time.clock()
		logging.debug("Katz Bigram Estimation = %f", (endTime - startTime))
		return bigramKatz

	"""
	This estimates the probability of trigram using Katz back off model
	"""
	def EstimateTrigramProbabalityUsingKatzOffDiscountingMethod(self, trigram):
		startTime = time.clock()
		if (self.trigramKatzBackOffCache.has_key(trigram)):
			return self.trigramKatzBackOffCache[trigram]

		trigramCount = self.trigramMLA.GetCount(trigram)
		trigramKatz = 0.0
		if (trigramCount != 0):
			trigramKatz = float(trigramCount - self.katzBackOffDiscount)/self.bigramMLA.GetCount((trigram[0], trigram[1]))
			self.trigramKatzBackOffCache[trigram] = trigramKatz
		else:
			alpha_u_v = self.calculate_Alpha_u_v((trigram[0], trigram[1]))
			sum_Qd_w_v = 1.0
			bigram = (trigram[0], trigram[1])
			for w in self.trigramMLA.GetAllWFromPrefixBigram(bigram):
				sum_Qd_w_v -= self.EstimateKatzBackOffForBigram((trigram[1], w))
			
			if sum_Qd_w_v == 0:
				sum_Qd_w_v = 1
			trigramKatz = float(self.EstimateKatzBackOffForBigram((trigram[1], trigram[2])))/sum_Qd_w_v
			self.trigramKatzBackOffCache[trigram] = trigramKatz
		logging.debug("probability with KatzBackOffModel = %f", trigramKatz)
		endTime = time.clock()
		logging.debug("Katz Backoff Trigram Estimation time = %f", (endTime - startTime))
		return trigramKatz

	"""
	This function estimates the probability of sentence using katzBackOff model
	"""
	def EstimateTheLikelieHoodOfSentence(self, sentence):
		startTime = time.clock()
		tokenGenerator = TokenGenerator('', [sentence])
		logging.debug("MLE of sentence %s", sentence)
		xi_2 = '*'
		xi_1 = '*'
		xi = '*'
		sentenceProbalityDistribution = 1.0
		for token in tokenGenerator.GetNextToken():
			xi = self.vocabularySets.GetMappedWordIfSatisfyUNKForMLE(token)
			logging.debug("Estimate MLE of Trigram %s,%s,%s", xi_2, xi_1, xi)
			sentenceProbalityDistribution += self.EstimateTrigramProbabalityUsingKatzOffDiscountingMethod((xi_2, xi_1, xi))
			xi_2 = xi_1
			xi_1 = xi
		endTime = time.clock()
		logging.debug("Katz Estimation Time of Sentence = %f", (endTime - startTime))
		return sentenceProbalityDistribution
	"""
	This function computes the perplexity on given corpus using trained model
	"""
	def CalculatePerplexity(self, tokenGenerator):
		l = 0.0

		for sentence in tokenGenerator.GetNextSentence():
			sentenceProbalityDistribution = self.EstimateTheLikelieHoodOfSentence(sentence)
			logging.debug("Sentence probability distribution = %f", sentenceProbalityDistribution)
			l += sentenceProbalityDistribution

		l = l/ tokenGenerator.GetCorpusSize()

		perplexity = math.pow(2, -l)
		return perplexity
	"""
	This function should have used to generate the estimates of Katz discounting mass.
	Did not get time to try different value of discounting mass. But we could use grid search algorith by 
	using differrent value of discount mass like (0.2, .4....0.9)
	"""
	def EstimateKatZBackOffMissingMass(self):
		self.katzBackOffDiscount = 0.5

	"""
	This function test the model of katzBackOff model
	"""
	def TrainAndTestLanguageModel(self):
		self.trigramMLA.GenerateTrigram(self.trainingSetTokenGenerator, True)
		self.bigramMLA.GenerateBigram(self.trainingSetTokenGenerator, True)
		self.unigramMLA.GenerateUigrams(self.trainingSetTokenGenerator)

		self.EstimateKatZBackOffMissingMass();
		perplexity = self.CalculatePerplexity(self.testSetTokenGenerator)
		print("Perplexity %f" %perplexity)

	def __init__(self, trainingSetFile, validationSetFile, testSetFile):
		self.trainingSetTokenGenerator = TokenGenerator(trainingSetFile)
		self.validationSetTokenGenerator = TokenGenerator(validationSetFile)
		self.testSetTokenGenerator = TokenGenerator(testSetFile)
		self.vocabularySets = VocabularySets()
		self.vocabularySets.GenerateVocubularyFromTrainingSet(self.trainingSetTokenGenerator)
		self.vocabularySets.GeneratePercentageOfUnkWordInValidationSet(self.validationSetTokenGenerator)
		self.vocabularySets.GenerateUNKWord(self.validationSetTokenGenerator)
		#self.vocabularySets.GenerateVocubularyFromTrainingSet(self.trainingSetTokenGenerator, True)
		self.trigramMLA = Trigram(self.vocabularySets)
		self.bigramMLA = Bigram(self.vocabularySets)
		self.unigramMLA = Unigram(self.vocabularySets)
		self.trigramBackOff = {}
		self.bigramBackOff = {}
		self.trigramKatzBackOffCache = {}
		self.bigramKatzBackOffCache = {}
		self.katzBackOffDiscount = 0.5
		self.Count_W_Prefix_V_NotInTrainingSet = {}
"""
This function break the corpora into training set, validation set and test set according to given percentage
"""
def SplitCorpora(traingSetPercentage, validationSetPercentage, testSetPercentage, corporaFileName):
	if (traingSetPercentage + validationSetPercentage + testSetPercentage) != 100.0:
		print("Error - Sum of corpora percentage must be 100")
	# Current alogirthm would be simpley sequentially break the corpora

	fileObject = open(corporaFileName)
	lines = fileObject.readlines();
	trainingSetenceCount = int(traingSetPercentage/100 * len(lines))
	validationSentenceCount = int(validationSetPercentage/100 * len(lines))
	testSentenceCount = len(lines) - trainingSetenceCount - validationSentenceCount

	print("Number of lines in TrainingSet = %d  ValidationSet = %d TestSet = %d" %(trainingSetenceCount, validationSentenceCount, testSentenceCount))

	#Generate the new training file
	traingSetFileName = os.path.splitext(corporaFileName)[0] + '_trainingSet.txt';
	validationSetFileName = os.path.splitext(corporaFileName)[0] + '_validationSet.txt'
	testSetFileName = os.path.splitext(corporaFileName)[0] + '_testSet.txt'
	trainingFileObject = open(traingSetFileName, 'w')
	trainingFileObject.writelines(lines[0: trainingSetenceCount])
	trainingFileObject.close();
	validationFileObject = open(validationSetFileName, 'w')
	validationFileObject.writelines(lines[trainingSetenceCount:(validationSentenceCount + trainingSetenceCount)])
	validationFileObject.close()
	testFileObject = open(testSetFileName, 'w')
	testFileObject.writelines(lines[(trainingSetenceCount + validationSentenceCount) : len(lines)])
	testFileObject.close()
	print("Training Set Line count = %d"%len(open(traingSetFileName).readlines()))
	print("Validation Set Line count = %d"%len(open(validationSetFileName).readlines()))
	print("Test Set Line count = %d"%len(open(testSetFileName).readlines()))


#Interpolation langauge model.
def main():
	open('debug_2.log', 'w').close()
	logging.basicConfig(filename='debug_2.log',level=logging.DEBUG)
	SplitCorpora(80.0, 10.0, 10.0, 'reuters.txt')
	startTime = time.clock()

	validationSetFileName = 'brown_validationSet.txt'
	trainingSetFileName = 'brown_trainingSet.txt'
	testSetFileName = 'brown_testSet.txt'
	
	interpolationLangaugeModel = LinearInterpolationLanguageModel(trainingSetFileName, validationSetFileName, testSetFileName)
	interpolationLangaugeModel.TrainAndTestLanguageModel()

	#katzBackOffModel = KatzBackOffModel(trainingSetFileName, validationSetFileName, testSetFileName)
	#katzBackOffModel.TrainAndTestLanguageModel()
	endTime = time.clock();
	logging.debug("Total Execution time = %f", (endTime - startTime))
if __name__ == '__main__':
	main()





