package edu.berkeley.nlp.assignments;

import java.util.HashSet;
import java.util.List;
import java.util.Set;


import edu.berkeley.nlp.assignments.POSTaggerTester.LabeledLocalTrigramContext;
import edu.berkeley.nlp.assignments.POSTaggerTester.LocalTrigramContext;
import edu.berkeley.nlp.assignments.POSTaggerTester.LocalTrigramScorer;
import edu.berkeley.nlp.classify.ProbabilisticClassifier;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Counters;

public class HMMTagScorerWithSuffixTree extends HMMTagScorerWithoutUNKHandling {

	int suffixLength = 4;
	
	// If the word whose frequency is less than below specified will be considered for suffix tree smoothing
	int frequencyOfWordConsideredForSuffixSmoothing = 10;
	
	CounterMap<String, String> suffixTagCounter = new CounterMap<String, String>();
	
	Counter<String> suffixCounter = new Counter<String>();
	
	Counter<String> tagCounter = new Counter<String>();
	
	Set<String> trainingWordNotConsideredForSmoothing = new HashSet<String>();
	
	double unContextualTheta = 0.0;

	@Override
	public Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) {
		// TODO Auto-generated method stub
		int position = localTrigramContext.getPosition();
		
	    String word = localTrigramContext.getWords().get(position);
	    
	    Counter<String> tagCounter = unknownWordTags;
	    
	    if (wordsToTags.keySet().contains(word)) {
	      tagCounter = wordsToTags.getCounter(word);
	    }else{
	    	tagCounter = this.tagUnigramCounter;
	    }
	    
	    Counter<String> logScoreCounter = new Counter<String>();
	    for (String tag : tagCounter.keySet()) {
	        double trigramScore = GetTrigramMLETagScore(localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag(), tag);
	        double emissionProbability = GetEmissionProbabilit(word, tag);
	        //double logScore = Math.log(tagCounter.getCount(tag) * trigramScore);
	        double logScore = Math.log(trigramScore * emissionProbability);
	        //System.out.println(logScore + " ");
	        logScoreCounter.setCount(tag, logScore);

	    }
	    
	    return logScoreCounter;
	}

	@Override
	public void train(List<LabeledLocalTrigramContext> localTrigramContexts) {
		// TODO Auto-generated method stub
		GenerateWordsNotConsideredForSuffixSmoothing(localTrigramContexts);
		
		for (LabeledLocalTrigramContext labeledLocalTrigramContext : localTrigramContexts) {
	        String word = labeledLocalTrigramContext.getCurrentWord();
	        String tag = labeledLocalTrigramContext.getCurrentTag();
	        
	        wordsToTags.incrementCount(word, tag, 1.0);
	        
	        tagCounter.incrementCount(tag, 1.0);
	        
	        seenTagTrigrams.add(makeTrigramString(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag()));
	        
	        IncrementTrigramCount(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag());
	        
	        IncrementBigramCount(labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag());
	        
	        IncrementUnigramCount(labeledLocalTrigramContext.getCurrentTag());
	        
	        // Using this function, suffix tree map will be generated UNK category word. Frequency less than specified
	        GenerateSuffixTagMap(word, tag);
	        
	        totalWordCount++;
	      }
		
	      //wordsToTags = Counters.conditionalNormalize(wordsToTags);
	      
	      unknownWordTags = Counters.normalize(unknownWordTags);
	      
	      // Using tag uni-gram counter, contextual weight theta would be calculated, which will be used for emission probability
	      GenerateUnContextualWeight();
	      
	}

	@Override
	public void validate(List<LabeledLocalTrigramContext> localTrigramContexts) {
		// TODO Auto-generated method stub
		
	}
	
	protected double GetEmissionProbabilit(String word, String tag){
		if (this.trainingWordNotConsideredForSmoothing.contains(word)){
			return wordsToTags.getCount(word, tag);
		}else{
			return GetEmissionProbabilityUsinSuffixSmoothing(word, tag);
		}
	}
	
	private double GetSuffixMLEProbability(String suffix, String tag){
		if (suffix.isEmpty()){
			return 0.0;
		}
		
		Counter<String> suffixTagDistribution = this.suffixTagCounter.getCounter(suffix);
		
		double tagSuffixCount = suffixTagDistribution.getCount(tag);
		
		double suffixTagMLE = tagSuffixCount/suffixTagDistribution.totalCount();
		
		return (suffixTagMLE + this.unContextualTheta * GetSuffixMLEProbability(suffix.substring(1), tag))/(1 + this.unContextualTheta);
	}
	
	protected double GetEmissionProbabilityUsinSuffixSmoothing(String word, String tag){
		int maxLengthOfSuffix = Math.min(this.suffixLength, word.length());
		
		String suffix = word.substring(word.length() - maxLengthOfSuffix);
		
		double probabilityOfTagGivenSuffix = GetSuffixMLEProbability(suffix, tag);
		
		double probabilityOfSuffixGivenTag = (probabilityOfTagGivenSuffix * GetProbabilityOfSuffix(suffix))/GetProbablityOfTag(tag);
		
		return probabilityOfSuffixGivenTag;
	}
	
	protected double GetProbabilityOfSuffix(String suffix){
		return this.suffixTagCounter.getCounter(suffix).totalCount()/totalWordCount;
	}
	
	protected double GetProbablityOfTag(String tag){
		return this.tagUnigramCounter.getCount(tag)/this.tagUnigramCounter.totalCount();
	}
	
	protected void GenerateWordsNotConsideredForSuffixSmoothing(List<LabeledLocalTrigramContext> localTrigramContexts){
		Counter<String> wordFrequencyCounter = new Counter<String>();
		
		for (LabeledLocalTrigramContext localTrigramContext : localTrigramContexts){
			String word = localTrigramContext.getCurrentWord();
			wordFrequencyCounter.incrementCount(word, 1.0);
		}
		
		for (String word : wordFrequencyCounter.keySet()){
			double wordCount = wordFrequencyCounter.getCount(word);
			if (wordCount > this.frequencyOfWordConsideredForSuffixSmoothing){
				// Here word is not considered for smoothing
				this.trainingWordNotConsideredForSmoothing.add(word);
			}else{
				// Here word is considered for smoothing
			}
		}
	}
	
	protected void GenerateSuffixTagMap(String word, String tag){
		
		if (this.trainingWordNotConsideredForSmoothing.contains(word)){
			// This means this word should not be considered for suffix tree smoothing
			return;
		}
		
		for (int suffixLength = 1; suffixLength <= this.suffixLength; suffixLength++){
			if (word.length() > suffixLength){
				String suffix = word.substring(word.length() - suffixLength);
				suffixTagCounter.incrementCount(suffix, tag, 1.0);
				suffixCounter.incrementCount(suffix, 1.0);
			}	
		}
	}
	
	// Present this description in report. How mean probability is considered
	protected void GenerateUnContextualWeight(){
		// As per my understanding 
		double meanTagProbability = 1.0/tagUnigramCounter.size();
		
		Counter<String> normalizedTag = Counters.normalize(tagUnigramCounter);
		
		double standardDeviationSum = 0.0; 
		
		for(String tag : normalizedTag.keySet()){
			standardDeviationSum += Math.pow(normalizedTag.getCount(tag) - meanTagProbability, 2); 
		}
		
		this.unContextualTheta = standardDeviationSum/tagUnigramCounter.size();
	}

}
