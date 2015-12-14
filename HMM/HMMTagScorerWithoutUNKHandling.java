package edu.berkeley.nlp.assignments;

import java.nio.channels.GatheringByteChannel;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.assignments.POSTaggerTester.LabeledLocalTrigramContext;
import edu.berkeley.nlp.assignments.POSTaggerTester.LocalTrigramContext;
import edu.berkeley.nlp.assignments.POSTaggerTester.LocalTrigramScorer;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Counters;

public class HMMTagScorerWithoutUNKHandling implements LocalTrigramScorer{

	boolean restrictTrigrams; // if true, assign log score of Double.NEGATIVE_INFINITY to illegal tag trigrams.

    CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
    
    Counter<String> unknownWordTags = new Counter<String>();
    
    Set<String> seenTagTrigrams = new HashSet<String>();
    
    Counter<String> tagTrigramCounter = new Counter<String>();
    
    Counter<String> tagBigramCounter = new Counter<String>();
    
    Counter<String> tagUnigramCounter = new Counter<String>();
    
    Counter<String> tagCounter = new Counter<String>();
    
    int totalTagCount = 0;
    
    int totalWordCount = 0;
	    
    public Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) {
	    int position = localTrigramContext.getPosition();
	    String word = localTrigramContext.getWords().get(position);
	    Counter<String> tagCounter = unknownWordTags;
	    if (wordsToTags.keySet().contains(word)) {
	      tagCounter = wordsToTags.getCounter(word);
	    }
	    Set<String> allowedFollowingTags = allowedFollowingTags(tagCounter.keySet(), localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag());
	    Counter<String> logScoreCounter = new Counter<String>();
	    for (String tag : tagCounter.keySet()) {
	      if (!restrictTrigrams || allowedFollowingTags.isEmpty() || allowedFollowingTags.contains(tag)){
	        double trigramScore = GetTrigramMLETagScore(localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag(), tag);
	        //double logScore = Math.log(tagCounter.getCount(tag) * trigramScore);
	        
	        double emissionProbability = tagCounter.getCount(tag)/this.tagCounter.getCount(tag);
	        
	        double logScore = Math.log(emissionProbability * trigramScore);
	        //System.out.println(logScore + " ");
	        logScoreCounter.setCount(tag, logScore);
	      }
	      else{
	    	  System.out.println("Could not able find");
	      }
	    }
	    return logScoreCounter;
	  }

	public HMMTagScorerWithoutUNKHandling(){
		
	}
	
	
	@Override
	public void train(List<LabeledLocalTrigramContext> localTrigramContexts) {
		// TODO Auto-generated method stub
		for (LabeledLocalTrigramContext labeledLocalTrigramContext : localTrigramContexts) {
	        String word = labeledLocalTrigramContext.getCurrentWord();
	        String tag = labeledLocalTrigramContext.getCurrentTag();
	        if (!wordsToTags.keySet().contains(word)) {
	          // word is currently unknown, so tally its tag in the unknown tag counter
	          unknownWordTags.incrementCount(tag, 1.0);
	        }
	        
	        wordsToTags.incrementCount(word, tag, 1.0);
	        tagCounter.incrementCount(tag, 1.0);
	        
	        seenTagTrigrams.add(makeTrigramString(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag()));
	        
	        IncrementTrigramCount(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag());
	        IncrementBigramCount(labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag());
	        IncrementUnigramCount(labeledLocalTrigramContext.getCurrentTag());
	      }
		
	      //wordsToTags = Counters.conditionalNormalize(wordsToTags);
	      unknownWordTags = Counters.normalize(unknownWordTags);
		
	}

	@Override
	public void validate(List<LabeledLocalTrigramContext> localTrigramContexts) {
		// TODO Auto-generated method stub
		
	}
	

	public HMMTagScorerWithoutUNKHandling(boolean trigramHanldingRestriction){
		restrictTrigrams = trigramHanldingRestriction;
	}
	
	protected Set<String> allowedFollowingTags(Set<String> tags, String previousPreviousTag, String previousTag) {
	      Set<String> allowedTags = new HashSet<String>();
	      for (String tag : tags) {
	        String trigramString = makeTrigramString(previousPreviousTag, previousTag, tag);
	        if (seenTagTrigrams.contains((trigramString))) {
	          allowedTags.add(tag);
	        }
	      }
	      return allowedTags;
	    }
	
	protected String makeTrigramString(String previousPreviousTag, String previousTag, String currentTag) {
	      return previousPreviousTag + " " + previousTag + " " + currentTag;
	}
  
	protected String makeBigramString(String previousPreviousTag, String previousTag){
		return previousPreviousTag + " " + previousTag;
	}
  
	protected void IncrementUnigramCount(String currentTag){
		tagUnigramCounter.incrementCount(currentTag, 1.0);
		totalTagCount++;
	}
	 
	protected void IncrementBigramCount(String previousTag, String currentTag){
	  	String bigramString = makeBigramString(previousTag, currentTag);
	  	tagBigramCounter.incrementCount(bigramString, 1.0);
	}
	  
	protected void IncrementTrigramCount(String previousPreviousTag, String previousTag, String currentTag){
		String trigramString = makeTrigramString(previousPreviousTag, previousTag, currentTag);
		tagTrigramCounter.incrementCount(trigramString, 1.0);
	}
	    
  protected double GetTrigramMLETagScore(String previousPreviousTag, String previousTag, String currentTag){
  	String trigramTag = makeTrigramString(previousPreviousTag, previousTag, currentTag);
  	String bigramTag = makeBigramString(previousTag, currentTag);
  	double trigramCount = tagTrigramCounter.getCount(trigramTag);
  	double bigramCount = tagBigramCounter.getCount(bigramTag);
  	double unigramCount = tagUnigramCounter.getCount(currentTag);
  	
  	if (trigramCount != 0){
  		return 0.4 * (trigramCount/bigramCount) + 0.3 *(bigramCount/unigramCount) + 0.3 *(unigramCount/totalTagCount);
  	}
  	else if (bigramCount != 0){
  		return 0.6 * (bigramCount/unigramCount) + 0.2 *(unigramCount/totalTagCount);
  	}
  	else{
  		return 0.2 * unigramCount/totalTagCount;
  	}
  }

	    

}
