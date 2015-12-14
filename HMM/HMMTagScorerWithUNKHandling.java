package edu.berkeley.nlp.assignments;

import java.sql.PseudoColumnUsage;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.omg.CORBA.INTERNAL;

import edu.berkeley.nlp.assignments.POSTaggerTester.LabeledLocalTrigramContext;
import edu.berkeley.nlp.assignments.POSTaggerTester.LocalTrigramContext;
import edu.berkeley.nlp.assignments.POSTaggerTester.LocalTrigramScorer;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Counters;
import edu.berkeley.nlp.util.Pair;

public class HMMTagScorerWithUNKHandling implements LocalTrigramScorer{

	boolean restrictTrigrams; // if true, assign log score of Double.NEGATIVE_INFINITY to illegal tag trigrams.

	// This is counter to measure count of pair <Word, Tag> - This will be used to find the emission probability.
    CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
    
    // This is counter to measure the UNK word tag.
    Counter<String> unknownWordTags = new Counter<String>();
    
    // List of all seen trigram tag in our training sentence.
    Set<String> seenTagTrigrams = new HashSet<String>();
    
    // This is the list of pseudoWords found while mapping training set word with less or equal than cutoff frequency. 
    // This map is required - Beacuse in test set, it might be possible, that word is mapped to date pseudo word, but none of word in our 
    // training set mapped to Date pseudo word. So we use to test, whether there exists some pseudo word, corresponding given UNK test word, if not,
    //  then assign it to pseudoWord "other". By this handling, we will never get, NEGATIVE.INFINITY probability for any UNK word.
    Set<String> pseudoWrodsFound = new HashSet<String>();
    
    // This is TAG tri-gram counter, used to measure HMM tag transition probability using linear interpolation method.
    Counter<String> tagTrigramCounter = new Counter<String>();
    
    // This is TAG Bigram counter, used to measure HMM tag transition probability using linear interpolation method.
    Counter<String> tagBigramCounter = new Counter<String>();
    
    // This is TAG Unigram counter, used to measure HMM tag transition probability using linear interpolation method.
    Counter<String> tagUnigramCounter = new Counter<String>();
    
    // This is counter of measure count of occurances of Each Tag in test sentence. This will be used to find the emission probability.
    Counter<String> tagCounter = new Counter<String>();
    
    // This is dictionary, which helps to find, whether given word is considered UNK word or not.
    Set<String> wordDictionaryExceptPsuedoWords = new HashSet<String>();
    
    // In training set, we use to cache, which words we mapped to pseduo word. Even in test sentence, for any UNK word mapped to pseudo word, we cache it
    // so that, if the same word appear in future, we dont have to recompute word classification.
    Map<String, String> unkWordToPseudoWordMap = new HashMap<String, String>();
    
    // Total Tag Count, which is equal to total word count.
    int totalTagCount = 0;
    
    // This is cut-off count. If the word frequency is less than specified value in our training set, then corresponding word will be replaced with 
    // pseudo word. 
    int pseudoWordClassifictionGroupCutoffCount = 3;
      
    //Feature
	
	public HMMTagScorerWithUNKHandling(boolean trigramHanldingRestriction){
		restrictTrigrams = trigramHanldingRestriction;
	}
	
	// In this function, we do following opeartions.
	// Replacing the training word with pseudo word, if their frequency is less than cut-off count. This will be used for UNK handling.
	// Counting tag trigram, bigram, unigram - Used to find HMM tag transition probability. 
	// Counting pair <word, tag> count - will be used to find emission probability.
	@Override
	public void train(List<LabeledLocalTrigramContext> localTrigramContexts) {
		// Generate the pseudo word classification group.
		// If the count of words is less than cutoff, that word would be mapped to pseudo group.
		GenerateClassficationGroupForUNKhandling(localTrigramContexts, pseudoWordClassifictionGroupCutoffCount);
		
		for (LabeledLocalTrigramContext labeledLocalTrigramContext : localTrigramContexts) {
			
			// Get the Mapped/Pseudo words
	        String word = GetMappedWordIfPossible(labeledLocalTrigramContext.getCurrentWord(), labeledLocalTrigramContext.getPosition(), true);
	        
	        String tag = labeledLocalTrigramContext.getCurrentTag();
	        
	        wordsToTags.incrementCount(word, tag, 1.0);
	        
	        tagCounter.incrementCount(tag, 1.0);
	        
	        seenTagTrigrams.add(makeTrigramString(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag()));
	        
	        // This is for Tag smoothening to find tag probabilities using HMM
	        IncrementTrigramCount(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag());
	        
	        IncrementBigramCount(labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag());
	        
	        IncrementUnigramCount(labeledLocalTrigramContext.getCurrentTag());
	        
	      }
		
		System.out.println("Model training completed");
		
	}
	
	// This is used to calculate the score of <word, tag> pair. this would be Log(TransitionProbability * Emission probability)
	@Override
    public Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) {
    	int position = localTrigramContext.getPosition();
    	
    	// Get mapped pseudo word. It will automatically handles UNK word.
	    String unMappedword = localTrigramContext.getWords().get(position);
	    
	    String word = GetMappedWordIfPossible(unMappedword, position, false);
	       
	    Counter<String> tagCounter = unknownWordTags;
	    if (wordsToTags.keySet().contains(word)) {
	      tagCounter = wordsToTags.getCounter(word);
	    }else{
	    	System.out.println("This part of code should not hit as after mapping. There should not be any unknown words after mapping");
	    }
	    
	    	    
	    Counter<String> logScoreCounter = new Counter<String>();
	    
	    for (String tag : tagCounter.keySet()) {
	    	
	    	// Transition probability
	        double trigramScore = GetTrigramMLETagScore(localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag(), tag);
	        
	        // Emission probability
	        double emissionProbability = tagCounter.getCount(tag)/this.tagCounter.getCount(tag);
	        
	        // Calculating the score of <word, tag> pair of given sentence.
	        double logScore = Math.log(emissionProbability * trigramScore);
	        
	        logScoreCounter.setCount(tag, logScore);
	      
	    }
	    
	    return logScoreCounter;
	}
	
	// Used to tune the parameter of lambdas.
	@Override
	public void validate(List<LabeledLocalTrigramContext> localTrigramContexts) {
		// TODO Auto-generated method stub
		
	}
	
	// Generate map for handling UNK words. 
    private void GenerateClassficationGroupForUNKhandling(List<LabeledLocalTrigramContext> localTrigramContexts, int cutOffCountValue){
    	Counter<String> wordCounter = new Counter<String>();
    	Map<String, Integer> wordPosition = new HashMap<String, Integer>();
    	for(LabeledLocalTrigramContext localTrigramContext : localTrigramContexts){
    		String word = localTrigramContext.getCurrentWord();
    		wordCounter.incrementCount(word, 1.0);
    		wordPosition.put(word, localTrigramContext.getPosition());
    	}
    	
    	for(String word : wordCounter.keySet()){
    		double countValueOfWord = wordCounter.getCount(word);
    		
    		// If the frequency of word in training set is less than cutoff value. It means it has chance of unknown word. So This word
    		// mapped to pseudo word
    		if (countValueOfWord <= cutOffCountValue){
				String pseudoWordGroup = ClassifyWords(word, wordPosition.get(word));
				unkWordToPseudoWordMap.put(word, pseudoWordGroup);
				if (!pseudoWrodsFound.contains(pseudoWordGroup))
    			{
    				pseudoWrodsFound.add(pseudoWordGroup);

    			}

    		}else{
    			wordDictionaryExceptPsuedoWords.add(word);
    		}
    	}
    }
    
    
    // This function returns mapped to pseudo word if the condition satisfies, otherwise returns same word.
    private String GetMappedWordIfPossible(String word, int wordPositionInSentence, boolean trainingTime){
    	if (wordDictionaryExceptPsuedoWords.contains(word)){
    		return word;
    	}else{
    		if (unkWordToPseudoWordMap.containsKey(word)){
    			return unkWordToPseudoWordMap.get(word);
    		}else{
    			String pseudoWord =  ClassifyWords(word, wordPositionInSentence);
    			
				if (!pseudoWrodsFound.contains(pseudoWord)){
					pseudoWord = "other";
				}
				
				System.out.println(" Umapped word  " + word + "  Mapped word " + pseudoWord);
				unkWordToPseudoWordMap.put(word, pseudoWord);
    			
    			// Caching so that to avoid recompute the classification group.
    			
    			return pseudoWord;
    		}
    	}
    }

    // Calculate the transition probability for given tag using linear interpolation method.
    private double GetTrigramMLETagScore(String previousPreviousTag, String previousTag, String currentTag){
    	String trigramTag = makeTrigramString(previousPreviousTag, previousTag, currentTag);
    	String bigramTag = makeBigramString(previousTag, currentTag);
    	double trigramCount = tagTrigramCounter.getCount(trigramTag);
    	double bigramCount = tagBigramCounter.getCount(bigramTag);
    	double unigramCount = tagUnigramCounter.getCount(currentTag);
    	
    	if (trigramCount != 0){
    		return 0.4 * (trigramCount/bigramCount) + 0.3 *(bigramCount/unigramCount) + 0.2 *(unigramCount/totalTagCount);
    	}
    	else if (bigramCount != 0){
    		return 0.3 * (bigramCount/unigramCount) + 0.2 *(unigramCount/totalTagCount);
    	}
    	else{
    		return 0.2 * unigramCount/totalTagCount;
    	}
    }
    
    // Returns the pseudo word group for given word.
	private String ClassifyWords(String word, int wordPositionInSentence){

		if (isTwoDigitNum(word)){
			return "twoDigitNum";
		}else if (isFourDigitNum(word)){
			return "fourDigitNum";
		}else if (isContainsDigitAndAlpha(word)){
			return "containsDigitAndAlph";
		}else if (isDate(word)){
			return "dateOrTime";
		}else if (isPercentage(word)){
			return "percentage";
		}else if (isRangeNumber(word)){
			return "numberRange";
		}else if (isContainsDigitAndPeriod(word)){
			return "containsDigitAndPeriod";
		}else if (isContainsDigitAndComma(word)){
			return "containsDigitAndComma";
		}else if (isNumberWithDash(word)){
			return "NumberDashWord";
		}else if (isAllCap(word)){
			return "allCaps";
		}else if (isInitCap(word) && wordPositionInSentence == 0){
			return "NameDeterminner";
		}else if (isInitCap(word)){
			return "initCap";
		}else if (isWordWitDash(word)){
			return "twoWordWithDash";
		}else if (isWordEndingWithIng(word)){
			return "wordEndingWithIng";
		}
		else if (isAllLowerCase(word)){
			return "lowerCase";
		}
		
		if (wordPositionInSentence == 0){
			return "firstWord";
		}
		
		return "other";
	}
	
	// This validate whether word with dash. For example -> up-down, 
	private boolean isWordWitDash(String word){
		String []words = word.split("-");
		if (words.length == 2){
			return isWord(words[0]) && isWord(words[1]);
		}
		
		return false;
	}
	
	// This validate whether word ending with ing for exmaple -> running, learning
	private boolean isWordEndingWithIng(String  word){
		if (word.toLowerCase().endsWith("ing")){
			return true;
		}
		
		return false;
	}
	
	// This validate, whether word is time or date. For example -> 09-96 or 3:15 or 10:20:30 or 11/9/89
	// I have not put strict restriction like - Month should be in range of 1 to 12 etc.
	private boolean isDate(String word){
		String words[] = word.split("-");
		
		if (words.length == 2 && isInteger(words[0]) && isInteger(words[1]) && isTwoDigitNum(words[1]) &&
				(isFourDigitNum(words[0]) || isTwoDigitNum(words[0]))){
			return true;
		}
		
		words = word.split(":");
		
		if (words.length == 2 && isInteger(words[0]) && isInteger(words[1])){
			return true;
		}else if (words.length == 3 && isInteger(words[0]) && isInteger(words[1]) && isInteger(words[2])){
			return true;
		}
		
		words = word.split("/");
		if (words.length == 2 && isInteger(words[0]) && isInteger(words[1])){
			return true;
		}else if (words.length == 3 && isInteger(words[0]) && isInteger(words[1]) && isInteger(words[2])){
			return true;
		}
		
		return false;
	}
	
	// This validates whetehr number is with Dash For example -> 01-02
	private boolean isNumberWithDash(String word){
		String []words = word.split("-");
		if (words.length == 2){
			return isInteger(words[0]) & isWord(words[1]);
		}
		
		return false;
	}
	
	// This validates, whether word is two digit number or not for example - 12
	private boolean isTwoDigitNum(String word){
		// Smart checking would be validating the actual value lies between 10 to 99. But we don't require this types of handling in training data.
		if (word.length() == 2 && isInteger(word)){
			return true;
		}
		
		return false;
	}
    
	
	private boolean isWord(String word){
		for (int index = 0 ; index < word.length(); index++){
			if (!(
					isLowerCase(word.charAt(index)) ||
					isUpperCase(word.charAt(index)) 
					)){
				return false;
			}
		}
		
		return true;
	}
	
	private boolean isFourDigitNum(String word){
		if (word.length() == 4 && isInteger(word)){
			return true;
		}
		
		return false;
	}
	
	private boolean isPercentage(String word){
		String []words = word.split("\\.");
		if (words.length == 2 && isInteger(words[0]) && isInteger(words[1]) && Integer.parseInt(words[0]) < 100){
			return true;
		}
		
		return false;
	}
	
	// There can be n number of possiblity
	private boolean isContainsDigitAndAlpha(String word){
		boolean containsDash = false;
		boolean containsDigit = false;
		boolean containsUpperCase = false;
		for (int index = 0; index < word.length(); index++){
			char tempChar = word.charAt(index);
			if (isOneDigitNumber(tempChar)){
				containsDigit = true;
			}else if (isUpperCase(tempChar)){
				containsUpperCase = true;
			}else if (tempChar == '-'){
				containsDash = true;
			}else{
				return false;
			}
		}
		
		return containsDash & containsDigit & containsUpperCase;
	}
	
	private boolean isRangeNumber(String word){
		String []words = word.split("-");
		if (words.length == 2 && isInteger(words[0]) && isInteger(words[1])){
			return true;
		}
		
		return false;
	}
	
	private boolean isContainsDigitAndDash(String word){
		return false;
	}
	
	private boolean isContainsDigitAndPeriod(String word){
		String []words = word.split("\\.");
		if (words.length == 2 && isInteger(words[0]) && isInteger(words[1])){
			return true;
		}
		
		return false;
	}
	
	private boolean isContainsDigitAndComma(String word){
		String []words = word.split("\\.");
		if (words.length > 2){
			return false;
		}else if (words.length == 2 && isInteger(words[1])){
			return isMoneyWithoutFractionPart(words[0]);
		}else if (words.length == 1){
			return isMoneyWithoutFractionPart(words[0]);
		}
		
		return false;
	}
	
	private boolean isMoneyWithoutFractionPart(String word){
		String []br_digits = word.split(",");
		for (String digits : br_digits){
			if (!isInteger(digits)){
				return false;
			}
		}
		
		return true;
	}
	
	private boolean isOtherNumber(String word){
		return false;
	}
	
	private boolean isAllCap(String word){
		for (int index = 0; index < word.length(); index++){
			if (word.charAt(index) < 'A' || word.charAt(index) > 'Z'){
				return false;
			}
		}
		
		return true;
	}
	
	private boolean isCapPeriod(String word){
		if (word.length() == 2 && isPeriod(word.charAt(1)) && isUpperCase(word.charAt(0))){
			return true;
		}
		
		return false;
	}
	
	private boolean isInitCap(String word){
		if (word.length() < 2) {
			return false;
		}
		
		if (isUpperCase(word.charAt(0)) && isAllLowerCase(word.substring(1))){
			return true;
		}
		
		return false;
	}
	
	private boolean isAllLowerCase(String word){
		for (int index= 0; index < word.length(); index++){
			if (!isLowerCase(word.charAt(index))){
				return false;
			}
		}
		
		return true;
	}
	
	private boolean isUpperCase(char c){
		if (c < 'A' || c >'Z'){
			return false;
		}
		
		return true;
	}
	
	private boolean isLowerCase(char c){
		if (c < 'a' || c > 'z'){
			return false;
		}
		
		return true;
	}
	
	private boolean isOneDigitNumber(char c){
		if (c < '0' || c> '9'){
			return false;
		}
		
		return true;
	}
	
	private boolean isPeriod(char c){
		return c == '.';
	}
	
	private boolean isInteger(String word){
		try{
			Integer.parseInt(word);
			return true;
		}catch(NumberFormatException ex){
			return false;
		}
	}
	
	private String makeTrigramString(String previousPreviousTag, String previousTag, String currentTag) {
	      return previousPreviousTag + " " + previousTag + " " + currentTag;
	}
	    
    private String makeBigramString(String previousPreviousTag, String previousTag){
    	return previousPreviousTag + " " + previousTag;
    }
    
    private void IncrementUnigramCount(String currentTag){
    	tagUnigramCounter.incrementCount(currentTag, 1.0);
    	totalTagCount++;
    }
   
    private void IncrementBigramCount(String previousTag, String currentTag){
    	String bigramString = makeBigramString(previousTag, currentTag);
    	tagBigramCounter.incrementCount(bigramString, 1.0);
    }
    
    private void IncrementTrigramCount(String previousPreviousTag, String previousTag, String currentTag){
    	String trigramString = makeTrigramString(previousPreviousTag, previousTag, currentTag);
    	tagTrigramCounter.incrementCount(trigramString, 1.0);
    }
	    

	private Set<String> allowedFollowingTags(Set<String> tags, String previousPreviousTag, String previousTag) {
	      Set<String> allowedTags = new HashSet<String>();
	      for (String tag : tags) {
	        String trigramString = makeTrigramString(previousPreviousTag, previousTag, tag);
	        if (seenTagTrigrams.contains((trigramString))) {
	          allowedTags.add(tag);
	        }
	      }
	      return allowedTags;
	    }

}
