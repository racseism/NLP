package edu.berkeley.nlp.assignments.decoding.student;

import java.util.Arrays;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;

/***
 * This defines state, as per collins note.
 * @author rajkuma
 *
 */
  public class DecodeState implements Comparable<DecodeState>{
	  
	public static int startSymbolIndex = EnglishWordIndexer.getIndexer().addAndGetIndex(NgramLanguageModel.START);
	
	public static int stopSymbolIndex = EnglishWordIndexer.getIndexer().addAndGetIndex(NgramLanguageModel.STOP);
	  
	public int e1WordIndex;
	
	public int e2WordIndex;
	
	public double score;
	
	ScoredPhrasePairForSentence phrasedSentence;
	
	DecodeState previousState;
	
	public int phraseEndIndex = -1;
	
	char []phraseFlag;
	
	int countOfSetFlag = 0;
	
	boolean flagBitEnabled = false;
	
	public DecodeState(int phraseLength) {
		// TODO Auto-generated constructor stub
		this.e1WordIndex = -1;
		this.e2WordIndex = -1;
		this.score = 0;
		this.previousState = null;
		this.phraseFlag = new char[phraseLength];
		Arrays.fill(phraseFlag, '0');
		this.flagBitEnabled = true;
	}
	
	public DecodeState(){
		this(0);
		this.flagBitEnabled = false;
	}
	
	public DecodeState(double score, ScoredPhrasePairForSentence phrasedSentence, DecodeState previousState){
		int [] phraseWordIndexer = phrasedSentence.english.indexedEnglish;
		if (phraseWordIndexer.length > 1){
			this.e1WordIndex = phraseWordIndexer[phraseWordIndexer.length - 2];
			this.e2WordIndex = phraseWordIndexer[phraseWordIndexer.length - 1];
		}else if (phraseWordIndexer.length == 1){
			this.e1WordIndex = previousState.e2WordIndex;
			this.e2WordIndex = phraseWordIndexer[phraseWordIndexer.length - 1];
		}else{
			this.e1WordIndex = previousState.e1WordIndex;
			this.e2WordIndex = previousState.e2WordIndex;
		}
		
		this.score = score;
		this.phrasedSentence = phrasedSentence;
		this.previousState = previousState;
		this.phraseEndIndex = phrasedSentence.getEnd() - 1;
		if (previousState.flagBitEnabled){
			this.phraseFlag = previousState.phraseFlag.clone();
			for (int index = phrasedSentence.getStart(); index < phrasedSentence.getEnd(); index++){
				if (this.phraseFlag[index] == '0'){
					this.phraseFlag[index] = '1';
				}else{
					System.out.println("There is some bug. Please fix it.");
				}
			}
			
			this.flagBitEnabled = true;
			this.countOfSetFlag = previousState.countOfSetFlag + phrasedSentence.getForeignLength();
		}
	}
	
	public DecodeState(double score, ScoredPhrasePairForSentence phrasedSentence, DecodeState previousState, int phraseEndIndex){
		this(score,phrasedSentence, previousState);
		this.phraseEndIndex = phraseEndIndex;
	}

	@Override
	public int compareTo(DecodeState o) {
		// TODO Auto-generated method stub
		
		return Double.compare(o.score, this.score);
	}
	
	public String toString(){
		String output = this.e1WordIndex + "_" + this.e2WordIndex + "_";
		for (char flag : this.phraseFlag){
			output += flag + "_";
		}
		
		output += this.phraseEndIndex;
		
		return output;
	}
}