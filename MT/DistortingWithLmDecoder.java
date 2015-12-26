package edu.berkeley.nlp.assignments.decoding.student;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.swing.text.AbstractDocument.LeafElement;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;
import edu.berkeley.nlp.util.Pair;

public class DistortingWithLmDecoder implements Decoder{

	PhraseTable phraseTable;
	
	NgramLanguageModel languageModel;
	
	DistortionModel distortionModel;
	
	//Change distortion limit according to your convenience. By Default I set to 0. In case of 0, it should match with Decoder 2.
	int distortionLimit = 0;
	
	double distorationWeight = -20.0;
	
	public static boolean debugFlagEnabled = false;
	
	public DistortingWithLmDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm) {
		// TODO Auto-generated constructor stub
		this.phraseTable = tm;
		this.languageModel = lm;
		this.distortionModel = new CustomDistortionModel(distortionLimit, distorationWeight);
		
	}
	
	@Override
	public List<ScoredPhrasePairForSentence> decode(List<String> frenchSentence) {
		// TODO Auto-generated method stub
		PhraseTableForSentence sentencePhraseTranslation = phraseTable.initialize(frenchSentence);
		
		int length = frenchSentence.size();
		
		BeamSearchQueue decoderStates[] = new BeamSearchQueue[frenchSentence.size() + 1];
		
		//Initializing the beam search queue
		for (int index = 0; index < decoderStates.length; index++){
			decoderStates[index] = new BeamSearchQueue();
		}
		
		// Here each queue keeps track of number of number of phrase already translated.
		
		//Decoder.StaticMethods.scoreHypothesis(hyp, languageModel, dm)
		
		//Qo state.
		DecodeState q0State = new DecodeState(length);
		decoderStates[0].add(q0State);
		
		int maxPhraseLength = sentencePhraseTranslation.getMaxPhraseLength();
		Map<String,String> alreadyDoneMap = new HashMap<String, String>();
		
		for (int wordIndex = 0; wordIndex < length; wordIndex++){
			DecodeState currentState = decoderStates[wordIndex].getNext();
			int maxLength = Math.min(maxPhraseLength, length - wordIndex);
			while(currentState != null){
					List<ScoredPhrasePairForSentence> nextPhrases = GetNextPhrasePairs(currentState, this.distortionLimit, sentencePhraseTranslation);
					if (nextPhrases.size() == 0){
						//System.out.println("You need to debug to understand why the next phrase size is 0");
					}
					
					List<DecodeState> nextStates = GetNextStates(currentState, nextPhrases);
					for(DecodeState nextState : nextStates){
						decoderStates[nextState.countOfSetFlag].add(nextState);
					}
					
				currentState = decoderStates[wordIndex].getNext();
			}
			
			alreadyDoneMap = new HashMap<String,String>();
			for (int index = wordIndex +2; index < decoderStates.length; index++){
				decoderStates[index].PruneOfBasisOfSize();
			}
		}
		
		//Getting back the traversed path
		DecodeState scoreState = decoderStates[length].getNext();
		List<ScoredPhrasePairForSentence> output = new ArrayList<ScoredPhrasePairForSentence>();
		while(scoreState.previousState != null){
			output.add(0, scoreState.phrasedSentence);
			scoreState = scoreState.previousState;
		}
		
		return output;
	}
	
	boolean isValidState(DecodeState state){
		int lastIndex = state.phraseEndIndex;
		int leftIndex = lastIndex -1;
		while(leftIndex >= 0 && state.phraseFlag[leftIndex] == '1'){
			leftIndex--;
		}
		
		if (Math.abs(lastIndex - leftIndex - 1) > this.distortionLimit){
			return false;
		}
		
		int rightIndex = lastIndex + 1;
		while(rightIndex < state.phraseFlag.length && state.phraseFlag[rightIndex] == '1'){
			rightIndex++;
		}
		
		if (Math.abs(rightIndex - leftIndex -1) > this.distortionLimit){
			return false;
		}
		
		return true;
	}
	
	public List<DecodeState> GetNextStates(DecodeState currentState, List<ScoredPhrasePairForSentence> nextPhrases){
		List<DecodeState> nextStates = new ArrayList<DecodeState>();
		//System.out.println("STARTING PRINTING NEXT STATE FOR GIVEN==============================");
		for (ScoredPhrasePairForSentence nextPhrase : nextPhrases){
			
			double lmScore = LanguageModelUtil.GetLMScore(currentState, nextPhrase.english, this.languageModel);
			
			double distortionScore = this.distortionModel.getDistortionScore(currentState.phraseEndIndex, nextPhrase.getStart());
			
			double nextStateScore = currentState.score + lmScore + distortionScore + nextPhrase.score;
			//System.out.println(lmScore + " " + nextPhrase.score + "  " + distortionScore);
			DecodeState newState = new DecodeState(nextStateScore, nextPhrase, currentState );
			//System.out.println("OldLength = " + currentState.countOfSetFlag + "  NewLength = " + newState.countOfSetFlag);
			
			if (newState.countOfSetFlag <= currentState.countOfSetFlag){
				//System.out.println("This can not be possible. So I am going some mistake. Please fix it.");
			}
			
			if (this.distortionLimit == 0 || (isValidState(newState) && newState.score  > -200) ){
				nextStates.add(newState);
				//System.out.println(currentState.toString() + "_" + currentState.score + "  ==>" + newState.toString() + " " + newState.score + "__" + newState.countOfSetFlag);
			}else{
				//System.out.println("NVS - "+newState.toString());
			}
		}
		
		return nextStates;
	}
	
	public List<ScoredPhrasePairForSentence> GetNextPhrasePairs(DecodeState currentState, int maxDistortionLimit, PhraseTableForSentence sentencePhraseTable){
		
		int currentIndex = 0;
		int startIndex = 0;
		List<ScoredPhrasePairForSentence> nextPhrasePair = new ArrayList<ScoredPhrasePairForSentence>();
		int maximumPhraseLength = sentencePhraseTable.getMaxPhraseLength();
		boolean startIndexFound = false;
		while(currentIndex < currentState.phraseFlag.length){
			if (currentState.phraseFlag[currentIndex] == '0' & startIndexFound == false){
				startIndexFound = true;
				startIndex = currentIndex;
			}else if (startIndexFound == true & currentState.phraseFlag[currentIndex] == '1'){
				int endIndex = currentIndex;
				startIndexFound = false;
				// Get all the phrases
				for (int index = startIndex; index < endIndex; index++){
					if (Math.abs(index - currentState.phraseEndIndex - 1) <= maxDistortionLimit){
						for (int length = 1; length <= maximumPhraseLength && index + length <= endIndex; length++){
							List<ScoredPhrasePairForSentence> phrases = sentencePhraseTable.getScoreSortedTranslationsForSpan(index, index + length);
							if (phrases != null){
								nextPhrasePair.addAll(phrases);
								/*for (ScoredPhrasePairForSentence tempPhrase : phrases){
									DecodeState tempState = new DecodeState(0.0, tempPhrase, currentState);
									System.out.println(tempState.toString());
								}*/
							}else{
								//System.out.println("I want to debug why we are not getting any phrase");
							}
						}
					}
				}
			}else{
				
			}
			currentIndex++;
		}
		
		if (startIndexFound){
			for (int index = startIndex; index < currentState.phraseFlag.length; index++){
				if (Math.abs(index - currentState.phraseEndIndex - 1) <= maxDistortionLimit){
					for (int length = 1; length <= maximumPhraseLength && index + length <= currentState.phraseFlag.length; length++){
						List<ScoredPhrasePairForSentence> phrases = sentencePhraseTable.getScoreSortedTranslationsForSpan(index, index + length);
						if (phrases != null){
							nextPhrasePair.addAll(phrases);
							/*for (ScoredPhrasePairForSentence tempPhrase : phrases){
								DecodeState tempState = new DecodeState(0.0, tempPhrase, currentState);
								System.out.println(tempState.toString());
							}*/
						}else{
							//System.out.println("I want to debug why I am not getting any phrase");
						}
					}
				}else{
					break;
				}
			}
		}
		
		return nextPhrasePair;
	}
}
