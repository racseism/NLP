package edu.berkeley.nlp.assignments.decoding.student;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.EnglishPhrase;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;
import edu.berkeley.nlp.util.FastPriorityQueue;

public class MonotonicWithLmDecoder implements Decoder{
	
	
	
	PhraseTable phraseTable;
	
	NgramLanguageModel languageModel;
	
	DistortionModel distortionModel;
	
	public MonotonicWithLmDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm) {
		// TODO Auto-generated constructor stub
		this.phraseTable = tm;
		this.languageModel = lm;
		this.distortionModel = dm;
	}
	
	@Override
	public List<ScoredPhrasePairForSentence> decode(List<String> frenchSentence) {
		// TODO Auto-generated method stub
		
		//get phrase translation for given foreign sentence
		PhraseTableForSentence sentencePhraseTranslation = phraseTable.initialize(frenchSentence);
		
		int length = frenchSentence.size();
		
		BeamSearchQueue decoderStates[] = new BeamSearchQueue[frenchSentence.size() + 1];
		
		//Initializing the beam search queue
		for (int index = 0; index < decoderStates.length; index++){
			decoderStates[index] = new BeamSearchQueue();
		}
		
		//Qo state.
		DecodeState q0State = new DecodeState();
		decoderStates[0].add(q0State);
		
		int maxPhraseLength = sentencePhraseTranslation.getMaxPhraseLength();
		for (int wordIndex = 0; wordIndex < length; wordIndex++){
			DecodeState currentState = decoderStates[wordIndex].getNext();
			int maxLength = Math.min(maxPhraseLength, length - wordIndex);
			while(currentState != null){
				for (int phraseLegth = 1; phraseLegth <= maxLength; phraseLegth++){
					List<ScoredPhrasePairForSentence> translatedPhrases = sentencePhraseTranslation.getScoreSortedTranslationsForSpan(wordIndex, wordIndex + phraseLegth);
					if (translatedPhrases != null){
						for (ScoredPhrasePairForSentence translatedPhrase : translatedPhrases){
							double languageModelScore = LanguageModelUtil.GetLMScore(currentState, translatedPhrase.english, this.languageModel);
							double totalScore = languageModelScore + translatedPhrase.score + currentState.score;
							decoderStates[wordIndex + phraseLegth].add(new DecodeState(totalScore, translatedPhrase, currentState));
						}
					}
				}
				
				currentState = decoderStates[wordIndex].getNext();
			}
			
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
}
