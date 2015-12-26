package edu.berkeley.nlp.assignments.decoding.student;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.phrasetable.EnglishPhrase;

public class LanguageModelUtil {
	
	// Assuming trigram language model.
	public static double GetLMScore(int previousPreviousTagIndex, int previousTagIndex, int [] newPhraseWordIndexs, NgramLanguageModel languageModel){
		int [] trigramWords = {previousPreviousTagIndex, previousTagIndex, 0};
		double score = 0.0;
		for (int wordIndex : newPhraseWordIndexs){
			trigramWords[2] = wordIndex;
			score = score + languageModel.getNgramLogProbability(trigramWords, 0, 0 + languageModel.getOrder());
			trigramWords[0] = trigramWords[1];
			trigramWords[1] = trigramWords[2];
		}
		
		return score;
	}
		
	public static double GetLMScore(DecodeState previousState, EnglishPhrase nextPhrase, NgramLanguageModel languageModel){		
		if (previousState == null || previousState.e2WordIndex == -1){
			return GetLMScore(DecodeState.startSymbolIndex, DecodeState.startSymbolIndex, nextPhrase.indexedEnglish, languageModel);
		}else if (previousState.e1WordIndex == -1){ // only one english word has been generated from foreign language uptil.
			return GetLMScore(DecodeState.startSymbolIndex, previousState.e2WordIndex, nextPhrase.indexedEnglish, languageModel);
		}
		
		return GetLMScore(previousState.e1WordIndex, previousState.e2WordIndex, nextPhrase.indexedEnglish, languageModel);
	}

}
