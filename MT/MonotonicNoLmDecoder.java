package edu.berkeley.nlp.assignments.decoding.student;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrase;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;

public class MonotonicNoLmDecoder implements Decoder
{
	PhraseTable phraseTable;
	
	NgramLanguageModel languageModel;
	
	DistortionModel distortionModel;
	
	public MonotonicNoLmDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm) {
		// TODO Auto-generated constructor stub
		this.phraseTable = tm;
		this.languageModel = lm;
		this.distortionModel = dm;
	}
	
	@Override
	public List<ScoredPhrasePairForSentence> decode(List<String> frenchSentence) {
		// TODO Auto-generated method stub
		PhraseTableForSentence sentencePhraseTable = phraseTable.initialize(frenchSentence);
		
		int length = frenchSentence.size();
		
		double cost[] = new double[length];
		
		for (int i = 0; i < length; i++)
		{
			cost[i] = Double.NEGATIVE_INFINITY;
		}
		
		// Follow Dynamic programming to solve this problem.
		// cost[i] => maximum cost to generate the sentence up to length i.
		// cost[i] = max (for j=0 to i, cost(j) + phraseCost(j,i))
		
		ScoredPhrasePairForSentence paths [] = new ScoredPhrasePairForSentence[length];
		
		int maxPhraseLength = sentencePhraseTable.getMaxPhraseLength();
		
		for (int i = 0; i < length; i++) {
			
			for (int j = i; j > i - maxPhraseLength &  j >= 0 ; j--) {
				List<ScoredPhrasePairForSentence> phraseTranslation = sentencePhraseTable.getScoreSortedTranslationsForSpan(j, i + 1);
				
				double previosPhraseSentenceCost = 0;
				
				if (i > 0) {
					previosPhraseSentenceCost = cost[i-1];
				}
				
				if(phraseTranslation != null && phraseTranslation.size() > 0) {
					
					double currentCost = previosPhraseSentenceCost + phraseTranslation.get(0).score;
					//System.out.println("Phrase translation cost " + phraseTranslation.get(index).score);
					if (currentCost > cost[i]) {
						cost[i] = currentCost;
						paths[i] = phraseTranslation.get(0);
					}
				}
			}
		}
		
		// Now traverse back to find the paths which forms english sentence with highest score.
		
		int backStartIndex = length - 1;
		List<ScoredPhrasePairForSentence> output = new ArrayList<ScoredPhrasePairForSentence>();
		
		while(backStartIndex >= 0) {
			output.add(0, paths[backStartIndex]);
			backStartIndex = backStartIndex - paths[backStartIndex].getForeignLength();
		}
		
		
		//Print Sentence score
		double totalScore = 0;
		
		for (ScoredPhrasePairForSentence sentencePhrase : output){
			totalScore = totalScore + sentencePhrase.score;
			//System.out.print(sentencePhrase.score + " ");
		}
		
		
		//System.out.println("");
		//System.out.println("Total Score " + totalScore);
		return output;
	}
	
}