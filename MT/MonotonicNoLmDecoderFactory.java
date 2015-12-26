package edu.berkeley.nlp.assignments.decoding.student;

import java.util.List;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DecoderFactory;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;

public class MonotonicNoLmDecoderFactory implements DecoderFactory
{
	
	public Decoder newDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm) {

		 return new MonotonicNoLmDecoder(tm, lm, dm);

	}

}