package edu.berkeley.nlp.assignments.decoding.student;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DecoderFactory;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;

public class DistortingWithLmDecoderFactory implements DecoderFactory
{

	public Decoder newDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm) {

		 return new DistortingWithLmDecoder(tm, lm, dm);
		 
	}

}
