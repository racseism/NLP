package edu.berkeley.nlp.assignments.decoding.student;

import edu.berkeley.nlp.mt.decoder.DistortionModel;

/***
 * Distortation model to caculate the distortion score.
 * @author rajkuma
 *
 */
public class CustomDistortionModel implements DistortionModel {
	  
	  int maxDistorationAllowed;
	  double weightFactor;

	  public CustomDistortionModel(int maxDistortionAllowed, double weightFactor) {
	    this.maxDistorationAllowed = maxDistortionAllowed;
	    this.weightFactor = weightFactor;
	  }

	  public int getDistortionLimit() {
	    return this.maxDistorationAllowed;
	  }

	  public double getDistortionScore(int endIndexOfLastPhrase, int startIndexOfNewPhrase) {
		
		if (this.maxDistorationAllowed == 0){
			return 0.0;
		}
		
		if (endIndexOfLastPhrase == -1){
			return startIndexOfNewPhrase * weightFactor;
		}
		
	    int distoration = Math.abs(startIndexOfNewPhrase - endIndexOfLastPhrase - 1);
	    
	    if (distoration > this.maxDistorationAllowed) {
	      return Double.NEGATIVE_INFINITY;
	    }
	    
	    return (distoration + 1) * weightFactor;
	  }
}
