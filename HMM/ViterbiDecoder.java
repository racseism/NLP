package edu.berkeley.nlp.assignments;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import edu.berkeley.nlp.assignments.POSTaggerTester.State;
import edu.berkeley.nlp.assignments.POSTaggerTester.Trellis;
import edu.berkeley.nlp.assignments.POSTaggerTester.TrellisDecoder;
import edu.berkeley.nlp.util.Counter;
// This is viterbi decoder, provides the optimal tag for given sentence.
// It does not calculate the score. But It producues the best Tag should be assigned to each word so that 
// Sum of tag score with each word should be maximum.
//
class ViterbiDecoder implements TrellisDecoder<State>  {

	 private List<State> GetListOfBackWardTransition(Map<State,State> pathTravesed, State startState, State endState){
		  List<State> reversedPath = new ArrayList<State>();
		  if (!pathTravesed.containsKey(endState)){
			  System.out.println("Map does not had endState, Hence could not traverse back");
			  return null;
		  }
		  
		  State currentState = endState;
		  while(currentState != startState){
			  reversedPath.add(currentState);
			  currentState = pathTravesed.get(currentState);
		  }
		  
		  reversedPath.add(startState);
		  
		  // Now reversing the back-ward path to make it forward path
		  Collections.reverse(reversedPath);
		  return reversedPath;
	  }
	  
	 public List<State> getBestPath(Trellis<State> trellis){
		  //1. Needs lists of tag with previous to previous state.
		  //2. Maximum score calculated up till up to previous to previous score.
		  //3. We need for current words - List of score we have generated and also list of previous and previousPrevious tag used.
		  // 4. Another strategy is traverse list of words to words = [k]^3
		  // My Aim is F(u,v, i) = max from all w {F(w,u, i-1)}
		  // boundary condition is F(ST, ST, 0) = 0
		  // Also it is noted that at Ith level we have all states using current Tag and previous Tag.
		  Counter<State> dpTable = new Counter<State>();
		  
		  Set<State> traversedStates = new HashSet<State>();
		  
		  Map<State, State> pathBackTraversingMap = new HashMap<State, State>();
		  
		  Queue<State> queue = new LinkedList<State>();

		  State currentState = trellis.getStartState();
		  // current state is currently START STATE
		  dpTable.setCount(currentState, 0.0);
		  queue.add(currentState);
		  // NULL to distinguish between level. here distinguish between words index
		  queue.add(null);
		  while(!queue.isEmpty()){
			  currentState = queue.remove();
			  Counter<State> forwardTransitions = trellis.getForwardTransitions(currentState);
			  for (State forwardState : forwardTransitions.keySet()){
				  // Here forward state mention (u,v) and currentState = (w,u)
				  // f(w,u,i-1) + transition cost
				  double costfrom_w_u_v = dpTable.getCount(currentState) + forwardTransitions.getCount(forwardState);
				  
				  //Now this u,v can come from multiple branches. So we have to keep updating the maximum value n f(u,v,i)
				  if (dpTable.containsKey(forwardState)){
					  // F(u,v,i) cost has been already been calculated form some different branch. which may not be optimal. So will update if we find some optimal value.
					  double currentOptimalValue = dpTable.getCount(forwardState);
					  if (currentOptimalValue < costfrom_w_u_v){
						  // update with new optimal value
						  dpTable.setCount(forwardState, costfrom_w_u_v);
						  pathBackTraversingMap.put(forwardState, currentState);
					  }
				  }else{
					  // First time F(u,v,i) has been calculated
					  dpTable.setCount(forwardState, costfrom_w_u_v);
					  pathBackTraversingMap.put(forwardState, currentState);
				  }
				  
				  // Add the child element in queue
				  // To avoid same state is not inserted multiple. This is optimization. This does not it wil bring infinite loop.
				  // It is simply means, we might have to process same node multiple times.
				  if (!traversedStates.contains(forwardState)){
					  traversedStates.add(forwardState);
					  queue.add(forwardState);
				  }
				  
			  }
		  }
		  
		  // Here currentState must be end state.
		  if (currentState == trellis.getEndState()){
			  //System.out.println("Yup we are on right track");
		  }else{
			  System.out.println("We have some issue in our code. so lets fix it");
			  // Break the further operations
		  }
		  
		  return GetListOfBackWardTransition(pathBackTraversingMap, trellis.getStartState(), currentState);
	  }

}
