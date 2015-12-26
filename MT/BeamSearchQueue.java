package edu.berkeley.nlp.assignments.decoding.student;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import javax.swing.RowFilter.Entry;
import javax.swing.text.html.HTMLDocument.Iterator;

import edu.berkeley.nlp.util.FastPriorityQueue;
/***
 * This is Fast priority queue, which prunes the elments if it is more than required
 * capacity. It internally uses fast priority queue.
 * It also has mechanism to update its data if elments key is already present in priority queue.
 * This has been achieved through using map.
 * @author rajkuma
 *
 */
  public class BeamSearchQueue{
	FastPriorityQueue<DecodeState> queue = new FastPriorityQueue<DecodeState>();
	HashMap<String, DecodeState> cache = new HashMap<String, DecodeState>();
	boolean isDistorationEnabled = false;
	public void add(DecodeState state){
		
		if (queue.size() > 5000){
			PruneOfBasisOfSize();
		}
		
		if (state.flagBitEnabled){
			isDistorationEnabled = true;
			String identifier = state.toString();
			//System.out.println(identifier + "_"+state.score);
			if (cache.containsKey(identifier)){
				DecodeState tempState = cache.get(identifier);
				if (state.score > tempState.score){
					tempState.score = Double.NEGATIVE_INFINITY;
					queue.setPriority(state, state.score);
					//System.out.print("C-");
				}
			}else{
				queue.setPriority(state, state.score);
				cache.put(identifier, state);
			}
		}else{
			queue.setPriority(state, state.score);
		}
	}
	
	public void remove(){
		if (queue.hasNext()){
			queue.next();
		}
	}
	
	public DecodeState getNext(){
		
		if (queue.hasNext()){
			return queue.next();
		}
		
		return null;
	}
	
	public void PruneOfBasisOfSize(){
		if (isDistorationEnabled){
			int allowedSize = Math.min(2000, queue.size());
			if (queue.size() > allowedSize){
				cache = new HashMap<String, DecodeState>();
				FastPriorityQueue<DecodeState> prunnedQueue = new FastPriorityQueue<DecodeState>();
				int count = 0;
				while(count < allowedSize && queue.hasNext()){
					DecodeState tempState = queue.next();
					if (tempState.score > Double.NEGATIVE_INFINITY){
						prunnedQueue.setPriority(tempState, tempState.score);
						cache.put(tempState.toString(), tempState);
						count++;
					}else{
						//System.out.print("D-");
					}
				}
				
				queue = prunnedQueue;
				//System.out.println("");
				
			}
		}else{
			int allowedSize = 2000;
			if (queue.size() > allowedSize){
				FastPriorityQueue<DecodeState> prunnedQueue = new FastPriorityQueue<DecodeState>();
				int count = 0;
				while(count < allowedSize && queue.hasNext()){
					DecodeState tempState = queue.next();
						prunnedQueue.setPriority(tempState, tempState.score);
						count++;
				}
				
				queue = prunnedQueue;
				//System.out.println("");
				
			}
		}
		
	}
}