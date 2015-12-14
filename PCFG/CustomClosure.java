package edu.berkeley.nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.assignments.PCFGParserTester.UnaryClosure;
import edu.berkeley.nlp.assignments.PCFGParserTester.UnaryRule;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;


/**
 * @author Rajeev Kumar
 * This is custom closure, added functionality of our given closure. Following are the issues, we faced on given closure.
 * In our CKY algorithm, we multiple time has to calculate the cost of X->Y->Z etc. This closure is generated from grammar,
 * which does not change on run time. 
 * Added functionality - Mainly for optimization.
 * 	1. Calculated cost of All unary closure path cost for given start and end tags.
 * 	2. Calculated All Unary Closure path.
 */
public class CustomClosure extends UnaryClosure{
	
	// Stores map of UnaryRule -> Closure of UnaryRule. For example (X->Y) -> {(X->L1), }
	Map<UnaryRule, List<UnaryRule>> closurePathMap = new HashMap<PCFGParserTester.UnaryRule, List<UnaryRule>>();
  
	// Stores the score for Pair (UnaryRule).
    CounterMap<String, String> destinatioToSouceClosureCost = new CounterMap<String, String>();
    
    /***
     * Calculates the closure of all unary rules.
     * @param unaryRules
     */
	public CustomClosure(Collection<UnaryRule> unaryRules) {
		super(unaryRules);
		// TODO Auto-generated constructor stub
		this.unaryRules.addAll(unaryRules);
		Map<UnaryRule, List<String>> closureMap = computeUnaryClosure(unaryRules);
		for (UnaryRule unaryRule : closureMap.keySet()) {
			addUnary(unaryRule, closureMap.get(unaryRule));
		}
	}
	
	/***
	 * Get Closure of given unary rule.
	 * @param unaryRule - Unary Rule
	 * @return - Closure of unary rule.
	 */
	public List<UnaryRule> getClosurePath(UnaryRule unaryRule){
    	return closurePathMap.get(unaryRule);
    }
    
	/***
	 * Returns Counts of all the closed rules with given destination node.
	 * @param destinationChild - Destination tag
	 * @return
	 */
    public Counter<String> GetClosueCostFromSouceGivenChild(String destinationChild){
    	return destinatioToSouceClosureCost.getCounter(destinationChild);
    }
    
    /***
     * Generates List with following property with new addition of Unary rule
     * 1. UnaryRule -> Unary rule closed with its parent
     * 2. UnaryRule -> Unary rule closed with its child.
     * 3. UnaryRule -> score of closed unary rule for given unary rule.
     * @param unaryRule - Unary Rule
     * @param path - Closed path for given unary rule.
     */
    private void addUnary(UnaryRule unaryRule, List<String> path) {
        CollectionUtils.addToValueList(closedUnaryRulesByChild, unaryRule.getChild(), unaryRule);
        CollectionUtils.addToValueList(closedUnaryRulesByParent, unaryRule.getParent(), unaryRule);
        pathMap.put(unaryRule, path);
             
        List<UnaryRule> unaryPaths = GenerateRuleFromPath(path);
    	  double score = GetScoreForPathForUnaryRules(unaryPaths);
    	  closurePathMap.put(unaryRule, unaryPaths);
    	destinatioToSouceClosureCost.setCount(unaryRule.child, unaryRule.parent, score);
    }
    

    /***
     * Generates unary rule form unary path
     * @param path - List of Unary path
     * @return - List of Unary Rule
     */
    private List<UnaryRule> GenerateRuleFromPath(List<String> path){
		if (path == null){
			return null;
		}
		
		List<UnaryRule> pathRules = new ArrayList<UnaryRule>();
		
		if (path.size() == 1){
			pathRules.add(new UnaryRule(path.get(0), path.get(0)));
			return pathRules;
		}
				
		for (int index = 1; index < path.size(); index++){
			pathRules.add(new UnaryRule(path.get(index -1), path.get(index)));
		}
		
		return pathRules;
	}
	
    /***
     * Calculate the score of Unary paths. This is calculating aggregate score of paths.
     * @param path - List of Unary Rules in its closure.
     * @return - score of aggregate path.
     */
	private double GetScoreForPathForUnaryRules(List<UnaryRule> path){
		
		if (path == null || path.size() < 1){
			System.out.println("This shoudl not be possible. There is mistake in code");
			return Double.NEGATIVE_INFINITY;
		}
		
		double aggregateScore = 1.0;
		
		
		for (int index = 0; index < path.size(); index++){
			UnaryRule unaryRule = path.get(index);
			if (unaryRule.child.equals(unaryRule.parent)){
				return 1.0;
			}
			
			// Return index can not be negative as path has been generated from unaryRules itself. 
			// There would not be any index check in below. If it throws invalid access exception, it means
			// there is some mistake in logic.
			int indexOrUnaryRule = unaryRules.indexOf(unaryRule);
			
			unaryRule = unaryRules.get(indexOrUnaryRule);
			
			aggregateScore = aggregateScore * unaryRule.getScore();
		}
		
		return aggregateScore;
	}
    
}
