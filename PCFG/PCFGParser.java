package edu.berkeley.nlp.assignments;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.BaseStream;

import edu.berkeley.nlp.assignments.PCFGParserTester.BaselineParser;
import edu.berkeley.nlp.assignments.PCFGParserTester.BinaryRule;
import edu.berkeley.nlp.assignments.PCFGParserTester.Parser;
import edu.berkeley.nlp.assignments.PCFGParserTester.TreeAnnotations;
import edu.berkeley.nlp.assignments.PCFGParserTester.UnaryClosure;
import edu.berkeley.nlp.assignments.PCFGParserTester.UnaryRule;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.Counters;

/***
 * 
 * @author Rajeev Kumar
 *
 */
public class PCFGParser extends BaselineParser{
	  
    CustomClosure customClosure;
    
    public static int MarkovHorizontalOrder = 0;
    
    public static int MarkovVerticalOrder = 0;
    
    // Top 500 tags will be only stored for each cells, others will be pruned.
    int PruningConstantTopKeysCount = 500;
    /***
     * This stores the back path of binary rule. It is used in CKY algorithm to traverse back path.
     * @author Rajeev Kumar
     *
     */
	static class BinaryBackPath{
		public int leftChildRow = -1;
		public int leftChildColumn = -1;
		public int rightChildRow = -1;
		public int rightChildColumn = -1;
		
		public BinaryRule binaryRule = null;
	}
	
	/***
	 * Stores back path of unary rule. It is used in CKY algorithm to traverse back path.
	 * @author Rajeev Kumar
	 *
	 */
	static class UnaryBackPath{
		public String endTag;
		List<UnaryRule> unaryRules = new ArrayList<PCFGParserTester.UnaryRule>();
	}
	
	/***
	 * Binary Cell data structure. It has score corresponding tags and also List of back-path,
	 * through which score has been generated corresponding each tags.
	 * @author Rajeev Kumar
	 *
	 */
	static class BinaryCkyCell{
		public Counter<String> tagScore = new Counter<String>();
		
		// Use to traverse back, to find out, which lower children has lead to maximum tag scores.
		public Map<String, BinaryBackPath> backPath = new HashMap<String, BinaryBackPath>();
	}
	
	/***
	 * 
	 * @author Rajeev Kumar
	 *
	 */
	static class UnaryCkyCell{
		public Counter<String> tagScore = new Counter<String>();
		
		public Map<String, UnaryBackPath> backPath = new HashMap<String, UnaryBackPath>();
	}
	
	
	/***
	 * Constructor.
	 * @param trainTrees
	 */
	public PCFGParser(List<Tree<String>> trainTrees) {
		super(trainTrees);
		// TODO Auto-generated constructor stub
		customClosure = new CustomClosure(grammar.getUnaryRules());
		
	}
	
	/***
	 * Calculate the score of terminal symbol using tagging probability.
	 * @param word - word of the sentence.
	 * @param tag - Tag assigned corresponding word.
	 * @return
	 */
	public double GetTerminalSymbolTagProbability(String word, String tag){
		// Change it to assignment2 tagging
		return lexicon.scoreTagging(word, tag);
	}
	
	public int GetIndex(int row, int column, int size){
		return row * size + column;
	}
	
	/***
	 * Annotate the tree according to horizontal and vertical markovization.
	 */
	@Override
    protected List<Tree<String>> annotateTrees(List<Tree<String>> trees) {
        List<Tree<String>> annotatedTrees = new ArrayList<Tree<String>>();
        for (Tree<String> tree : trees) {
          annotatedTrees.add(CustomTreeAnnotations.annotateTree(tree, this.MarkovHorizontalOrder, this.MarkovVerticalOrder));
        }
        
        return annotatedTrees;
      }
	
	/***
	 * Apply pruning on each cell. It will only keep top K tags. 
	 * @param unaryCkyCell
	 */
	public void ApplyPruning(UnaryCkyCell unaryCkyCell){

		if (unaryCkyCell.tagScore.size() > 1){
			double maxScoreUsingAllTags = unaryCkyCell.tagScore.getCount(unaryCkyCell.tagScore.argMax());
			
			// All tags whose score is less than cutOff value will be dropped.
			Counter<String> prunnedScore = new Counter<String>();
			List<String> sortedKeyList = Counters.sortedKeys(unaryCkyCell.tagScore);
			int maxCount = Math.min(sortedKeyList.size(), PruningConstantTopKeysCount);
			for (int index = 0; index < maxCount; index++){
				prunnedScore.setCount(sortedKeyList.get(index), unaryCkyCell.tagScore.getCount(sortedKeyList.get(index)));
			}
			
			unaryCkyCell.tagScore = prunnedScore;
		}
		
		//System.out.println("After pruning size is " + unaryCkyCell.tagScore.size());

	}
	
	/***
	 * Will apply unary closure on each tags and update the score of each tags using unary rule.
	 * @param unaryCkyCell
	 * @param binaryCkyCell
	 */
	public void ApplyUnaryRule(UnaryCkyCell unaryCkyCell, BinaryCkyCell binaryCkyCell){
		//System.out.println("Enter Unary Tag count = " + unaryCkyCell.tagScore.size());
		for (String tag : binaryCkyCell.tagScore.keySet()){
			Counter<String> unaryClosureParentsCost = customClosure.GetClosueCostFromSouceGivenChild(tag);
			if (unaryClosureParentsCost != null){
				//System.out.println("total parent count = " + unaryClosureParentsCost.size());
				for (String unaryClosureParent : unaryClosureParentsCost.keySet()){
					UnaryRule combinedRule = new UnaryRule(unaryClosureParent, tag);
					double pathCost = unaryClosureParentsCost.getCount(unaryClosureParent);
					if (pathCost > 0){
						double totalCostIncludingTransition = pathCost * binaryCkyCell.tagScore.getCount(tag);
						List<UnaryRule> optmizedPathRules = customClosure.getClosurePath(combinedRule);
						if (unaryCkyCell.tagScore.containsKey(unaryClosureParent)){
							double oldScore = unaryCkyCell.tagScore.getCount(unaryClosureParent);
							if (oldScore < totalCostIncludingTransition){
								unaryCkyCell.tagScore.setCount(unaryClosureParent, totalCostIncludingTransition);
								UnaryBackPath unaryBackPath = new UnaryBackPath();
								unaryBackPath.unaryRules = optmizedPathRules;
								unaryBackPath.endTag = optmizedPathRules.get(optmizedPathRules.size() - 1).child;
								unaryCkyCell.backPath.put(unaryClosureParent, unaryBackPath);
							}
						}else{
							unaryCkyCell.tagScore.setCount(unaryClosureParent, totalCostIncludingTransition);
							UnaryBackPath unaryBackPath = new UnaryBackPath();
							unaryBackPath.unaryRules = optmizedPathRules;
							unaryBackPath.endTag = optmizedPathRules.get(optmizedPathRules.size() - 1).child;
							unaryCkyCell.backPath.put(unaryClosureParent, unaryBackPath);
						}
					}
				}
			}
		}
				
		//System.out.println("Exit Unary Tag count = " + unaryCkyCell.tagScore.size());
	}
	
	/***
	 * Initialize all the CKY Cell with its DATA structure.
	 * @param binaryCky
	 * @param unaryCky
	 * @param sentence
	 */
	public void InitializeCKYTable(List<BinaryCkyCell> binaryCky, List<UnaryCkyCell> unaryCky, List<String> sentence){
		for (int row = 0;  row < sentence.size(); row++){
			for (int col = 0; col < sentence.size(); col++){
				binaryCky.add(new BinaryCkyCell());
				unaryCky.add(new UnaryCkyCell());
			}
		}
	}
	
	/***
	 * Apply CKY on terminal node. Here mainly tagging probability and unary closure.
	 * @param binaryCky - Cell with each tag Generated through Binary Rule.
	 * @param unaryCky - Cell with teach tag Generated through Unary Rule.
	 * @param sentence
	 */
	public void ApplyCKYOnTerminalNode(List<BinaryCkyCell> binaryCky, List<UnaryCkyCell> unaryCky, List<String> sentence){
		
		for (int wordIndex = 0; wordIndex < sentence.size(); wordIndex++){
			int rowIndex = wordIndex;
			int columnIndex = wordIndex;
			//System.out.println("Starting Terminal Node Processing for Row = " + rowIndex + "  Column = " + columnIndex);
			
			BinaryCkyCell binaryCkyCell = binaryCky.get(GetIndex(rowIndex, columnIndex, sentence.size()));
			UnaryCkyCell unaryCkyCell = unaryCky.get(GetIndex(wordIndex, wordIndex, sentence.size()));
			
			for (String tag : lexicon.getAllTags()){
				UnaryRule unaryRule = new UnaryRule(tag, tag);
				double score = GetTerminalSymbolTagProbability(sentence.get(wordIndex), tag);
				
				binaryCkyCell.tagScore.setCount(unaryRule.parent, score);
				binaryCkyCell.backPath.put(unaryRule.parent, null);
				
				unaryCkyCell.tagScore.setCount(unaryRule.parent, score);
				UnaryBackPath unaryBackPath  = new UnaryBackPath();
				unaryBackPath.endTag = unaryRule.parent;
				unaryBackPath.unaryRules.add(unaryRule);
				unaryCkyCell.backPath.put(unaryRule.parent, unaryBackPath);
				
			}
			
			// Handle Unary rules.
			ApplyUnaryRule(unaryCkyCell, binaryCkyCell);
			
		}
	}
	
	/***
	 * Apply CKY algorithm on Non-terminal Node. On each cell, we apply Binary Rule. And for each tags, It finds the best score,
	 * using Binary Rule. Once new tag score is generated for all tags of cell using Binary Rule, Unary Rule is applied on all tags to 
	 * get updated score.
	 * @param binaryCky
	 * @param unaryCky
	 * @param sentence
	 */
	public void ApplyCkyOnNonTerminalNode(List<BinaryCkyCell> binaryCky, List<UnaryCkyCell> unaryCky, List<String> sentence){
		// Actual span length span + 1. But for simplicity of index access. I started form span = 1
		for (int span = 1; span < sentence.size(); span++){
			for (int wordIndex = 0; wordIndex < sentence.size() - span ; wordIndex++){
				int rowIndex = wordIndex;
				int columnIndex = rowIndex + span;
				//System.out.println("");
				//System.out.println("( " + rowIndex + ", " + columnIndex + ")=>");
				BinaryCkyCell row_Column_BinaryCell = binaryCky.get(GetIndex(rowIndex,  columnIndex, sentence.size()));
				UnaryCkyCell row_Column_UnaryCell= unaryCky.get(GetIndex(rowIndex, columnIndex, sentence.size()));
				
				for (int leftSpanEndIndex = rowIndex; leftSpanEndIndex < columnIndex; leftSpanEndIndex++){
					int leftChildRowIndex = rowIndex;
					int leftChildColIndex = leftSpanEndIndex;
					int rightChildRowIndex = leftSpanEndIndex + 1;
					int rightChildColIndex = columnIndex;
					//System.out.print(String.format("[%s,%s]^[%s,%s],", leftChildRowIndex, leftChildColIndex, rightChildRowIndex, rightChildColIndex));
					for (BinaryRule binaryRule : grammar.getBinaryRules()){
						UnaryCkyCell leftSpanUnaryCell = unaryCky.get(GetIndex(leftChildRowIndex, leftChildColIndex, sentence.size()));
						UnaryCkyCell rightSpanUnaryCell = unaryCky.get(GetIndex(rightChildRowIndex, rightChildColIndex, sentence.size()));
						//System.out.println(leftSpanUnaryCell.tagScore.size() + "  " + rightSpanUnaryCell.tagScore.size() + "  " + grammar.getBinaryRules().size());
						double leftSpanUnaryCellScore = leftSpanUnaryCell.tagScore.getCount(binaryRule.leftChild);
						double rightSpanUnaryCellScore = rightSpanUnaryCell.tagScore.getCount(binaryRule.rightChild);
						if (leftSpanUnaryCellScore > 0 && rightSpanUnaryCellScore > 0){
							double score = leftSpanUnaryCellScore * rightSpanUnaryCellScore * binaryRule.getScore();
							
							if (row_Column_BinaryCell.tagScore.containsKey(binaryRule.parent)){
								double oldScore = row_Column_BinaryCell.tagScore.getCount(binaryRule.parent);
								
								if (oldScore < score){
									// Update the reverse back-pointer.
									row_Column_BinaryCell.tagScore.setCount(binaryRule.parent, score);
									BinaryBackPath binaryBackPath = new BinaryBackPath();
									binaryBackPath.binaryRule = binaryRule;
									
									binaryBackPath.leftChildRow = leftChildRowIndex;
									binaryBackPath.leftChildColumn = leftChildColIndex;
									binaryBackPath.rightChildRow = rightChildRowIndex;
									binaryBackPath.rightChildColumn = rightChildColIndex;
									
									row_Column_BinaryCell.backPath.put(binaryRule.parent, binaryBackPath);
									
									row_Column_UnaryCell.tagScore.setCount(binaryRule.parent, score);
									UnaryBackPath unaryBackPath  = new UnaryBackPath();
									unaryBackPath.endTag = binaryRule.parent;
									unaryBackPath.unaryRules.add(new UnaryRule(binaryRule.parent, binaryRule.parent));
									row_Column_UnaryCell.backPath.put(binaryRule.parent, unaryBackPath);
								}
							}else if (score > 0){
								row_Column_BinaryCell.tagScore.setCount(binaryRule.parent, score);
								
								BinaryBackPath binaryBackPath = new BinaryBackPath();
								binaryBackPath.binaryRule = binaryRule;
								
								binaryBackPath.leftChildRow = leftChildRowIndex;
								binaryBackPath.leftChildColumn = leftChildColIndex;
								binaryBackPath.rightChildRow = rightChildRowIndex;
								binaryBackPath.rightChildColumn = rightChildColIndex;
								
								row_Column_BinaryCell.backPath.put(binaryRule.parent, binaryBackPath);
								
								row_Column_UnaryCell.tagScore.setCount(binaryRule.parent, score);
								UnaryBackPath unaryBackPath  = new UnaryBackPath();
								unaryBackPath.endTag = binaryRule.parent;
								unaryBackPath.unaryRules.add(new UnaryRule(binaryRule.parent, binaryRule.parent));
								row_Column_UnaryCell.backPath.put(binaryRule.parent, unaryBackPath);
							}
						}
					} // End of left span looping
				} // End of Binary Rule looping
				
				// Handling for Unary rules
				
				ApplyUnaryRule(row_Column_UnaryCell, row_Column_BinaryCell);
				
				//Apply pruning
				//ApplyPruning(row_Column_UnaryCell);
				
			} // End of WordIndex looping
		} // End span looping
	}
	
	/***
	 * Generate Tree node using unary rules. 
	 * @param startTag
	 * @param unaryCkyCell
	 * @return
	 */
	public List<Tree<String>> GetTreeWitUnaryRule(String startTag, UnaryCkyCell unaryCkyCell){
		if (unaryCkyCell.backPath.containsKey(startTag)){
			List<UnaryRule> backPathRules = unaryCkyCell.backPath.get(startTag).unaryRules;
			List<Tree<String>> outputNodes = new ArrayList<Tree<String>>();
			Tree<String> parentNode = new Tree<String>(startTag);
			outputNodes.add(parentNode);
			for (UnaryRule rule : backPathRules){
				Tree<String> childNode = new Tree<String>(rule.child);
				outputNodes.add(childNode);
				List<Tree<String>> children = new ArrayList<Tree<String>>();
				children.add(childNode);
				parentNode.setChildren(children);
				parentNode = childNode;
			}
			
			return outputNodes;
		}
		
		return null;
	}
	
	/***
	 * This will remove self unary rule. If X->X and X parent has only one children, then
	 * one X will be removed because it has been added as self-unary rule to make CKY algorithm easy.
	 * @param tree - Input tree
	 * @return - Output tree after removal of self unary rule.
	 */
	public Tree<String> RemoveSelfUniaryRule(Tree<String> tree){
		if (tree.isPreTerminal()){
			return tree;
		}
		
		List<Tree<String>> children = tree.getChildren();
		if (children.size() == 1 && (children.get(0).getLabel().equals(tree.getLabel()))){
			// This is case of X->X, which is extra relation added to handle unary. Hence need to be removed.
			Tree<String> node = RemoveSelfUniaryRule(children.get(0));
			return node;
		}
		else{
			for (int index = 0; index < children.size(); index++){
				Tree<String> node = RemoveSelfUniaryRule(children.get(index));
				children.set(index, node);
			}
		}
		
		return tree;
		
	}
	
	/***
	 * Get optimized tree using CKY path traversal algorithm.
	 * @param sentence - Original sentence,for which we have to find optimized PCFG tree.
	 * @param rowIndex - Start Index of CKY dynamic table. It would start with 0.
	 * @param colIndex - Column Index of CKY dynamic table. It would start with n.
	 * @param startTag - Tag to be chosen of given column and row. Initially, it starts with "ROOT"
	 * @param binaryCky - binaryCky dynamic table generated by CKY algorithm.
	 * @param unaryCky - unaryCky dynamic table generated from CKY algorithm.
	 * @return
	 */
 	public Tree<String> GetOptmizedTree(List<String> sentence,
			int rowIndex,
			int colIndex,
			String startTag,
			List<BinaryCkyCell> binaryCky,
			List<UnaryCkyCell> unaryCky){
		
		
		UnaryCkyCell unaryCkyCell = unaryCky.get(GetIndex(rowIndex, colIndex, sentence.size()));
		
		BinaryCkyCell binaryCkyCell = binaryCky.get(GetIndex(rowIndex, colIndex, sentence.size()));
		
		List<Tree<String>> skewTreeNodesWithUnaryRule = GetTreeWitUnaryRule(startTag, unaryCkyCell);
		
		Tree<String> startNodeDueToUnary = skewTreeNodesWithUnaryRule.get(0);
		Tree<String> endNodeDueToUnary = skewTreeNodesWithUnaryRule.get(skewTreeNodesWithUnaryRule.size() - 1);
		
		List<Tree<String>> binaryTreeChildren = new ArrayList<Tree<String>>();
		endNodeDueToUnary.setChildren(binaryTreeChildren);

		BinaryBackPath binaryBackPath = binaryCkyCell.backPath.get(endNodeDueToUnary.getLabel());
		
		if (binaryBackPath == null)
		{
			// It means it is terminal node.
			// Here we get terminal Node tag = sentence[rowIndex] or sentence[colIndex]
			Tree<String> leftTreeDueToBinary = new Tree<String>(sentence.get(rowIndex));
			binaryTreeChildren.add(leftTreeDueToBinary);
			
		}else{
			Tree<String> leftTreeDueToBinary = GetOptmizedTree(sentence,
					binaryBackPath.leftChildRow,
					binaryBackPath.leftChildColumn,
					binaryBackPath.binaryRule.leftChild,
					binaryCky,
					unaryCky);
			
			Tree<String> rightTreeDueToBinary = GetOptmizedTree(sentence,
					binaryBackPath.rightChildRow,
					binaryBackPath.rightChildColumn,
					binaryBackPath.binaryRule.rightChild,
					binaryCky,
					unaryCky);
			
			binaryTreeChildren.add(leftTreeDueToBinary);
			binaryTreeChildren.add(rightTreeDueToBinary);
		}
		
		return startNodeDueToUnary;
	}
	
 	/***
 	 * Get best parse method using CKY algorithm. It has following steps.
 	 * 1. Apply CKY algorithm on terminal node.
 	 * 2. Apply CKY algorithm on non-terminal node.
 	 * 3. Get optimized tree by back-path traversal.
 	 * 4. Remove self unary rule added while CKY algorithm path generation.
 	 * 5. Un-annotate the tree.
 	 */
 	public  Tree<String> getBestParse(List<String> sentence){
		//sentence = sentence.subList(0, 4);
		//System.out.println("Starting CKY parser for sentence " + PCFGUtils.Join(sentence, ","));
		
		long startTime = System.currentTimeMillis();
		// Generate for test
		
		List<BinaryCkyCell> binaryCky = new ArrayList<BinaryCkyCell>();
		
		List<UnaryCkyCell> unaryCky = new ArrayList<UnaryCkyCell>();
		
		InitializeCKYTable(binaryCky, unaryCky, sentence);
		
		// CKY on terminal node.
		ApplyCKYOnTerminalNode(binaryCky, unaryCky, sentence);
		
		// CKY on non-terminal node.
		ApplyCkyOnNonTerminalNode(binaryCky, unaryCky, sentence);
		
		Tree<String> pcfgTree = GetOptmizedTree(sentence, 0, sentence.size() -1, "ROOT", binaryCky, unaryCky);
		
		//System.out.println("Tree " + pcfgTree.toString());
		pcfgTree = RemoveSelfUniaryRule(pcfgTree);
		//System.out.println("Tree " + pcfgTree.toString());
		
		pcfgTree = TreeAnnotations.unAnnotateTree(pcfgTree);
		//System.out.println("Un-annonated Tree" + pcfgTree.toString());
		
		long elapseTimeInMilliSeconds = System.currentTimeMillis() - startTime;
		
		System.out.println("ExeuctionTime of CKY parser " + elapseTimeInMilliSeconds + "  milliseconds");
		return pcfgTree;
	}
	  
  }