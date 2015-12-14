package edu.berkeley.nlp.assignments;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.assignments.PCFGParserTester.TreeAnnotations;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;
import edu.berkeley.nlp.util.Filter;


	/**
	 * @author Rajeev Kumar
	 *	This class helps to handle the vertical and horizontal markov process.
	 *	In Horizontal markovization - Tree node will be labeled with combination
	 *  of its nearest siblings and parent. Number of siblings consideration depends on 
	 *  order of horizontal markovization.
	 *  In Verical markovization - Tree node will be labeled  with combinations of nearest ancestor.
	 *  By default, only node label will be considered. which is 1st order vertical markovization.
	 *  In second order verical markovization, Parents and root level will be considered for generating
	 *  new label of root. 
	 */
	public class CustomTreeAnnotations extends TreeAnnotations{
		public CustomTreeAnnotations(){
	
		}
	
		/**
		 * Annotating the tree by without considering vertical and horizontal markovization.
		 * @param unAnnotatedTree
		 * @return - Annotated Tree, have maximum two children for node.
		 */
		public static Tree<String> annotateTree(Tree<String> unAnnotatedTree) {
			return binarizeTree(unAnnotatedTree, 0);
		}

		/**
		 * Binarizing the tree by considering Horizontal Markov process.
		 * @param tree - Tree to be converted with maximum two children from n-children tree.
		 * @param markovHorizontalOrder - Order of horizontal markov process.
		 * @return - Annotated tree, have maximum two children and annotated by considering horizontal markov process.
		 */
		protected static Tree<String> binarizeTree(Tree<String> tree, int markovHorizontalOrder) {
			String label = tree.getLabel();
			if (tree.isLeaf())
				return new Tree<String>(label);
			if (tree.getChildren().size() == 1) {
				return new Tree<String>(label, Collections.singletonList(binarizeTree(tree.getChildren().get(0), markovHorizontalOrder)));
			}
			// otherwise, it's a binary-or-more local tree, so decompose it into a sequence of binary and unary trees.
			String intermediateLabel = "@" + label + "->";
			Tree<String> intermediateTree = binarizeTreeHelper(tree, 0, intermediateLabel, markovHorizontalOrder,  new ArrayList<String>());
			return new Tree<String>(label, intermediateTree.getChildren());
		}
	
		/***
		 * Binarizing the tree by considering Horizontal and Vertical Markov process.
		 * @param unAnnotatedTree - Tree to be converted with maximum two children from n-children tree.
		 * @param markovHorizontalOrder - order of horizontal markov process.
		 * @param markovVerticalOrder - Order of vertical markov process.
		 * @return
		 */
		public static Tree<String> annotateTree(Tree<String> unAnnotatedTree, int markovHorizontalOrder, int markovVerticalOrder){
	
			if (markovVerticalOrder > 1){
				// Vertical Markovization is enabled. Default vertical markovization is 1.
				AnnotateTreeWithVerticalMarkovization(unAnnotatedTree, new ArrayList<String>(), markovVerticalOrder-1);
			}
	
			return binarizeTree(unAnnotatedTree, markovHorizontalOrder);
		}
		
		/***
		 * Annotatign the tree by considering vertical markov process.
		 * @param unAnnotatedTree - Tree to be annotated with vertical markov process.
		 * @param ancestors - Label of ancestors node from root.
		 * @param markovVerticalOrder - Order of vertical Markov process.
		 */
		public static void AnnotateTreeWithVerticalMarkovization(Tree<String> unAnnotatedTree, List<String> ancestors, int markovVerticalOrder){
			if (unAnnotatedTree.isLeaf()|| unAnnotatedTree.isPreTerminal()){
				return ;
			}
	
			String newLabel = unAnnotatedTree.getLabel();
			int maxLength = Math.min(markovVerticalOrder, ancestors.size());
			int startIndex = ancestors.size() - maxLength;
	
			// Adding the Ancestor Tag in bottom to top order of ancestors. 
			for (int index = ancestors.size() -1; index >= startIndex; index--){
				newLabel += "^"+ ancestors.get(index);
			}
	
			ancestors.add(unAnnotatedTree.getLabel());
			unAnnotatedTree.setLabel(newLabel);
			for (Tree<String> node : unAnnotatedTree.getChildren()){
				AnnotateTreeWithVerticalMarkovization(node, ancestors, markovVerticalOrder);
			}
			ancestors.remove(ancestors.size()-1);
		}
	
		/***
		 * Generates the Label of node by considering horizontal markov process.
		 * @param tree - parent tree.
		 * @param markovHorizontalOrder - Horizontal markov order
		 * @param siblingLevels - List of siblings in its left in tree.
		 * @return
		 */
		public static String AnnotateWithHorizontalMarkovization(Tree<String> tree, int markovHorizontalOrder, List<String> siblingLevels){
			if (markovHorizontalOrder > 0){
				if (siblingLevels == null || siblingLevels.size() < 1){
					System.out.println("There is some mistake in code. Fix it");
				}
	
				String nodeLevel = "@" + tree.getLabel() + "->";
				int leftSiblingToLook = Math.min(markovHorizontalOrder, siblingLevels.size());
				int startIndex = siblingLevels.size() - leftSiblingToLook;
				for (int index = startIndex; index < siblingLevels.size(); index++){
					nodeLevel += "_" + siblingLevels.get(index);
				}
	
				return nodeLevel;
			}
	
			return null;
		}
	
		/***
		 * Binarizing the tree considering the horizontal markov process.
		 * @param tree - Tree to be annotated.
		 * @param numChildrenGenerated - number of children already processed on its left.
		 * @param intermediateLabel - current Node label assigned to node if it does not considers horizontal markov process.
		 * @param markovHorizontalOrder - Horizontal order of markov process.
		 * @param siblingLevels - List of siblings of the given node.
		 * @return
		 */
		protected static Tree<String> binarizeTreeHelper(Tree<String> tree,
				int numChildrenGenerated,
				String intermediateLabel,
				int markovHorizontalOrder,
				List<String> siblingLevels) {
	
			Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
			List<Tree<String>> children = new ArrayList<Tree<String>>();
			children.add(binarizeTree(leftTree, markovHorizontalOrder));
			if (numChildrenGenerated < tree.getChildren().size() - 1) {
				siblingLevels.add(leftTree.getLabel());
				Tree<String> rightTree = binarizeTreeHelper(tree, numChildrenGenerated + 1, intermediateLabel + "_" + leftTree.getLabel(),
						markovHorizontalOrder, siblingLevels);
				children.add(rightTree);
			}
	
			// Now if Horizontal markov process is enabled.
			if (markovHorizontalOrder > 0){
				intermediateLabel = AnnotateWithHorizontalMarkovization(tree, markovHorizontalOrder, siblingLevels);
			}
	
			return new Tree<String>(intermediateLabel, children);
		}
		
		/***
		 * Un-annotate the tree. It has added functionality to un-annotate to handle the vertical markovization.
		 * @param annotatedTree
		 * @return
		 */
		public static Tree<String> unAnnotateTree(Tree<String> annotatedTree) {
			Tree<String> unAnnotateTree = TreeAnnotations.unAnnotateTree(annotatedTree);
			
			// Un-annotate tree added due to veritcal markovization.
			unAnnotateTree = ( new MarkovVerticalNodeAnnotationStripper()).transformTree(unAnnotateTree);
		    return unAnnotateTree;
		}
}
