package edu.berkeley.nlp.assignments;

import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees.TreeTransformer;

/***
 * This call un-annotate the annontation added during vertical markovization.
 * @author rajkuma
 *
 */
public class MarkovVerticalNodeAnnotationStripper implements TreeTransformer<String> {

	@Override
	public Tree<String> transformTree(Tree<String> tree) {
		// TODO Auto-generated method stub
		if (tree == null || tree.isLeaf()){
			return tree;
		}
		
		String currentTag = tree.getLabel();
		int ancestoreStartIndex = currentTag.indexOf('^');
		if (ancestoreStartIndex > 0){
			currentTag = currentTag.substring(0, ancestoreStartIndex);
			tree.setLabel(currentTag);
		}
		
		for (Tree<String> child : tree.getChildren()){
			transformTree(child);
		}
		
		return tree;
	}

}
