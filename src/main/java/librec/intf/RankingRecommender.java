/**
 * 
 */
package librec.intf;

import coding.eval.EvaluationMetrics;
import coding.eval.RankEvaluator;
import coding.io.Logs;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;

/**   
*    
* project_name：librec_zhouge   
* type_name：RankingRecommender   
* type_description：   
* creator：zhoug_000   
* create_time：2015年5月3日 下午11:22:27   
* modification_user：zhoug_000   
* modification_time：2015年5月3日 下午11:22:27   
* @version    1.0
*    
*/
public class RankingRecommender extends IterativeRecommender {
	protected final static int lossCode = 2;
	/**
	 * @param trainMatrix
	 * @param testMatrix
	 * @param fold
	 */
	public RankingRecommender(SparseMatrix trainMatrix,
			SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		// TODO Auto-generated constructor stub
	}
	@Override
	protected void initModel() throws Exception {
		// initialization
		super.initModel();
		System.out.println(RankEvaluator.printTitle()+ "\tAvgP");

	}
	/**
	 * Post each iteration, we do things:
	 * 
	 * <ol>
	 * <li>print debug information</li>
	 * <li>check if converged</li>
	 * <li>if not, adjust learning rate</li>
	 * </ol>
	 * 
	 * @param iter
	 *            current iteration
	 * 
	 * @return boolean: true if it is converged; false otherwise
	 * 
	 */
	protected boolean isConverged(int iter) {
		
		try {
			if (iter<numPrintbegin) {
				return false;
			}
			if (iter==1) {
				eval(iter);
			}
			else if ( iter%numPrintIters==0 ) {
				eval(iter);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return false;
	}
	
	public void eval(int iter){
		EvaluationMetrics evalPointTrain = this.evaluate(trainMatrix);
		EvaluationMetrics evalPointTest = this.evaluate(testMatrix);
		RankEvaluator evalRank = new RankEvaluator(trainMatrix, testMatrix, evalPointTrain.getPrediction().plus(evalPointTest.getPrediction()));
		System.out.println(algoName+"\t"+lossCode + "\t"+iter + "\t"  + evalRank.printOneLine() + "\t" + String.format("%.4f", evalPointTest.getAveragePrecision()));
	
	}
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param testMatrix The rating matrix with test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	public EvaluationMetrics evaluate(SparseMatrix testMatrix) {
		SparseMatrix predicted =testMatrix.clone();
		
		for (int u = 0; u < numUsers; u++) {
			SparseVector pu=testMatrix.row(u);
			int[] testItems = pu.getIndex();
			if (testItems != null) {
				for (int t = 0; t < testItems.length; t++) {
					int i = testItems[t];
					double prediction = predict(u, i);
					predicted.set(u, i, prediction);
				}
			}
		}
		
		return new EvaluationMetrics(testMatrix, predicted,minRate,maxRate,cf.getInt("num.reclist.len"));
	}
}
