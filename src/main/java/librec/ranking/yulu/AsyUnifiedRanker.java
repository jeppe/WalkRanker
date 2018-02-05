//Copyright (C) 2017 Lu Yu
//
//This file is part of LibRec.
//
//LibRec is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//LibRec is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//
package librec.ranking.yulu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import coding.io.KeyValPair;
import coding.io.Lists;
import coding.io.Logs;
import coding.io.Strings;
import coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.Recommender.Measure;
import librec.intf.IterativeRecommender;

public class AsyUnifiedRanker extends IterativeRecommender {
	
	private int walkNum;
	private int walkLength;
	private double walkExpWeight;
	private int walkWinSize;
	private List<List<String>> walkseq;
	protected  int sFixedSteps ;
	protected  int maxiRankSamples;
	protected static double beta;
	
	public AsyUnifiedRanker(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		
		walkNum = cf.getInt("walk.num");
		walkLength = cf.getInt("walk.length");
		walkExpWeight = cf.getDouble("walk.exp.weight");
		walkWinSize = cf.getInt("walk.window.size");
		
		isRankingPred = true;
		initByNorm = false;
		
		sFixedSteps= cf.getInt("RankBPR.sFixedSteps");
		maxiRankSamples = cf.getInt("RankBPR.maxiRankSamples");
		beta=cf.getDouble("RankBPR.beta");
		
		cUserBias = new DenseVector(numUsers);
		cItemBias = new DenseVector(numItems);
		cUserBias.init(0.01);
		cItemBias.init(0.01);
		
		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);
		userBias.init(0.01);
		itemBias.init(0.01);
		
		//index by user index not userid
		PC = new DenseMatrix(numUsers, numFactors);
		QC = new DenseMatrix(numItems, numFactors);

		// initialize model
		if (initByNorm) {
			PC.init(initMean, initStd);
			QC.init(initMean, initStd);
		} else {
			PC.init(); // P.init(smallValue);
			QC.init(); // Q.init(smallValue);
		}
	}
	
	protected void simulateWalkSeq() throws Exception{
		walkseq = new ArrayList<List<String>>();
		for (int i = 0; i < walkNum; i++){
			Logs.info(String.valueOf(i + 1) + "/" + String.valueOf(walkNum));
			for (int j = 0; j < numUsers; j++){
				SparseVector uitems = trainMatrix.row(j);
				if(uitems.getCount() == 0) continue;
				walkseq.add(user_walk_sentence(j));
			}
			for (int j = 0; j < numItems; j++){
				SparseVector iusers = trainMatrix.column(j);
				if(iusers.getCount() == 0) continue;
				walkseq.add(item_walk_sentence(j));
			}
		}
	}
	
	public static int getRandom(int[] array) {
	    int rnd = new Random().nextInt(array.length);
	    return array[rnd];
	}
	
	protected List<String> user_walk_sentence(int seed_node){	
		List<String> sen = new ArrayList<String>();
		int next = seed_node;
		sen.add("u"+String.valueOf(seed_node));
		for(int step=0; step < walkLength; step++){
			if(step%2 == 0){
				// seed node is a user
				SparseVector uitems = trainMatrix.row(next);
				int[] uitemsArray = uitems.getIndex();
				next = getRandom(uitemsArray);
				sen.add("i" + String.valueOf(next));
			}else{
				// seed node is a item
				SparseVector iusers = trainMatrix.column(next);
				int[] iusersArray = iusers.getIndex();
				next = getRandom(iusersArray);
				sen.add("u" + String.valueOf(next));
			}
		}
		return sen;
	}
	
	protected List<String> item_walk_sentence(int seed_node){	
		List<String> sen = new ArrayList<String>();
		int next = seed_node;
		sen.add("i"+String.valueOf(seed_node));
		for(int step=0; step < walkLength; step++){
			if(step%2 != 0){
				// seed node is a user
				SparseVector uitems = trainMatrix.row(next);
				int[] uitemsArray = uitems.getIndex();
				next = getRandom(uitemsArray);
				sen.add("i" + String.valueOf(next));
			}else{
				// seed node is a item
				SparseVector iusers = trainMatrix.column(next);
				int[] iusersArray = iusers.getIndex();
				next = getRandom(iusersArray);
				sen.add("u" + String.valueOf(next));
			}
		}
		return sen;
	}
	
	protected int getNodeId(String ele){
		return Integer.valueOf(ele.substring(1));
	}
	
	@Override
	protected void buildModel() throws Exception {
		// run in parallel
		ExecutorService executor = Executors.newFixedThreadPool(numberOfCoreForMeasure);
		
		simulateWalkSeq();
		int walkSeqSize = walkseq.size();
		List<Future<Integer>> futures = new ArrayList<>();
		for (int iter = 0; iter < numIters; iter++) {
			// iterative in user
			if( (iter + 1) % walkSeqSize == 0) {
				simulateWalkSeq();
				walkSeqSize = walkseq.size();
			}
			
			int senId = iter%walkSeqSize;
			List<String> sen = walkseq.get(senId);
			
			futures.add(executor.submit(new senLearner(sen)));
			
			if(iter%3000 == 2999) {
				for(Future<Integer> f : futures){
					f.get();
				}
				futures = new ArrayList<>();
			}
			if (isConverged(iter)) break;
		}
	}
	
	private class senLearner implements Callable<Integer>{
		List<String> sentence;
		private Map<Integer, Integer> seqUserMap, seqItemMap;
		
		senLearner(List<String> sen){
			sentence = sen;
		}
		
		@Override
		public Integer call() throws Exception{
			senMapInit(sentence);
			List<ContextSample> pairs;
			for(int cutIndex = 0; cutIndex < sentence.size(); cutIndex++){
				pairs = contextSample(cutIndex, sentence, walkWinSize);
				for(ContextSample cs : pairs){
					double w = Math.exp(walkExpWeight * (1 - cs.weight));
					
					if(cs.sameType){
						embUpdate(cs.sid, cs.cid, w, seqItemMap, seqUserMap);
					}
					else{
						recUpdate(getNodeId(cs.sid), getNodeId(cs.cid), w, seqItemMap, seqUserMap);
					}
				}
			}
			
			return 1;
		}
		
		protected void senMapInit(List<String> sen){
			seqUserMap = new HashMap<Integer, Integer>();
			seqItemMap = new HashMap<Integer, Integer>();
			for(int i = 0; i < sen.size(); i++){
				String ele = sen.get(i);
				if(ele.charAt(0) == 'u') seqUserMap.put(Integer.valueOf(ele.substring(1)), 1);
				else seqItemMap.put(Integer.valueOf(ele.substring(1)), 1);
			}
		}
	}
	
	
	protected List<ContextSample> contextSample(int curIndex, 
												List<String> scen, int winSize) throws Exception{
		
		List<ContextSample> pairs = new ArrayList<ContextSample>();
		String cutWord = scen.get(curIndex);
		//Random ran = new Random();
		int contextWidth = Randoms.uniform(winSize) + 1;
		
		int scenLength = scen.size();
		int lowerIndex = 0 - contextWidth;
		int upperIndex = contextWidth + 1;
		
		int ct = 0;
		for(int i = lowerIndex; i < upperIndex; i++){
			if (i == 0) continue;
			
			ct = curIndex + i;
			if(ct >= 0 && ct < scenLength){
				String context = scen.get(ct);
				if(cutWord.charAt(0) != context.charAt(0)){
					if(cutWord.charAt(0) == 'i'){
						ContextSample cs = new ContextSample(scen.get(ct), cutWord, Math.abs(i), false);
						pairs.add(cs);
					}
					else{
						ContextSample cs = new ContextSample(cutWord, scen.get(ct), Math.abs(i), false);
						pairs.add(cs);
					}
				}
				else{
					ContextSample cs = new ContextSample(cutWord, scen.get(ct), Math.abs(i), true);
					pairs.add(cs);
				}
			}
		}
		return pairs;
		
	}
	// loss = SUM_i 1.0 / rank_i
	public double rankLoss(int steps){
		double loss = 0.0;
		for(int i = 1; i < (steps + 1); i++){
			loss += 1.0 / i;
		}
		return loss;
	}
	
	public void embUpdate(String sid, String cid, double weight, 
					      Map<Integer,Integer> seqItemMap,
					      Map<Integer,Integer> seqUserMap){
		loss = 0;
		errs = 0;
		
		int s = getNodeId(sid);
		int c = getNodeId(cid);
		
		// 运行 RankBPR 算法获取预估排序值
		List<Integer> rankTuple;
		int steps;
		double rLoss;
		if(sid.charAt(0) == 'i'){
			rankTuple = embItemStepEstimate(s, c, seqItemMap);
			steps = rankTuple.get(0);
			rLoss = rankLoss(steps);
			int j = rankTuple.get(1);
		
			// update parameters
			double xsc = itemEmbPredict(s, c);
			double xsj = itemEmbPredict(s, j);
			double xscj = xsc - xsj;
			
			double cmguij = g(-xscj) * rLoss;
		      double cgui = g(-xsc);
		      for (int f = 0; f < numFactors; f++)
		      {
		        double qsf = Q.get(s, f);
		        double qcf = QC.get(c, f);
		        double qjf = QC.get(j, f);
		        
		        Q.add(s, f, lRate * weight * (cmguij * (qcf - qjf) + cgui * qcf - regI * qsf));
		        QC.add(c, f, lRate * weight * (cmguij * qsf + cgui * qsf - regI * qcf));
		        QC.add(j, f, lRate * weight * (cmguij * -qsf - regI * qjf));
		      }
		      cItemBias.add(c, this.lRate * weight * (cmguij + cgui - regI * cItemBias.get(c)));
		      cItemBias.add(j, this.lRate * weight * (-cmguij - regI * cItemBias.get(j)));
		      cItemBias.add(s, this.lRate * weight * (cgui - regI * cItemBias.get(s)));
		}
		else{
			// 运行 RankBPR 算法获取预估排序值
			rankTuple = embUserStepEstimate(s, c, seqUserMap);
			steps = rankTuple.get(0);
			double usrLoss = rankLoss(steps);
			int v = rankTuple.get(1);
			// update parameters
			double xsc = userEmbPredict(s, c);
			double xsv = userEmbPredict(s, v);
			double xscv = xsc - xsv;
			
			double cmgiuv = g(-xscv) * usrLoss;
			double cgui = g(-xsc);
			
			for (int f = 0; f < numFactors; f++) {
				double psf = P.get(s, f);
				double pcf = PC.get(c, f);
				double pvf = PC.get(v, f);

				P.add(s, f, lRate * weight * (cmgiuv * (pcf - pvf) + cgui * pcf - regU * psf));
				PC.add(c, f, lRate * weight * (cmgiuv * psf + cgui * psf - regU * pcf));
				PC.add(v, f, lRate * weight * (cmgiuv * (-psf) - regU * pvf));
			}
			
			cUserBias.add(c,lRate * weight * (cmgiuv - regU*cUserBias.get(c)));
			cUserBias.add(v,lRate * weight * (-cmgiuv - regU*cUserBias.get(v)));
			cUserBias.add(s,lRate * weight * (cgui - regU*cUserBias.get(s)));
		}		
		
	}
	public void recUpdate(int u, int i, double weight, 
						  Map<Integer,Integer> seqItemMap,
						  Map<Integer,Integer> seqUserMap){

		loss = 0;
		errs = 0;
			
		// 运行 RankBPR 算法获取预估排序值
		List<Integer> rankTuple = itemStepEstimate(u, i, seqItemMap);
		int steps = rankTuple.get(0);
		double rLoss = rankLoss(steps);
		int j = rankTuple.get(1);
		
		// 运行 RankBPR 算法获取预估排序值
		rankTuple = userStepEstimate(u, i, seqUserMap);
		steps = rankTuple.get(0);
		double usrLoss = rankLoss(steps);
		int v = rankTuple.get(1);

		// update parameters
		double xui = predict(u, i);
		double xuj = predict(u, j);
		double xvi = predict(v, i);
		double xuij = xui - xuj;
		double xiuv = xui-xvi;
		double vals = -Math.log(g(xuij));
		loss += vals;
		errs += vals;

		double cmguij = g(-xuij) * rLoss;
		double cmgiuv = g(-xiuv) * usrLoss;
		double cgui = g(-xui);
		double cgiu = 0.0;
		
		for (int f = 0; f < numFactors; f++) {
			double puf = P.get(u, f);
			double pvf = P.get(v, f);
			double qif = Q.get(i, f);
			double qjf = Q.get(j, f);

			P.add(u, f, lRate * weight * (cmguij * (qif - qjf) + (cgui + cgiu) * qif + cmgiuv*qif - regU * puf));
			P.add(v, f, lRate * weight * (cmgiuv*(-qif) - regU * pvf));
			
			Q.add(i, f, lRate * weight * (cmguij * puf + (cgui + cgiu) * puf + cmgiuv *(puf-pvf)- regI * qif));
			Q.add(j, f, lRate * weight * (cmguij * (-puf) - regI * qjf));
			
		}
		
		itemBias.add(i,lRate * weight * (cmguij + (cgui + cgiu) -regI*itemBias.get(i)));
		itemBias.add(j,lRate * weight * (-cmguij-regI*itemBias.get(j)));
		userBias.add(u,lRate * weight * (cmgiuv + (cgui + cgiu) -regU*userBias.get(u)));
		userBias.add(v,lRate * weight * (-cmgiuv-regU*userBias.get(v)));

	}
	
	/*
	在给定目标用户 uid 的情况下，该函数用于预估物品i的排序
	如果商品池中的商品数量非常庞大的话，每次迭代计算所有商品排序会消耗巨大资源
	严重影响算法模型的训练。因此，我们通过采样的方式估计商品的大致排序，根据 i
	的预估排序定义一个与排序相关的损失函数，用于动态调整 feature vectors 更新
	速率。
	算法步骤:
	1. 进行一定数量的采样次数预估商品正样本排名
	2. 根据预估排名计算与排序相关的损失值，动态调整模型的更新速率，排名估计值越大
	   更新幅度越大，反之成立
	*/
	public List<Integer> itemStepEstimate(int uid, int iid, Map<Integer,Integer> seqItemMap){
		
		// 获取用户 u 点击序列
		SparseVector pu = trainMatrix.row(uid);
	
		// 计算 u 对 i 的偏好值
		double xui = predict(uid, iid);
	
		// 固定采样次数
		int _sFixedSteps = sFixedSteps;
		boolean susFlag = true;
	
		// 存放返回结果
		List<Integer> resultsList = new ArrayList<Integer>();
	
		// 初始化样本缓存，用于存储中间采样时获取的商品序列
		Map<Integer, Double> sampleBuffer = new HashMap<Integer, Double>();
		int j=0;
		double xuji;
		// 进行采样获取偏好值大于 i 的商品 j 并记录下采样次数
		while ( _sFixedSteps > 0 && susFlag){
			do {
				j = Randoms.uniform(numItems);
			} while (pu.contains(j)||seqItemMap.containsKey(j));
	
			double xuj = predict(uid, j);
	
			//xuji = 1.0 + xuj - xui;
			xuji = beta+ xuj - xui;
			sampleBuffer.put(j, xuji);
	
			if (xuji > 0) susFlag = false;
	
			_sFixedSteps -= 1;
		}
		
		// 根据采样序列中每个样本得分从高到低对sampleBuffer进行排序，获取第一个元素
		// 这里需要你来写，排序部分我不是很懂
		// sampleBuffer.sort(key = x[1])，x[1] 为sampleBuffer 元素的第二个值
		// int selectedItemID = sampleBuffer[0][0];
		List<KeyValPair<Integer>> sorted = Lists.sortMap(sampleBuffer,
				true);
		int selectedItemID = sorted.get(0).getKey();
		// 获取预估采样步数
		int rank = 0;
	
		// 开始预估正样本 i 的当前排名
		if (sFixedSteps <= 0) rank = 1;
		else{
	
			// 这里计算公式为 er = (商品总数 - 1) / (sFixedSteps - _sFixedSteps)
			if(maxiRankSamples > 0) rank = (maxiRankSamples - 1)/(sFixedSteps - _sFixedSteps);
			else rank=(numItems-1)/(sFixedSteps -_sFixedSteps);
		}
	
		resultsList.add(rank);
		resultsList.add(selectedItemID);
	
		return resultsList; 
	}
	
	public List<Integer> embItemStepEstimate(int sid, int cid, Map<Integer,Integer> seqItemMap){
		// get seqItemMap
	
		// calculate relevance between source item and context item
		double xui = itemEmbPredict(sid, cid);
	
		// fixed sample steps
		int _sFixedSteps = sFixedSteps;
		boolean susFlag = true;
	
		// store return list
		List<Integer> resultsList = new ArrayList<Integer>();
	
		// 初始化样本缓存，用于存储中间采样时获取的商品序列
		Map<Integer, Double> sampleBuffer = new HashMap<Integer, Double>();
		int jid=0;
		double xuji;
		// 进行采样获取偏好值大于 i 的商品 j 并记录下采样次数
		while ( _sFixedSteps > 0 && susFlag){
			do {
				jid = Randoms.uniform(numItems);
			} while (seqItemMap.containsKey(jid));
	
			double xuj = itemEmbPredict(sid, jid);
	
			xuji = beta+ xuj - xui;
			sampleBuffer.put(jid, xuji);
	
			if (xuji > 0) susFlag = false;
	
			_sFixedSteps -= 1;
		}
		
		// 根据采样序列中每个样本得分从高到低对sampleBuffer进行排序，获取第一个元素
		// 这里需要你来写，排序部分我不是很懂
		// sampleBuffer.sort(key = x[1])，x[1] 为sampleBuffer 元素的第二个值
		// int selectedItemID = sampleBuffer[0][0];
		List<KeyValPair<Integer>> sorted = Lists.sortMap(sampleBuffer,
				true);
		int selectedItemID = sorted.get(0).getKey();
		// 获取预估采样步数
		int rank = 0;
	
		// 开始预估正样本 i 的当前排名
		if (sFixedSteps <= 0) rank = 1;
		else{
			if(maxiRankSamples > 0) rank = (maxiRankSamples - 1)/(sFixedSteps - _sFixedSteps);
			else rank=(numItems-1)/(sFixedSteps -_sFixedSteps);
		}
	
		resultsList.add(rank);
		resultsList.add(selectedItemID);
	
		return resultsList; 
	}
	
	public List<Integer> embUserStepEstimate(int sid, int cid, Map<Integer,Integer> seqUserMap){
		// get seqItemMap
	
		// calculate relevance between source item and context item
		double xui = userEmbPredict(sid, cid);
	
		// fixed sample steps
		int _sFixedSteps = sFixedSteps;
		boolean susFlag = true;
	
		// store return list
		List<Integer> resultsList = new ArrayList<Integer>();
	
		// 初始化样本缓存，用于存储中间采样时获取的商品序列
		Map<Integer, Double> sampleBuffer = new HashMap<Integer, Double>();
		int jid=0;
		double xuji;
		// 进行采样获取偏好值大于 i 的商品 j 并记录下采样次数
		while ( _sFixedSteps > 0 && susFlag){
			do {
				jid = Randoms.uniform(numUsers);
			} while (seqUserMap.containsKey(jid));
	
			double xuj = userEmbPredict(sid, jid);
	
			xuji = beta+ xuj - xui;
			sampleBuffer.put(jid, xuji);
	
			if (xuji > 0) susFlag = false;
	
			_sFixedSteps -= 1;
		}
		
		// 根据采样序列中每个样本得分从高到低对sampleBuffer进行排序，获取第一个元素
		// 这里需要你来写，排序部分我不是很懂
		// sampleBuffer.sort(key = x[1])，x[1] 为sampleBuffer 元素的第二个值
		// int selectedItemID = sampleBuffer[0][0];
		List<KeyValPair<Integer>> sorted = Lists.sortMap(sampleBuffer,
				true);
		int selectedItemID = sorted.get(0).getKey();
		// 获取预估采样步数
		int rank = 0;
	
		// 开始预估正样本 i 的当前排名
		if (sFixedSteps <= 0) rank = 1;
		else{
			if(maxiRankSamples > 0) rank = (maxiRankSamples - 1)/(sFixedSteps - _sFixedSteps);
			else rank=(numUsers-1)/(sFixedSteps -_sFixedSteps);
		}
	
		resultsList.add(rank);
		resultsList.add(selectedItemID);
	
		return resultsList; 
	}
	
	public List<Integer> userStepEstimate(int uid, int iid, Map<Integer,Integer> seqUserMap){
		
		// 获取商品 i 点击序列
		SparseVector pi = trainMatrix.column(iid);
	
		// 计算 u 对 i 的偏好值
		double xui = predict(uid, iid);
	
		// 固定采样次数
		int _sFixedSteps = sFixedSteps;
		boolean susFlag = true;
	
		// 存放返回结果
		List<Integer> resultsList = new ArrayList<Integer>();
	
		// 初始化样本缓存，用于存储中间采样时获取的商品序列
		Map<Integer, Double> sampleBuffer = new HashMap<Integer, Double>();
		int v=0;
		double xiuv;
		// 进行采样获取偏好值大于 i 的商品 j 并记录下采样次数
		while ( _sFixedSteps > 0 && susFlag){
			do {
				v = Randoms.uniform(numUsers);
			} while (pi.contains(v)||seqUserMap.containsKey(v));
	
			double xiv = predict(v, iid);
	
			//xiuv = 1.0 + xiv - xui;
			xiuv = beta + xiv - xui;
			sampleBuffer.put(v, xiuv);
	
			if (xiuv > 0) susFlag = false;
	
			_sFixedSteps -= 1;
		}
		
		// 根据采样序列中每个样本得分从高到低对sampleBuffer进行排序，获取第一个元素
		// 这里需要你来写，排序部分我不是很懂
		// sampleBuffer.sort(key = x[1])，x[1] 为sampleBuffer 元素的第二个值
		// int selectedItemID = sampleBuffer[0][0];
		List<KeyValPair<Integer>> sorted = Lists.sortMap(sampleBuffer,
				true);
		int selectedItemID = sorted.get(0).getKey();
		// 获取预估采样步数
		int rank = 0;
	
		// 开始预估正样本 i 的当前排名
		if (sFixedSteps <= 0) rank = 1;
		else{
	
			// 这里计算公式为 er = (商品总数 - 1) / (sFixedSteps - _sFixedSteps)
			//rank = (int) math.floor( er ) 对 er 进行向下取整
			if(maxiRankSamples > 0) rank = (maxiRankSamples - 1)/(sFixedSteps - _sFixedSteps);
			else rank=(numUsers-1)/(sFixedSteps-_sFixedSteps);
		}
	
		resultsList.add(rank);
		resultsList.add(selectedItemID);
	
		return resultsList; 
	}
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { numFactors, initLRate,
				regU, regI, sFixedSteps, maxiRankSamples, walkNum, walkLength, walkExpWeight, walkWinSize}, ",");
	}
	
	@Override
	public double  predict(int u, int j){
		return userBias.get(u)+itemBias.get(j)+DenseMatrix.rowMult(P, u, Q, j);
	}
	
	public double userEmbPredict(int sid, int cid){
		return userBias.get(sid) + cUserBias.get(cid) + DenseMatrix.rowMult(P, sid, PC, cid); 
	}
	
	public double itemEmbPredict(int sid, int cid){
		return itemBias.get(sid) + cItemBias.get(cid) + DenseMatrix.rowMult(Q, sid, QC, cid); 
	}
	private class ContextSample{  
	    String sid;  
	    String cid;  
	    double weight;
	    boolean sameType;
	    ContextSample(String sid,String cid,double weight, boolean sameType){  
	        this.sid = sid;  
	        this.cid = cid;  
	        this.weight = weight; 
	        this.sameType = sameType;
	    }  
	}
}

