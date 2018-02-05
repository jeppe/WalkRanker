// Copyright (C) 2015 zhouge
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.intf;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import librec.data.DataDAO;
import librec.data.DenseMatrix;
import librec.data.SparseMatrix;
import librec.intf.Recommender.Measure;
import coding.io.FileIO;
import coding.io.KeyValPair;
import coding.io.Lists;
import coding.io.Logs;
import coding.io.Strings;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

/**
 * Recommenders in which tag information is used
 * 
 * @author zhouge
 * 
 */
public abstract class TagRecommender extends IterativeRecommender {

	// tagMatrix: social rate matrix, indicating a user is connecting to a
	// number of other users
	// tagMatrix: inverse social matrix, indicating a user is connected by a
	// number of other users
	protected static Map<String, Map<String, Set<String>>> trainMap;
	protected static Map<String, Map<String, Set<String>>> testMap;
	// social regularization
	protected static float regt;
	protected static BiMap<String, Integer> tagids;
	public static BiMap<Integer, String> idtags;
	protected DenseMatrix tag_factor;
	protected static int numtag;
	private static int totalTest = 0;

	// initialization
	static {
		String tagPathtrain = cf.getString("dataset.tag.train");
		String tagPathtest = cf.getString("dataset.tag.test");
		try {
			trainMap = readTag(tagPathtrain, true);
			testMap = readTag(tagPathtest, false);
			if (idtags == null) {
				idtags = tagids.inverse();
			}
			numtag = tagids.size();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	public TagRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix,
			int fold) {
		super(trainMatrix, testMatrix, fold);
	}

	/**
	 * @param filename
	 * @param train
	 *            是否为训练集，如果是测试集则不把id 加入集合
	 * @return 读取 文件 userid,tagid,itemid 创建时间：2015年4月30日下午1:20:31
	 *         修改时间：2015年4月30日 下午1:20:31
	 */
	public static Map<String, Map<String, Set<String>>> readTag(
			String filename, boolean train) {
		Map<String, Map<String, Set<String>>> result = new HashMap<String, Map<String, Set<String>>>();
		try {
			List<String> list = FileIO.readAsList(filename);
			for (String oneline : list) {
				String[] split = oneline.split("\t");
				if (!result.containsKey(split[0])) {
					Map<String, Set<String>> tempMap = new HashMap<>();
					result.put(split[0], tempMap);
				}
				if (!result.get(split[0]).containsKey(split[1])) {
					Set<String> itemstr = new HashSet();
					result.get(split[0]).put(split[1], itemstr);
				}
				result.get(split[0]).get(split[1]).add(split[2]);
				if (train) {
					if (tagids == null) {
						tagids = HashBiMap.create();
					}
					// inner id starting from 0
					int row = tagids.containsKey(split[1]) ? tagids
							.get(split[1]) : tagids.size();
					tagids.put(split[1], row);
				} else {
					totalTest++;
				}
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return result;
	}

	@Override
	protected void generateRecommenderList(String fileName) {
		// for each user
		Map<String, List<String>> recomlistMap = new HashMap<>();
		ExecutorService executor = Executors
				.newFixedThreadPool(numberOfCoreForMeasure);
		List<Future<Map<String, List<String>>>> results = new ArrayList<>();

		for (String userstr : testMap.keySet()) {
			if (!rateDao.getUserIds().containsKey(userstr)) {
				System.out.println("no user " + userstr);
				continue;
			}
			int userid = rateDao.getUserId(userstr);
			Set<String> bugitemSet = new HashSet<>();
			for (String tagstr : trainMap.get(userstr).keySet()) {
				bugitemSet.addAll(trainMap.get(userstr).get(tagstr));
			}
			for (String tagstr : testMap.get(userstr).keySet()) {
				if (!tagids.containsKey(tagstr)) {
					System.out.println("no tag " + tagstr);
					continue;
				}
				int tagid = tagids.get(tagstr);
				// the item that have bug by user

				List<String> canitem = new ArrayList<>();
				for (String itemstr : rateDao.getItemIds().keySet()) {
					if (bugitemSet.contains(itemstr)) {
						continue;
					}
					canitem.add(itemstr);
				}
				results.add(executor
						.submit(new calvalue(canitem, userid, tagid)));
			}
		}

		for (Future<Map<String, List<String>>> result : results) {
			try {
				recomlistMap.putAll(result.get());
			} catch (InterruptedException | ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		executor.shutdown();
		try {
			FileIO.writeMapValueInList(fileName, recomlistMap, false, ",");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		recall(fileName);
	}

	private class calvalue implements Callable<Map<String, List<String>>> {
		private final List<String> candItems;
		private final int u;
		private final int tag;

		calvalue(List<String> candItems, int u, int tag) {
			this.candItems = candItems;
			this.u = u;
			this.tag = tag;
		}

		@Override
		public Map<String, List<String>> call() {
			Map<String, List<String>> resultMap = new HashMap<>();
			String userstr = rateDao.getUserId(u);
			String tagstr = idtags.get(tag);
			List<String> predict = getTopkItem(candItems, u, tag);
			double count = 0, total = 0;
			Set<String> real = testMap.get(userstr).get(tagstr);
			total = real.size();
			for (String itemid : predict) {
				if (real.contains(itemid)) {
					count++;
				}
			}
			System.out.println(userstr + "  " + tagstr + "recall :" + count
					/ total + " total:" + total);
			resultMap.put(userstr + "," + tagstr, predict);

			return resultMap;
		}
	}

	public List<String> getTopkItem(final List<String> candItems, final int u,
			final int tag) {
		Map<Integer, Double> itemScores = new HashMap<>();
		for (String itemidStr : candItems) {
			int itemid = rateDao.getItemId(itemidStr);
			itemScores.put(itemid, predict(u, itemid, tag));
		}
		List<String> rankedItems = new ArrayList<>();
		if (itemScores.size() > 0) {
			List<KeyValPair<Integer>> sorted = Lists.sortMap(itemScores, true);
			List<KeyValPair<Integer>> recomd = (numRecs < 0 || sorted.size() <= numRecs) ? sorted
					: sorted.subList(0, numRecs);

			for (KeyValPair<Integer> kv : recomd)
				rankedItems.add(rateDao.getItemId(kv.getKey()));
		}

		return rankedItems;
	}

	// cal recall
	public void recall(String resultFile) {
		int length = 5;
		double hitcount = 0;
		try {
			List<String> resultlist = FileIO.readAsList(resultFile);

			for (String oneline : resultlist) {
				String[] split = oneline.split(",");

				String userstr = split[0];
				String tagstr = split[1];
				for (int i = 2; i < 2 + length; i++) {
					String itemstr = split[i];
					if (testMap.get(userstr).get(tagstr).contains(itemstr)) {
						hitcount++;
					}
				}
			}

			System.out.println("the recall @" + length + " :" + hitcount
					/ totalTest);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	/**
	 * default prediction method
	 */
	@Override
	protected double predict(int u, int j) {
		return DenseMatrix.rowMult(P, u, Q, j);
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

		// print out debug info
		if (verbose) {
			Logs.debug("{}{} iter {}: errs = {}, delta_errs = {}, loss = {}, delta_loss = {}, learn_rate = {}",
					new Object[] { algoName, foldInfo, iter, (float) errs, (float) (last_errs - errs), (float) loss,
							(float) (Math.abs(last_loss) - Math.abs(loss)), (float) lRate });
		}

		System.out.println(iter);
		
		if (iter%numPrintIters==0) {
		
			generateRecommenderList(cf.getString("recresult")+algoName+numFactors+"_"+numIters);
			return false;
		}else if(iter==numIters){
			System.out.println(iter);
			generateRecommenderList(cf.getString("recresult")+algoName+numFactors+"_"+numIters);
			return true;
		}else {
			return false;
		}
	}
	public static int getTagids(String tags) {
		return tagids.get(tags);
	}

	protected double predict(int user, int item, int tag) {
		return 0;
	}

	public static String getIdtags(int id) {
		if (idtags == null) {
			idtags = tagids.inverse();
		}
		return idtags.get(id);
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { initLRate, maxLRate, regB, regU,
				regI, regt, numFactors, numIters, isBoldDriver }, ",");
	}

}
