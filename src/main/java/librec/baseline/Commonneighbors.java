// Copyright (C) 2014 Guibing Guo
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.baseline;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import coding.math.Sims;
import librec.data.SparseMatrix;
import librec.intf.Recommender;

/**
 * Baseline: items are weighted by the number of ratings they received.
 * 
 * @author guoguibing
 * 
 */
public class Commonneighbors extends Recommender {

	private Map<Integer, Set<Integer>> userfollow;

	public Commonneighbors(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		// force to set as the ranking prediction method
		isRankingPred = true;
		algoName = "commonneighbors";
	}

	@Override
	protected void initModel() {
		userfollow = new HashMap<>();
	}

	@Override
	protected double ranking(int u, int j) {
		if (rateDao.getUserId(j)==null) {
			return 0;
		}
		
		if (!userfollow.containsKey(u)){
			Set<Integer> temp=trainMatrix.row(u).getIndexInset();
			userfollow.put(u,temp );
		}
		Set<Integer> ufollowee=userfollow.get(u);
		
		if (!userfollow.containsKey(j)){
			Set<Integer> temp=trainMatrix.row(j).getIndexInset();
			userfollow.put(j,temp );
		}
		Set<Integer> jfollowee=userfollow.get(j);
		return Sims.cn(ufollowee,jfollowee);
	}

}
