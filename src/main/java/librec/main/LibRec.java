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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.main;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.Map.Entry;

import coding.io.Configer;
import coding.io.FileIO;
import coding.io.Logs;
import coding.io.Strings;
import coding.io.net.EMailer;
import coding.system.Dates;
import coding.system.Systems;
import librec.baseline.Commonneighbors;
import librec.baseline.ConstantGuess;
import librec.baseline.GlobalAverage;
import librec.baseline.ItemAverage;
import librec.baseline.MostPopular;
import librec.baseline.RandomGuess;
import librec.baseline.UserAverage;
import librec.data.DataDAO;
import librec.data.DataSplitter;
import librec.data.SparseMatrix;
import librec.intf.Recommender;
import librec.intf.Recommender.Measure;
import librec.ranking.yulu.*;

/**
 * Main Class of the LibRec Library
 * 
 * @author guoguibing
 * 
 */
public class LibRec {
	// version: MAJOR version (significant changes), followed by MINOR version (small changes, bug fixes)
	private static String version = "1.2";
	// configuration
	private static Configer cf;
	private static String algorithm;
	private static File trainFile;
	
	// rate DAO object
	private static DataDAO rateDao;

	// rating matrix
	private static SparseMatrix rateMatrix = null;

	public static void main(String[] args) throws Exception {
		String configFile = "configs/walkranker.conf";
		String trainPath = "/data/ml100k/u1.base";
		String testPath = "/data/ml100k/u1.test";
		
		// read arguments
		int i = 0;
		while (i < args.length) {
			if (args[i].equals("-c")) { // configuration file
				configFile = args[i + 1];
				i += 2;
				cf = new Configer(configFile);
			}else if(args[i].equals("-train")){
				trainPath = args[i + 1];
				i += 2;
			}else if(args[i].equals("-test")){
				testPath = args[i + 1];
				i += 2;
			}
			else if (args[i].equals("-v")) { // print out short version information
				System.out.println("LibRec version " + version);
				System.exit(0);
			}
			else if (args[i].equals("--version")) { // print out full version information
				printMe();
				System.exit(0);
			}
		}
		if (cf==null) {
			cf = new Configer(configFile);
		}
		// get configuration file
		
		trainFile=new File(trainPath);
		// debug info
		debugInfo();
		
		// prepare data
		rateDao = new DataDAO(trainPath);
		rateMatrix = rateDao.readData(cf.getDouble("val.binary.threshold"));

		// config general recommender
		Recommender.cf = cf;
		Recommender.rateMatrix = rateMatrix;
		Recommender.rateDao = rateDao;

		// run algorithms
		runTestFile(testPath);

		// collect results
		String destPath = FileIO.makeDirectory("Results/"+algorithm);
		String dest = destPath + algorithm +"_"+ trainFile.getName()+ "@" + Dates.now()+".txt";
		FileIO.copyFile("results.txt", dest);

	}
	
	/**
	 * Interface to run testing using data from an input file
	 * 
	 */
	private static void runTestFile(String path) throws Exception {

		//it will and itemid and userid that no in the train set
		DataDAO testDao = new DataDAO(path, rateDao.getUserIds(), rateDao.getItemIds());
		SparseMatrix testMatrix = testDao.readData(false,cf.getDouble("val.binary.threshold"));

		Recommender algo = getRecommender(new SparseMatrix[] { rateMatrix, testMatrix }, -1);
		algo.execute();

		//printEvalInfo(algo, algo.measures);
	}

	/**
	 * print out the evaluation information for a specific algorithm
	 */
	private static void printEvalInfo(Recommender algo, Map<Measure, Double> ms) {

		String result = Recommender.getEvalInfo(ms);
		String time = Dates.parse(ms.get(Measure.TrainTime).longValue()) + ","
				+ Dates.parse(ms.get(Measure.TestTime).longValue());
//		String evalInfo = String.format("%s,%s,%s,%s", algo.algoName, result, algo.toString(), time);
//
//		Logs.info(evalInfo);
	}

	/**
	 * Send a notification of completeness
	 * 
	 */
	private static void notifyMe(String dest) throws Exception {
		if (!cf.isOn("is.email.notify"))
			return;

		EMailer notifier = new EMailer();
		Properties props = notifier.getProps();

		props.setProperty("mail.debug", "false");

		String host = cf.getString("mail.smtp.host");
		String port = cf.getString("mail.smtp.port");
		props.setProperty("mail.smtp.host", host);
		props.setProperty("mail.smtp.port", port);
		props.setProperty("mail.smtp.auth", cf.getString("mail.smtp.auth"));

		props.put("mail.smtp.socketFactory.port", port);
		props.put("mail.smtp.socketFactory.class", "javax.net.ssl.SSLSocketFactory");

		final String user = cf.getString("mail.smtp.user");
		final String pwd = cf.getString("mail.smtp.password");
		props.setProperty("mail.smtp.user", user);
		props.setProperty("mail.smtp.password", pwd);

		props.setProperty("mail.from", user);
		props.setProperty("mail.to", cf.getString("mail.to"));

		props.setProperty("mail.subject", FileIO.getCurrentFolder() + "." + algorithm + " [" + Systems.getIP() + "]");
		props.setProperty("mail.text", "Program was finished @" + Dates.now());

		String msg = "Program [" + algorithm + "] has been finished !";
		notifier.send(msg, dest);
	}

	/**
	 * @return a recommender to be run
	 */
	private static Recommender getRecommender(SparseMatrix[] data, int fold) throws Exception {

		SparseMatrix trainMatrix = data[0], testMatrix = data[1];
		algorithm = "AsyUnifiedRanker";
		return new AsyUnifiedRanker(trainMatrix, testMatrix, fold);
			
	}

	/**
	 * Print out debug information
	 */
	private static void debugInfo() {
		String cv = "kFold: " + cf.getInt("num.kfold")
				+ (cf.isOn("is.parallel.folds") ? " [Parallel]" : " [Singleton]");

		float ratio = cf.getFloat("val.ratio");
		int givenN = cf.getInt("num.given.n");
		float givenRatio = cf.getFloat("val.given.ratio");

		String cvInfo = cf.isOn("is.cross.validation") ? cv : (ratio > 0 ? "ratio: " + ratio : "given: "
				+ (givenN > 0 ? givenN : givenRatio));

		String testPath = cf.getPath("dataset.testing");
		boolean isTestingFlie = !testPath.equals("-1");
		String mode = isTestingFlie ? String.format("Testing:: %s.", Strings.last(testPath, 38)) : cvInfo;

		if (!Recommender.isRankingPred) {
			String view = cf.getString("rating.pred.view");
			switch (view.toLowerCase()) {
				case "cold-start":
					mode += ", " + view;
					break;
				case "trust-degree":
					mode += String.format(", %s [%d, %d]",
							new Object[] { view, cf.getInt("min.trust.degree"), cf.getInt("max.trust.degree") });
					break;
				case "all":
				default:
					break;
			}
		}
		String debugInfo = String.format("Training: %s, %s", Strings.last(cf.getPath("dataset.training"), 38), mode);
		Logs.info(debugInfo);
	}

	/**
	 * Print out software information
	 */
	public static void printMe() {
		String readme = "\nLibRec version " + version + ", copyright (C) 2014 Guibing Guo Change By Zhouge \n\n"

		/* Description */
		+ "LibRec is free software: you can redistribute it and/or modify \n"
				+ "it under the terms of the GNU General Public License as published by \n"
				+ "the Free Software Foundation, either version 3 of the License, \n"
				+ "or (at your option) any later version. \n\n"

				/* Usage */
				+ "LibRec is distributed in the hope that it will be useful, \n"
				+ "but WITHOUT ANY WARRANTY; without even the implied warranty of \n"
				+ "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the \n"
				+ "GNU General Public License for more details. \n\n"

				/* licence */
				+ "You should have received a copy of the GNU General Public License \n"
				+ "along with LibRec. If not, see <http://www.gnu.org/licenses/>.";

		System.out.println(readme);
	}
}
