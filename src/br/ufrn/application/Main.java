package br.ufrn.application;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import mulan.data.MultiLabelInstances;

import org.apache.commons.math3.stat.descriptive.summary.Sum;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import br.ufrn.classifier.meta.RAkELEnsemble;

public class Main {
	public static void main(String[] args) throws Exception {
		String path = System.getProperty("user.dir");

		for (String base : bases()) {
			System.out.println(base);
			
			String arff = path + "/arff/" + base + ".arff";
			String xml = path + "/xml/" + base + ".xml";

			Instances dataset = DataSource.read(arff);

			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < 10; i++) {
				Instances train = dataset.trainCV(10, i);
				Instances test = dataset.testCV(10, i);

				double[] v = new double[test.size()];
				double goodDiverdity = 0;
				double badDiversity = 0;
				for (int j = 0; j < test.size(); j++) {
					Instance instance = test.get(j);

					RAkELEnsemble alg = new RAkELEnsemble(10, 0.002);
//					MLkNNEnsemble alg = new MLkNNEnsemble(10,0.02);
					alg.build(train, xml);

					v[j] = alg.evaluate(instance, new MultiLabelInstances(
							dataset, xml).getNumLabels(),
							new MultiLabelInstances(dataset, xml)
									.getLabelIndices());
					if (v[j] <= alg.getConfidence()) {
						goodDiverdity += alg.getError();
					} else {
						badDiversity += alg.getHit();
					}
				}

				Sum sum = new Sum();
				System.out.println(base + "," + sum.evaluate(v) / train.size()
						+ "," + goodDiverdity / train.size() + ","
						+ badDiversity / train.size());
				sb.append(base);
				sb.append(",");
				sb.append(sum.evaluate(v) / train.size());
				sb.append(",");
				sb.append(goodDiverdity / train.size());
				sb.append(",");
				sb.append(badDiversity / train.size());
				sb.append("\n");		
			}
			saveResults(sb.toString(), base);
		}
	}

	public static List<String> bases() {
		List<String> l = new ArrayList<String>();
//	N	l.add("bibtex");
//		l.add("birds");
//	N	l.add("bookmarks");
//		l.add("cal500");
//		l.add("corel5k");
//	N	l.add("delicious");
//		l.add("emotions");
//	N	l.add("enron");
//		l.add("flags");
		l.add("genbase");	
//		l.add("medical");
//		l.add("scene");
//		l.add("yeast");
		return l;
	}

	public static void saveResults(String text, String fileName) {
		String path = System.getProperty("user.dir");

		try {
			File file = new File(path + "/output/" + fileName);

			FileWriter fileWriter = new FileWriter(file);
			fileWriter.write(text);
			fileWriter.close();
		} catch (IOException e) {
			System.out.println(e.toString());
		}
	}
}