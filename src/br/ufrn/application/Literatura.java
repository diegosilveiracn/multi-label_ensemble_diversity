package br.ufrn.application;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.loss.RankingLoss;
import mulan.evaluation.measure.HammingLoss;

import org.apache.commons.math3.stat.descriptive.moment.Mean;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Literatura {

	public static void main(String[] args) throws Exception {
		String path = System.getProperty("user.dir");
		for (String base : bases()) {
			String arff = path + "/arff/" + base + ".arff";
			String xml = path + "/xml/" + base + ".xml";

			Instances dataset = DataSource.read(arff);

			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < 10; i++) {
				Instances train = dataset.trainCV(10, i);
				Instances test = dataset.testCV(10, i);

				RAkEL alg = new RAkEL(new LabelPowerset(new J48()));
//				MLkNN alg = new MLkNN();
				MultiLabelInstances multiLabelInstances = new MultiLabelInstances(
						train, xml);
				alg.build(multiLabelInstances);

				double[] values1 = new double[test.size()];
				double[] values2 = new double[test.size()];
				for (int j = 0; j < test.size(); j++) {
					Instance instance = test.get(j);

					MultiLabelOutput ml = alg.makePrediction(instance);

					HammingLoss hl = new HammingLoss();
					hl.reset();
					hl.update(
							ml,
							new GroundTruth(getTrueLabels(instance,
									multiLabelInstances.getNumLabels(),
									multiLabelInstances.getLabelIndices())));

					values1[j] = hl.getValue();
					
					RankingLoss rl = new RankingLoss();
					values2[j] = rl.computeLoss(ml, new GroundTruth(getTrueLabels(instance,
							multiLabelInstances.getNumLabels(),
							multiLabelInstances.getLabelIndices())).getTrueLabels());
				}
				Mean mean = new Mean();
				System.out.println(base + "," + mean.evaluate(values1) + "," + mean.evaluate(values2));
				sb.append(base);
				sb.append(",");
				sb.append(mean.evaluate(values1));
				sb.append(",");
				sb.append(mean.evaluate(values2));
				sb.append("\n");
			}
			saveResults(sb.toString(), base);
		}
	}

	public static List<String> bases() {
		List<String> l = new ArrayList<String>();
		// N l.add("bibtex");
//		l.add("birds");
		// N l.add("bookmarks");
//		l.add("cal500");
		// l.add("corel5k");
		// N l.add("delicious");
//		l.add("emotions");
		// N l.add("enron");
//		l.add("flags");
//		l.add("genbase");
//		l.add("medical");
		l.add("scene");
		l.add("yeast");
		return l;
	}

	private static boolean[] getTrueLabels(Instance instance, int numLabels,
			int[] labelIndices) {
		boolean[] trueLabels = new boolean[numLabels];
		for (int counter = 0; counter < numLabels; counter++) {
			int classIdx = labelIndices[counter];
			String classValue = instance.attribute(classIdx).value(
					(int) instance.value(classIdx));
			trueLabels[counter] = classValue.equals("1");
		}
		return trueLabels;
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