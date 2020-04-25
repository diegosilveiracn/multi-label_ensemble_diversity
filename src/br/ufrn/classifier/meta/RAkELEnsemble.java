package br.ufrn.classifier.meta;

import java.util.ArrayList;
import java.util.List;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.HammingLoss;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class RAkELEnsemble {
	private double confidence;

	private double error;

	private double hit;

//	private List<MLkNN> components;
	private List<RAkEL> components;

	public RAkELEnsemble(int numComponent, double confidence) {
		this.confidence = confidence;
//		this.components = new ArrayList<MLkNN>();
		this.components = new ArrayList<RAkEL>();
		for (int i = 0; i < numComponent; i++) {
//			this.components.add(new MLkNN());
			this.components.add(new RAkEL(new LabelPowerset(new J48())));
		}
	}

	public void build(Instances instances, String xml) throws Exception {
		for (int i = 0; i < components.size(); i++) {
			Resample resample = new Resample();
			resample.setOptions(new String[] { "-S", String.valueOf(i + 1),
					"-Z", "100.0" });
			resample.setInputFormat(instances);
			Instances newInstances = Filter.useFilter(instances, resample);

			components.get(i).build(new MultiLabelInstances(newInstances, xml));
		}
	}

	public MultiLabelOutput makePrediction(Instance instance, int numLabel,
			int[] indecesLabel) throws Exception {
		int[] predict0 = new int[numLabel];
		int[] predict1 = new int[numLabel];
//		for (MLkNN c : components) {
		for (RAkEL c : components) {
			MultiLabelOutput ml = c.makePrediction(instance);
			boolean[] bipart = ml.getBipartition();

			if (this.evaluateHL(ml, instance, numLabel, indecesLabel) <= this.confidence) {
				this.hit += 1.0/this.components.size();
			} else {
				this.error += 1.0/this.components.size();
			}

			for (int i = 0; i < bipart.length; i++) {
				if (bipart[i]) {
					predict1[i]++;
				} else {
					predict0[i]++;
				}
			}
		}
		return this.combinationPredict(predict0, predict1);
	}

	private MultiLabelOutput combinationPredict(int[] p0, int[] p1) {
		boolean[] finalPredict = new boolean[p0.length];
		for (int i = 0; i < p0.length; i++) {
			finalPredict[i] = p1[i] > p0[i];
		}
		return new MultiLabelOutput(finalPredict);
	}

	public double evaluate(Instance instance, int numLabel, int[] indecesLabel)
			throws Exception {
		MultiLabelOutput ml = this.makePrediction(instance, numLabel,
				indecesLabel);
		return evaluateHL(ml, instance, numLabel, indecesLabel);
	}

	private double evaluateHL(MultiLabelOutput ml, Instance instance,
			int numLabel, int[] indecesLabel) {
		HammingLoss hl = new HammingLoss();
		hl.reset();
		hl.update(
				ml,
				new GroundTruth(getTrueLabels(instance, numLabel, indecesLabel)));
		return hl.getValue();
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

	public double getConfidence() {
		return confidence;
	}

	public double getError() {
		return error;
	}

	public double getHit() {
		return hit;
	}
}