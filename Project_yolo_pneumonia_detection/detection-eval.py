import sys
sys.path.append('/Users/justina/Documents/uc/uc_courses/fall20/mlcancer/project/mAP')
from main import eval
import numpy as np
import os
import argparse
import pandas as pd
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--multiple_iou', action='store_true')
	parser.add_argument('--iou', type=float, default=0.5)
	parser.add_argument('--iou_stop', type=float,  default = 0.75)
	parser.add_argument('--iou_step', type=float,  default=0.05)
	parser.add_argument('--truth_dir', default = '/Users/justina/Documents/uc/uc_courses/fall20/mlcancer/project/mAP/input/ground-truth')
	parser.add_argument('--detection_dir', default = '/Users/justina/Documents/uc/uc_courses/fall20/mlcancer/project/mAP/input/detection-results')
	parser.add_argument('--annotations', default = '/Users/justina/Documents/uc/uc_courses/fall20/mlcancer/project/s1_test.csv')
	parser.add_argument('--predictions', default = '/Users/justina/Documents/uc/uc_courses/fall20/mlcancer/project/final_prediction_s1_5cv_noClassifier.csv')
	return parser.parse_args()

def remove_files(dir):
	filelist = [f for f in os.listdir(dir) if f.endswith(".txt")]
	for f in filelist:
		os.remove(os.path.join(dir, f))

def reformat_pred_df(df):
	detection = pd.DataFrame(columns=['patientId', 'class', 'left', 'top','right','bottom'])

	for i,row in df.iterrows():
		if isinstance(row['PredictionString'],str):
			split_string = row['PredictionString'].split(' ')
			num_boxes = int(len(split_string)/5)
			
			for n in range(0, num_boxes*5, 5):
				confidence = float(split_string[n])
				left = float(split_string[n+1])
				top = float(split_string[n+2])
				right = left + float(split_string[n+3])
				bottom = top + float(split_string[n+4])
				data = {'patientId':row['patientId'], 'class':'opacity', 'confidence': confidence,'left':left, 'top':top, 'right':right, 'bottom':bottom}
				detection = detection.append(data, ignore_index=True)
		else:
			data = {'patientId':row['patientId'], 'class':float('NaN'), 'confidence':float('NaN'),'left':float('NaN'), 'top':float('NaN'), 'right':float('NaN'), 'bottom':float('NaN')}
			detection = detection.append(data, ignore_index=True)
	return detection

def main(args):

	# Remove existing files from the input directories
	ground_truth = args.truth_dir
	detection_results= args.detection_dir
	remove_files(ground_truth)
	remove_files(detection_results)

	# Load and reformat truth and prediction annotations
	annotations = pd.read_csv(args.annotations)
	df = pd.read_csv(args.predictions)
	detection = reformat_pred_df(df)

	annotations.rename(columns={'x': 'left', 'y':'top'}, inplace=True)
	annotations['right'] = annotations.apply(lambda x: x['left']+x['width'],axis=1)
	annotations['bottom'] = annotations.apply(lambda x: x['top']+x['height'],axis=1)
	annotations = annotations[['patientId', 'class','left','top','right','bottom']].reset_index(drop=True)

	# Create .txt files for each image with bounding box coordinates
	for patient in annotations.patientId.unique():
		f = open(os.path.join(ground_truth,f'{patient}.txt'), 'a+')
		for i,row in annotations[annotations.patientId==patient].iterrows():
			if row['class'] == 'Lung Opacity':
				f.write(f"opacity {row['left']} {row['top']} {row['right']} {row['bottom']}\n")
		f.close()

	for patient in detection.patientId.unique():
		f = open(os.path.join(detection_results,f'{patient}.txt'), 'a+')
		for i,row in detection[detection.patientId==patient].iterrows():
			if row['class'] == 'opacity':
				f.write(f"opacity {row['confidence']} {row['left']} {row['top']} {row['right']} {row['bottom']}\n")
		f.close()

	# Calculate mAP
	scores = []
	if args.multiple_iou:
		ious = np.arange(args.iou, args.iou_stop+args.iou_step, args.iou_step)
	else:
		ious = [args.iou]

	print(ious)
	for iou in ious:
		print(iou)
		print(eval(iou))
		scores.append(eval(iou))

	print(f'mAP over IoU {ious}: {np.mean(scores)}')

if __name__ == '__main__':
	args = parse_args()
	main(args)
