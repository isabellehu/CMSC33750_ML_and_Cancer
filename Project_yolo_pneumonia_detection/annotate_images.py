from PIL import Image, ImageDraw
import pandas as pd
import os
### Modify to correct corresponding paths ###
src = '/Users/justina/Documents/uc/uc_courses/fall20/mlcancer/project/kaggle_rsna_pneumonia_dancingbears/data/s1_test_images'
dst = '/Users/justina/Documents/uc/uc_courses/fall20/mlcancer/project/kaggle_rsna_pneumonia_dancingbears/data/stage_1_test_boxed'
ground_truth_annotations = '/Users/justina/Documents/uc/uc_courses/fall20/mlcancer/project/s1_test.csv'
predictions = '/Users/justina/Documents/uc/uc_courses/fall20/mlcancer/project/final_prediction_s1_5cv.csv'

# Load ground truth annotations and adjust dataframe to correct format
truth = pd.read_csv(ground_truth_annotations)
truth.rename(columns={'x': 'left', 'y':'top'}, inplace=True)
truth['right'] = truth.apply(lambda x: x['left']+x['width'],axis=1)
truth['bottom'] = truth.apply(lambda x: x['top']+x['height'],axis=1)
truth = truth[['patientId','left','top','right','bottom']].reset_index(drop=True)

# Load predictions and adjust to correct format
pred = pd.read_csv(predictions)
detection = pd.DataFrame(columns=['patientId', 'class', 'left', 'top','right','bottom'])

for i,row in pred.iterrows():
    if isinstance(row['PredictionString'],str):
        split_str = row['PredictionString'].split(' ')
        num_boxes = int(len(split_str)/5)
        
        for n in range(0, num_boxes*5, 5):
            confidence = float(split_str[n])
            left = float(split_str[n+1])
            top = float(split_str[n+2])
            right = left + float(split_str[n+3])
            bottom = top + float(split_str[n+4])
            data = {'patientId':row['patientId'], 'class':'opacity', 'confidence': confidence,'left':left, 'top':top, 'right':right, 'bottom':bottom}
            detection = detection.append(data, ignore_index=True)
    else:
        data = {'patientId':row['patientId'], 'class':float('NaN'), 'confidence':float('NaN'),'left':float('NaN'), 'top':float('NaN'), 'right':float('NaN'), 'bottom':float('NaN')}
        detection = detection.append(data, ignore_index=True)

# Annotate each image with corresponding ground truth and predicted bounding boxes
for patient in truth.patientId.unique():
	img = Image.open(os.path.join(src, f"{patient}.png")).convert('RGB')
	draw = ImageDraw.Draw(img)
	for i, row in truth[truth.patientId==patient].iterrows():
	    if row['left'] > 0:
	        draw.rectangle([row['left'], row['top'], row['right'], row['bottom']], outline='blue',width=10)
	for i,row in detection[detection.patientId==patient].iterrows():
	    if row['left'] > 0:
	        draw.rectangle([row['left'], row['top'], row['right'], row['bottom']], outline='red', width=10)
	img.save(os.path.join(dst, f"{row['patientId']}.png"))
	                                      
