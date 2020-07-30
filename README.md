# drunk-detection-CNN

A CNN for classifying drunkeness based on infrared facial images (essentially the heat distribution across the face).

Written by Kevin Tang and Andrew Vernier as a class project for EECS 498/598: Affective ML.

## Results
We were able to achieve a *statistically significant accuracy of 0.87* for classifying facial infrared images as drunk or sober.

**Paper:** [Drunk Detection using Machine Learning Methods and Infrared Images](https://github.com/kvntng17/drunk-detection-CNN/blob/master/pdf/EECS_498_598_Project.pdf)

**Presentation:** [Drunk Detection using Machine Learning Methods and Infrared Images](https://youtu.be/5g5jwqD-mt0)

## Data
The images used to train our CNN comes from the [SOBER-DRUNK DATA BASE](http://www.physics.upatras.gr/sober/) created by Georgia Koukiou and Vassilis Anastassopoulos from the University of Patras. 

The images used are infrared images of 41 particpants before and after consuming various amounts of alcohol. 
Although the original data contains infrared images of many body parts, we only utilize the frontal facial infrared images for training our CNN.

Below is an example of a frontal facial infrared image represented in grayscale: <br>
![grayscale_example.png](https://github.com/kvntng17/drunk-detection-CNN/blob/master/images/grayscale_example.png)

The following README contains more detailed information about data collection and methodology: <br>
[00_Readme_sober_drunk.txt](https://github.com/kvntng17/drunk-detection-CNN/blob/master/data/00_Readme_sober_drunk.txt)

## Setup
```
# Setup virtualenv
python3 -m venv env; source env/bin/activate

# Python 3.7 or above
python -V

pip install -r requirements
```

## Usage
```
# Prepare data to be used (saves data to data_<YYYY-MM-DD-hh-mm-ss>.pickle)
python drunk_detector data --train-files data/train/*/* --test-files data/test/*/* --val-files data/validation/*/*

# Predict drunkeness using best CNN model
python drunk_detector predict -m sum -d data_*.pickle

# Train a new hyperparameter tuned CNN
python drunk_detector train -m sum -d data_*.pickle
```






