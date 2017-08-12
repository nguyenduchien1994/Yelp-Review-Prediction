Included in the Yelp Prediction project are two python files, a perl script, the report PDF, and this README.  Each of the usages of them will be explained here.

------------------------------------

data_prep.py

to launch    : $ python data_prep.py
requirements : the yelp_academic_dataset_business.json and yelp_academic_dataset_review.json should be in the same directory, json and pandas
usage        : this program will use the two aforementioned yelp json files to produce a csv that can be more quickly read and used in the machine learning program
output       : reviews.csv 

------------------------------------

predict.py

to launch    : $ ./predict.py [-v/-q] [options]
requirements : the reviews.csv should exist, pandas, sklearn, os, sys, argparse
usage        : will evaluate both or one of the machine learning algorithms over the data in reviews.csv
output       : prints the accuracy and mean squared error of the models
notes        : this one may require changing the shebang line at the top
               you may have to modify this to be executable on your system ($chmod +x predict.py)
               Windows systems might need to launch on cmd as follows since the shebang is ignored
                        > python predict.py options
               For more info, call this $./predict.py --help

------------------------------------

RUNME.pl

to launch    : $ ./RUNME.pl
requirements : Perl, possibly changing the script to be executable, all the requirements for predict working, potentially a Unix like subsystem
usage        : runs several instances of the predict.py code.  Number for each review count can be altered internally
output       : multiple of predict.py

------------------------------------

reviews.csv

This is a result of data_prep.py icluded to make the users experience easier.

------------------------------------

report.pdf

This is the project report.  It only needs to be opened in some pdf viewer.               

------------------------------------

Thank you Professor Chin and Xiao for teaching this class

-John, Sherman, and Hien