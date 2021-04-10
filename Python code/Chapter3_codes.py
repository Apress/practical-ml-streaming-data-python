#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 20:20:32 2020

@author: Dr. Sayan Putatunda
"""

""" Chapter 3 Codes """

import os
os.getcwd()  # to see the current path of working directory
os.chdir('./Python codes') 


### Creating a synthetic dataset
from skmultiflow.data import HyperplaneGenerator
import pandas as pd
import numpy as np

create = HyperplaneGenerator(random_state = 888, n_features= 10, noise_percentage = 0)
create.prepare_for_use()
X , Y = create.next_sample(10000)
data = pd.DataFrame(np.hstack((X, np.array([Y]).T)))
# Cast the last column to int
data = data.astype({10:int}) 

data.shape 
# output- (10000, 11)   

# Store it in csv                   
data.to_csv('data_stream.csv', index=False) 


# Applying Hoeffding Tree on the synthetic data stream

# Import the relevant libraries
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream
import pandas as pd
import numpy as np

# Load the synthetic data stream
dstream = FileStream('data_stream.csv')
dstream.prepare_for_use()

# Create the model instance
ht_class = HoeffdingTreeClassifier()

# perform prequential evaluation
evaluate1 = EvaluatePrequential(show_plot=False,
pretrain_size=400,
max_samples=10000,
metrics=['accuracy']
)
evaluate1.evaluate(stream=dstream, model=ht_class)

###################################################

# Hoeffding Adaptive tree
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.evaluation import EvaluateHoldout

# Simulate a sample data stream
ds = ConceptDriftStream(random_state=777, position=30000)
ds
# Output:
#ConceptDriftStream(alpha=0.0,
#                   drift_stream=AGRAWALGenerator(balance_classes=False,
#                                                 classification_function=2,
#                                                 perturbation=0.0,
#                                                 random_state=112),
#                   position=30000, random_state=777,
#                   stream=AGRAWALGenerator(balance_classes=False,
#                                           classification_function=0,
#                                           perturbation=0.0, random_state=112),
#                   width=1000)


# Instantiate the model object
model_hat = HoeffdingAdaptiveTreeClassifier()

# Prequential evaluation
eval1 = EvaluatePrequential(pretrain_size=400, max_samples=300000, batch_size=1,
                                n_wait=100, max_time=2000,
                                show_plot=False, metrics=['accuracy'])

eval1.evaluate(stream=ds, model=model_hat)

# Holdout evaluation
eval2 = EvaluateHoldout(max_samples=30000,
                            max_time=2000,
                            show_plot=False,
                            metrics=['accuracy'],
                            dynamic_test_set=True)

eval2.evaluate(stream=ds, model=model_hat)
###################################################




# Extremely Fast Decision Tree
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.evaluation import EvaluateHoldout

# Simulate a sample data stream
ds = ConceptDriftStream(random_state=777, position=30000)
ds
# Output:
#ConceptDriftStream(alpha=0.0,
#                   drift_stream=AGRAWALGenerator(balance_classes=False,
#                                                 classification_function=2,
#                                                 perturbation=0.0,
#                                                 random_state=112),
#                   position=30000, random_state=777,
#                   stream=AGRAWALGenerator(balance_classes=False,
#                                           classification_function=0,
#                                           perturbation=0.0, random_state=112),
#                   width=1000)

# Instantiate the model object
model_hat = ExtremelyFastDecisionTreeClassifier()

# Prequential evaluation
eval1 = EvaluatePrequential(pretrain_size=400, max_samples=300000, batch_size=1,
                                n_wait=100, max_time=2000,
                                show_plot=False, metrics=['accuracy'])

eval1.evaluate(stream=ds, model=model_hat)

# Holdout evaluation
eval2 = EvaluateHoldout(max_samples=30000,
                            max_time=2000,
                            show_plot=False,
                            metrics=['accuracy'],
                            dynamic_test_set=True)

eval2.evaluate(stream=ds, model=model_hat)
###################################################










from skmultiflow.data import HyperplaneGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream
import pandas as pd
import numpy as np
generate = HyperplaneGenerator ( random_state = 112, n_features= 8, noise_percentage = 0)
generate.prepare_for_use()
x , y = generate.next_sample(5000)
df = pd.DataFrame(np.hstack((x, np.array([y]).T)))  # all columns are set as float
df = df.astype({8:int})                             # Cast the last column to int
df.to_csv('file.csv', index=False)
stream = FileStream('file.csv')
stream.prepare_for_use()
ht = HoeffdingTree()
evaluator = EvaluatePrequential(show_plot=False,
pretrain_size=200,
max_samples=5000,
metrics=['accuracy']
)
evaluator.evaluate(stream=stream, model=ht)


# The first example demonstrates how to evaluate one model
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluateHoldout

# Set the stream
stream = SEAGenerator(random_state=1)

# Set the model
ht = HoeffdingTreeClassifier()

# Set the evaluator
evaluator = EvaluateHoldout(max_samples=100000,
                            max_time=100000,
                            show_plot=False,
                            metrics=['accuracy'],
                            dynamic_test_set=True)

# Run evaluation
evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])




### HT regressor
# Import the relevant libraries
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.data import RegressionGenerator
import pandas as pd
from skmultiflow.evaluation import EvaluatePrequential

# Setup a data stream
dstream = RegressionGenerator(n_features=9, n_samples=800,n_targets=1, random_state=456)
dstream
#RegressionGenerator(n_features=9, n_informative=10, n_samples=800, n_targets=1,
#                    random_state=456)

dstream.next_sample()
#(array([[ 0.72465838, -1.92979924, -0.02607907,  2.35603757, -0.37461529,
#         -0.38480019,  0.06603468, -2.1436878 ,  0.49182531]]),
# array([61.302191]))


# Instantiate the Hoeffding Tree Regressor object
htr = HoeffdingTreeRegressor()

# Prequential evaluation
eval1 = EvaluatePrequential(pretrain_size=400, max_samples=800, batch_size=1,
                                n_wait=100, max_time=2000,
                                show_plot=False, metrics=['mean_square_error', 'mean_absolute_error'])

eval1.evaluate(stream=dstream, model=htr)

#############################################################################################





### Hoeffding Adaptive Tree regressor
# Import the relevant libraries
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.data import RegressionGenerator
import pandas as pd
from skmultiflow.evaluation import EvaluatePrequential


# Setup a data stream
dstream = RegressionGenerator(n_features=9, n_samples=800,n_targets=1, random_state=456)
dstream
#RegressionGenerator(n_features=9, n_informative=10, n_samples=800, n_targets=1,
#                    random_state=456)

dstream.next_sample()
#(array([[ 0.72465838, -1.92979924, -0.02607907,  2.35603757, -0.37461529,
#         -0.38480019,  0.06603468, -2.1436878 ,  0.49182531]]),
# array([61.302191]))


# Instantiate the Hoeffding Adaptive Tree Regressor object
model_hatr  = HoeffdingAdaptiveTreeRegressor()

# Prequential evaluation
eval1 = EvaluatePrequential(pretrain_size=400, max_samples=800, batch_size=1,
                                n_wait=100, max_time=2000,
                                show_plot=False, metrics=['mean_square_error', 'mean_absolute_error'])

eval1.evaluate(stream=dstream, model=model_hatr )

#############################################################################################




# Applying KNN Classifier on the synthetic data stream
from skmultiflow.lazy import KNNClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream
import pandas as pd
import numpy as np


dstream = FileStream('data_stream.csv')
dstream.prepare_for_use()
knn_class = KNNClassifier(n_neighbors=10, max_window_size=1000)

# Prequential Evaluation
evaluate1 = EvaluatePrequential(show_plot=False,
pretrain_size=1000,
max_samples=10000,
metrics=['accuracy']
)
# Run the evaluation
evaluate1.evaluate(stream=dstream, model=knn_class)

###################################################

# Applying KNN ADWIN Classifier on the synthetic data stream
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.sea_generator import SEAGenerator


# Simulate the data stream
dstream = SEAGenerator(classification_function = 2, balance_classes = True, noise_percentage = 0.3, random_state = 333)

#Retrieve five samples
dstream.next_sample(5)
# Output:
#(array([[3.68721825, 0.48303666, 1.04530188],
#        [2.45403315, 8.73489354, 0.51611639],
#        [2.38740114, 2.03699194, 1.74533621],
#        [9.41738118, 4.66915281, 9.59978205],
#        [1.05404748, 0.42265956, 2.44130999]]), array([1, 0, 0, 1, 1]))

# Instatntiate the KNN ADWIN classifier method
adwin_knn_class = KNNADWINClassifier(n_neighbors=10, max_window_size=1000)

# Prequential Evaluation
evaluate1 = EvaluatePrequential(show_plot=False,
pretrain_size=1000,
max_samples=10000,
metrics=['accuracy']
)
# Run the evaluation
evaluate1.evaluate(stream=dstream, model=adwin_knn_class)

###################################################


# Applying SAM-KNN Classifier on the synthetic data stream
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.sea_generator import SEAGenerator


# Simulate the data stream
dstream = SEAGenerator(classification_function = 2, balance_classes = True, noise_percentage = 0.3, random_state = 333)

#Retrieve five samples
dstream.next_sample(5)

# Instatntiate the KNN ADWIN classifier method
sam_knn_class = SAMKNNClassifier(n_neighbors=10, weighting='distance', max_window_size=1000,stm_size_option='maxACCApprox', use_ltm=True)

# Prequential Evaluation
evaluate1 = EvaluatePrequential(show_plot=False,
pretrain_size=1000,
max_samples=10000,
metrics=['accuracy']
)
# Run the evaluation
evaluate1.evaluate(stream=dstream, model=sam_knn_class)

###################################################



### KNN regressor
# Import the relevant libraries
from skmultiflow.lazy import KNNRegressor
from skmultiflow.data import RegressionGenerator
import pandas as pd
from skmultiflow.evaluation import EvaluatePrequential

# Setup a data stream
dstream = RegressionGenerator(n_features=9, n_samples=800,n_targets=1, random_state=456)
dstream
#RegressionGenerator(n_features=9, n_informative=10, n_samples=800, n_targets=1,
#                    random_state=456)

dstream.next_sample()
#(array([[ 0.72465838, -1.92979924, -0.02607907,  2.35603757, -0.37461529,
#         -0.38480019,  0.06603468, -2.1436878 ,  0.49182531]]),
# array([61.302191]))


# Instantiate the KNN Regressor object
knn_reg = KNNRegressor()

# Prequential evaluation
eval1 = EvaluatePrequential(pretrain_size=400, max_samples=800, batch_size=1,
                                n_wait=100, max_time=2000,
                                show_plot=False, metrics=['mean_square_error', 'mean_absolute_error'])

eval1.evaluate(stream=dstream, model=knn_reg)

#############################################################################################




# Applying Adaptive Random Forest Classifier on a synthetic data stream
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.sea_generator import SEAGenerator


# Simulate the data stream
dstream = SEAGenerator(classification_function = 2, balance_classes = True, noise_percentage = 0.3, random_state = 333)

# Instatntiate the KNN ADWIN classifier method
ARF_class = AdaptiveRandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=333)

# Prequential Evaluation
evaluate1 = EvaluatePrequential(show_plot=False,
pretrain_size=1000,
max_samples=10000,
metrics=['accuracy']
)
# Run the evaluation
evaluate1.evaluate(stream=dstream, model=ARF_class)

###################################################



### ARF regressor
# Import the relevant libraries
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.data import RegressionGenerator
from skmultiflow.evaluation import EvaluatePrequential

# Setup a data stream
dstream = RegressionGenerator(n_features=9, n_samples=800,n_targets=1, random_state=456)


# Instantiate the ARF Regressor object
ARF_reg = AdaptiveRandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=333)

# Prequential evaluation
eval1 = EvaluatePrequential(pretrain_size=400, max_samples=800, batch_size=1,
                                n_wait=100, max_time=2000,
                                show_plot=False, metrics=['mean_square_error', 'mean_absolute_error'])

eval1.evaluate(stream=dstream, model=ARF_reg)

#############################################################################################





# Applying Oza Bagging Classifier on a synthetic data stream
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.sea_generator import SEAGenerator


# Simulate the data stream
dstream = SEAGenerator(classification_function = 2, balance_classes = True, noise_percentage = 0.3, random_state = 333)

# Instantiate the Oza Bagging classifier method with KNN ADWIN classifier as the base model
oza_class = OzaBaggingClassifier(base_estimator=KNNADWINClassifier(n_neighbors=10, max_window_size=1000), n_estimators=6, random_state = 333)

# Prequential Evaluation
evaluate1 = EvaluatePrequential(show_plot=False,
pretrain_size=1000,
max_samples=10000,
metrics=['accuracy']
)
# Run the evaluation
evaluate1.evaluate(stream=dstream, model=oza_class)

###################################################



# Applying Leveraging Bagging Classifier on a synthetic data stream
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.sea_generator import SEAGenerator


# Simulate the data stream
dstream = SEAGenerator(classification_function = 2, balance_classes = True, noise_percentage = 0.3, random_state = 333)

# Instantiate the Leveraging Bagging classifier method with KNN ADWIN classifier as the base model
leverage_class = LeveragingBaggingClassifier(base_estimator=KNNADWINClassifier(n_neighbors=10, max_window_size=1000), n_estimators=6, random_state = 333)

# Prequential Evaluation
evaluate1 = EvaluatePrequential(show_plot=False,
pretrain_size=1000,
max_samples=10000,
metrics=['accuracy']
)
# Run the evaluation
evaluate1.evaluate(stream=dstream, model=leverage_class)

###################################################



# Applying Online Boosting Classifier on a synthetic data stream
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.sea_generator import SEAGenerator


# Simulate the data stream
dstream = SEAGenerator(classification_function = 2, balance_classes = True, noise_percentage = 0.3, random_state = 333)

# Instantiate the Online Boosting Classifier method
boost_class = OnlineBoostingClassifier(random_state = 333)

# Prequential Evaluation
evaluate1 = EvaluatePrequential(show_plot=False,
pretrain_size=1000,
max_samples=10000,
metrics=['accuracy']
)
# Run the evaluation
evaluate1.evaluate(stream=dstream, model=boost_class)

###################################################










