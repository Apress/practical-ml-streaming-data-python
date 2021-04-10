#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on sun July 19 19:22:37 2020

@author: Dr. Sayan Putatunda
"""

""" Chapter 2 Codes """

import os
os.getcwd()  # to see the current path of working directory
os.chdir('./Python codes') 


### ADWIN code
# Import the relevant libraries
import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN

# Call the ADWIN object
A = ADWIN(delta=0.002)

# set seed for reproducibility
np.random.seed(123)

# Simulate a data stream of size 1000 from a Standard normal distribution
stream = np.random.randn(1000)

stream[:10]
# Output: 
#array([-1.0856306 ,  0.99734545,  0.2829785 , -1.50629471, -0.57860025,
#        1.65143654, -2.42667924, -0.42891263,  1.26593626, -0.8667404 ])

# Data concept are changed from index 599 to 999
for j in range(599, 1000):
    stream[j] = np.random.randint(5, high=9)
    
# Stream elements are added to ADWIN and checking whether drift occured
for j in range(1000):
     A.add_element(stream[j])
     if A.detected_change():
         print('Concept Drift detected in data: ' + str(stream[j]) + ' - at index: ' + str(j))
### Output:
#Concept Drift detected in data: 8.0 - at index: 607
#Concept Drift detected in data: 5.0 - at index: 639
#Concept Drift detected in data: 6.0 - at index: 671

########
         
         
### DDM code
import numpy as np
from skmultiflow.drift_detection import DDM

# call the DDM object
d2m = DDM()

# set seed for reproducibility
np.random.seed(123)

# Simulate a data stream of size 1000 from a Standard normal distribution
stream = np.random.randn(1000)

stream[:10]
## Output- 
#array([-1.0856306 ,  0.99734545,  0.2829785 , -1.50629471, -0.57860025,
#        1.65143654, -2.42667924, -0.42891263,  1.26593626, -0.8667404 ])

# Data concept are changed from index 299 to 600
for j in range(299, 600):
    stream[j] = np.random.randint(5, high=9)

# Stream elements are added to DDM and checking whether drift occured
for j in range(1000):
    d2m.add_element(stream[j])
    if d2m.detected_change():
        print('Concept drift detected in data: ' + str(stream[j]) + ' - at index: ' + str(j))
    if d2m.detected_warning_zone():
         print('Warning detected in data: ' + str(stream[j]) + ' - at index: ' + str(j))

### Output:
#Concept drift detected in data: 1.0693159694243486 - at index: 55
#Concept drift detected in data: 2.0871133595881854 - at index: 88
#Concept drift detected in data: 0.8123413299768204 - at index: 126
#Warning detected in data: 1.3772574828673068 - at index: 158
#Warning detected in data: -0.1431759743261871 - at index: 159
#Warning detected in data: 0.02031599823462459 - at index: 160
#Warning detected in data: -0.19396387055266243 - at index: 161
#Warning detected in data: 0.13402679274666512 - at index: 162
#Warning detected in data: 0.7044740740436035 - at index: 163
#Concept drift detected in data: 0.6656534379123312 - at index: 164
#Concept drift detected in data: 8.0 - at index: 302    

    

    
## HDDM_A code
import numpy as np
from skmultiflow.drift_detection.hddm_a import HDDM_A

# Initialize the HDDM_A object
HA = HDDM_A()

# set seed for reproducibility
np.random.seed(123)

# Simulate a data stream of size 1000 from a binomial distribution
# here, n= number of trials, p= probability of each trial
n, p = 10, 0.6 
stream = np.random.binomial(n, p, 1000)


stream[:10]
# Output- array([5, 7, 7, 6, 5, 6, 3, 5, 6, 6])

# Data concept are changed from index 299 to 500
for j in range(299, 500):
    stream[j] = np.random.randint(5, high=9)

# Stream elements are added to DDM and checking whether drift occured
for j in range(1000):
    HA.add_element(stream[j])
    if HA.detected_change():
        print('Concept drift detected in data: ' + str(stream[j]) + ' - at index: ' + str(j))
    if HA.detected_warning_zone():
         print('Warning detected in data: ' + str(stream[j]) + ' - at index: ' + str(j))
         
         
         
         
         
## HDDM_W code
import numpy as np
from skmultiflow.drift_detection.hddm_w import HDDM_W

# Initialize the HDDM_W object
HW = HDDM_W()

# set seed for reproducibility
np.random.seed(123)

# Simulate a data stream of size 1000 from a binomial distribution
# here, n= number of trials, p= probability of each trial
n, p = 10, 0.6  
stream = np.random.binomial(n, p, 1000)


stream[:10]
# Output- array([5, 7, 7, 6, 5, 6, 3, 5, 6, 6])

# Data concept are changed from index 299 to 500
for j in range(299, 500):
    stream[j] = np.random.randint(5, high=9)

# Stream elements are added to DDM and checking whether drift occured
for j in range(1000):
    HW.add_element(stream[j])
    if HW.detected_change():
        print('Concept drift detected in data: ' + str(stream[j]) + ' - at index: ' + str(j))
    if HW.detected_warning_zone():
         print('Warning detected in data: ' + str(stream[j]) + ' - at index: ' + str(j)) 






# page hinkley test
import numpy as np
from skmultiflow.drift_detection import PageHinkley

# Initialize the PageHinkley object
ph = PageHinkley()

# set seed for reproducibility
np.random.seed(123)


# Simulate a data stream of size 1000 from a normal distribution
# with mean=0 and standard deviation=0.1
stream = np.random.normal(0, 0.1, 1000)

# Data concept are changed from index 299 to 799
for j in range(299, 800):
    stream[j] = np.random.randint(5, high=9)

# Adding stream elements to the PageHinkley drift detector and verifying if drift occurred
for j in range(1000):
    ph.add_element(stream[j])
    if ph.detected_change():
        print('Concept drift detected in data: ' + str(stream[j]) + ' - at index: ' + str(j))

        







