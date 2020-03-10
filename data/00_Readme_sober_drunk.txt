***********************************************************

                SOBER  -  DRUNK   DATA BASE

(Started September 2012    ----   Completed April 2013)

Electronics Laboratory - Physics Department
University of Patras  -  Greece

By Georgia Koukiou and Vassilis Anastassopoulos
gkoukiou@upatras.gr      vassilis@upatras.gr 

Every body can use this data base testing and publishing 
experimental results, provided that she/he will refer to 
relevant publications of the creators of this database. 

***********************************************************

It contains data for 41 persons.
For each person there are 16 different acquisitions
Each acquisition corresponds to each file of the data base
Each file contains 50 sequential frames of the same object 
   acquired every 100msec, i.e. in 5 sec all 50 frames.


The following MATLAB program is provided for reading each 
   separate file

***********************************************************

clc;
clear all;
close all;

c=zeros(128,160);
for i=1:50
    a(i).data=imread('filename.tif',i);
    xm(i).data=min(min(a(i).data)); 
    a(i).data=(a(i).data-xm(i).data);
    for j=1:128
        for k=1:160
            c(j,k)=c(j,k)+a(i).data(j,k);
        end
    end
end

***********************************************************

Infrared image acquisition

Time 20:50
Firstly, for each sober person, which is in calm condition,
an infrared sequence (1) is obtained from his Face (f), from 
his Eyes (e), from his Ear-profile (r), and his Hand (h).

21:00 - 22:00
After that four glasses of wine are drunk in one hour time.

22:20
A new sequence (2) of infrared images is acquired 
    (f), (e), (r), (h).

22:50
A new sequence (3) of infrared images is acquired 
    (f), (e), (r), (h).

23:20
A new sequence (4) of infrared images is acquired 
    (f), (e), (r), (h).

We had the people in groups of 4 or 5 or 6 persons. 
For two groups (10 people) we asked the police to carry 
out measurement with alcohol-meter. 
We have the correspondence of these measurements 
with the persons. 

***********************************************************

Naming the files
serialnumber_personfirstname_acquisitionsequence_imagecontent
_sex_age_weight__alcoholmeter

The two last measurements were obtained from few persons

***********************************************************
