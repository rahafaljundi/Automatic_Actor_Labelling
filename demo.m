%demo file on the first episode from Breaking Bad
%load the structure .mat file that contains the parameters
load('bb1_opts.mat')
%for the tracks features. Please go to the http://www.jaychakravarty.com/whos-that-actor/ download the track-features zip file for the first episode of Breaking Bad, unzip it put it in the demo folder
cd .. 
%do the job
actor_labelling(opts);
