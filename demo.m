clear;
clc;

addpath(genpath('./'));

% load data
load('Flags.mat');

% define parameters
params.alpha = 0.3;
params.beta = 0.7;

params.lambda1 = 0.01;
params.lambda2 = 100;
params.lambda3 = 10;
params.k = 10;
params.max_iter = 200;

% define labeling ratios
params.rate = 0.1;

[result, std_result] = Train_SMLE(features, labels, params.rate, params);



