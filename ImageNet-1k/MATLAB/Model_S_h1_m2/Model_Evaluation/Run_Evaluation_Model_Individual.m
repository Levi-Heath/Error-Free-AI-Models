% clear all
model='S_h1_m2';
nmb_of_hidden_layers=1;
% model_type='GDT'; % or "SGD'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Choose any combination to run  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgnt1kdataset=2; % in [1,10]
image_batch=1:100; % randomly choose in [1, nmb_of_images]

reportname1 = sprintf('/work/mathbiology/lheath2/data/imagenet1k/mat/train_data_batch_%d.mat', imgnt1kdataset);
% or choose validation data
% reportname1 = sprintf('/work/mathbiology/lheath2/data/imagenet1k/mat/val_data.mat');
% image_batch=1:100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Input Model Parameters
nmb_of_modules=40;
nmb_of_module_subsets=2;
nmb_of_labs_per_module=25;
cross_entropy=1;
channels_names={'R','G','B','RGg1','RBg1','GBg1','RGg2','RBg2','GBg2','RB','RG','GB','eRGB','BW','X','Y','Z'};
feature_RGB=[1 0 0
    0      1      0
    0      0      1
    0.618  0.382  0
    0.618  0      0.382
    0      0.618  0.382
    0.382  0.618  0
    0.382  0      0.618
    0      0.382  0.618
    0.5    0.5    0
    0.5    0      0.5
    0      0.5    0.5
    1/3    1/3    1/3    
    0.299  0.587  0.114
    0.4125 0.3576  0.1804
    0.2126 0.7152  0.0722
    0.0193 0.1192  0.9502]; 
nmb_of_colors=length(channels_names);

param.model=model;
param.nmb_of_hidden_layers=nmb_of_hidden_layers;
param.image_size=[64,64];
param.nmb_of_colors=nmb_of_colors;
param.downsizing=2;
param.x_trim=1;
param.y_trim=1;
param.compute_decimal_place=4;
param.dwnsz_on=1;

param.patch=0;
param.nmb_of_modules=nmb_of_modules;
param.nmb_of_module_subsets=nmb_of_module_subsets;
param.channels_names=channels_names;
param.cross_entropy=cross_entropy;
param.nmb_of_labs_per_module=nmb_of_labs_per_module;
param.feature_RGB=feature_RGB;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
data_load=load(reportname1,'data','labels');

input_image=data_load.data(image_batch,:); % Input images to the trained model
true_label=double(data_load.labels(image_batch)); 
t_img=fun_transform_data_rgbfeatures(input_image,param); % Transform the input
image=t_img.transformed_image;
mpout=fun_ANN_Model(image,param); % Output of the model's ANN

toc
%%
% Choose any combination for 'param.channel_sel' to build  
%   the ala-carte, feature-aggregated model's accuracy rate
%

param.channel_sel=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17];
% param.channel_sel=[1,2,3,4,5,6,7,8,9];
% param.channel_sel=[1,2,3];

[pred_label,likelyhood]=fun_majority_rule_prediction(mpout,param); % Output of the model

nmb_of_images=length(true_label);
idx=(abs(true_label-pred_label)==0);
aa=sum(1*idx);
pr=aa/nmb_of_images*100 % Possitive Rate, or accuracy of the model
%%
% Top-1 rate
bb=likelyhood(idx);
cc=length(bb);
idx1=(bb==length(param.channel_sel));
aa=sum(idx1*1);
top_1=aa/cc*100