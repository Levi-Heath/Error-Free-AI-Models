% clear all

channels_names={'R','G','B','RGg1','RBg1','GBg1','RGg2','RBg2','GBg2','RB','RG','GB','eRGB','BW','X','Y','Z'};
patch=0;
module=9;%4;%38;
subset=2;
color=4;
nmb_of_labs_per_module=25;
nmb_of_labels=nmb_of_labs_per_module;
maxlngth=3000;

reportname1 = sprintf('Data_Transformation/Transformed_IN1k_Data/Transformed_Data_for_SGD/train_data_patch_%d_module_%d_subset_%d_for_%d_labels_per_module.mat', ...
    patch,module,subset,nmb_of_labs_per_module);
data_load=load(reportname1);
vect_image=data_load.data(:,1:maxlngth,color);
true_label=data_load.labels(1:maxlngth);

model='T_h2_m1';
reportname1 = sprintf(['Model_%s/Model_Parameter/Trained_Parameter_patch_%d_module_%d_subset_%d_ch_%s.mat'],...
    model,patch, module, subset, char(channels_names(color)));
%      str=struct('W', W, 'b', b);
temp_load=load(reportname1);
Wt=temp_load.W;
bt=temp_load.b;
reportname1 = sprintf('Model_%s/Training_Evaluation/%s_1_performance.mat',model,model);
load(reportname1)
tpr=pstvrt_model(color,module,subset);

model='S_h2_m1';
reportname1 = sprintf('Model_%s/Model_Parameter/Trained_Parameter_patch_%d_module_%d_subset_%d_ch_%s.mat',...
    model, patch, module, subset, char(channels_names(color)));
%      str=struct('W', W, 'b', b);
temp_load=load(reportname1);
Ws=temp_load.W;
bs=temp_load.b;
reportname1 = sprintf('Model_%s/Training_Evaluation/%s_1_performance.mat',model,model);
load(reportname1)
spr=pstvrt_model(color,module,subset);

fun_spoke_plot(vect_image,true_label,Ws,bs,spr,Wt,bt,tpr,nmb_of_labels)
