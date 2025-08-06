%%%%%%%%%%% Model Evaluation %%%%%%%%%%%
% The model is defined by its parameters W and b are saved in folder:
%     ../Model_Parameter
%
% Choose model = 'any_name'.
%
% Step 3 can take a long time.
%
% The last part/section of Step 4 can be run anytime once this script
%    is executed.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Step 1: Create two folders: ../Evaluation_Data/Model_Performance
%                                   ../Evaluation_Data/Model_Accuracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Step 2: Initialize storage for performance result, only onetime.
%
model='S_h1_m1'; %'ANN' % for no distinction
nmb_of_hidden_layers=1;

nmb_of_data_batches=10;
nmb_of_modules=40;
nmb_of_module_subsets=2;
nmb_of_feature_processes=6;
nmb_of_feature_sel=3;
nmb_of_features=17;

for imgnt1kdataset=1:nmb_of_data_batches
    reportname1 = sprintf('/work/mathbiology/lheath2/data/imagenet1k/mat/train_data_batch_%d.mat', imgnt1kdataset);
    data_load=load(reportname1,'labels');
    nmb_of_images=length(data_load.labels);
    aa=mod(nmb_of_features,nmb_of_feature_sel);
    for feature=1:nmb_of_feature_processes
        if feature<nmb_of_feature_processes
            nmb_of_sel_features=nmb_of_feature_sel;
        else
            if aa==0
                nmb_of_sel_features=nmb_of_feature_sel;
            else
                nmb_of_sel_features=aa;
            end
        end
        training_performance=zeros(2,nmb_of_modules,nmb_of_module_subsets,nmb_of_sel_features,nmb_of_images);
        image0_perf=1;
        reportname1 = sprintf('../Evaluation_Data/Model_Performance/training_performance_batch_%d_feature_%d_performance_%s_var.mat',...
            imgnt1kdataset, feature, model);
        str=struct('training_performance',training_performance,'image0_perf',image0_perf);
        save(reportname1,"-fromstruct",str);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%% Step 3: Obtain the output of the model and the input, i.e., the training images
%                 This part can be repeadely run until it is completed
patch=0;
nmb_of_labs_per_module=25;
cross_entropy=1;

param.model=model;
param.nmb_of_hidden_layers=nmb_of_hidden_layers;

param.image_size=[64,64];
param.downsizing=2;
param.x_trim=1;
param.y_trim=1;
param.compute_decimal_place=4;
param.dwnsz_on=1;

param.patch=patch;
param.nmb_of_modules=nmb_of_modules;
param.nmb_of_module_subsets=nmb_of_module_subsets;

param.cross_entropy=cross_entropy;
param.nmb_of_labs_per_module=nmb_of_labs_per_module;
param.channel_sel=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17];
nmb_of_features=length(param.channel_sel);
nmb_of_batches=100;

param.channels_names=[];
param.nmb_of_colors=0;
param.feature_RGB=[];

parfor imgnt1kdataset=1:nmb_of_data_batches
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
    pm=param;
    %%%%%%%%%%% Model's input, i.e., the training images
    reportname1 = sprintf('/work/mathbiology/lheath2/data/imagenet1k/mat/train_data_batch_%d.mat', imgnt1kdataset);
    data_load=load(reportname1,'data');
    [nmb_of_images,~]=size(data_load.data);
    for feature=1:nmb_of_feature_processes
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        reportname1 = sprintf('../Evaluation_Data/Model_Performance/training_performance_batch_%d_feature_%d_performance_%s_var.mat',...
            imgnt1kdataset, feature, model);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        saved_load=load(reportname1);
        training_performance=saved_load.training_performance;
        %     training_performance=zeros(2,nmb_of_modules,nmb_of_module_subsets,nmb_of_colors,nmb_of_images);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        image0_perf=saved_load.image0_perf;
        if feature<nmb_of_feature_processes
            feature_seq=(1:nmb_of_feature_sel)+(feature-1)*nmb_of_feature_sel;
        else
            feature_seq=(1+(feature-1)*nmb_of_feature_sel):nmb_of_features;
        end
        pm.channels_names=channels_names(feature_seq);
        pm.nmb_of_colors=length(pm.channels_names);
        pm.feature_RGB=feature_RGB(feature_seq,:);
        if image0_perf<nmb_of_images
            for batch_nmb=1:nmb_of_batches
                image_batch=fun_proc_batch(nmb_of_images,nmb_of_batches,batch_nmb,image0_perf);
                if ~isempty(image_batch)
                    inputimage=data_load.data(image_batch,:);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    output=fun_transform_data_rgbfeatures(inputimage,pm);
                    image=output.transformed_image;
                    batch_training_performance=fun_ANN_Model(image,pm);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    training_performance(:,:,:,:,image_batch)=batch_training_performance;
                    image0_perf=image_batch(end);
                end
                [imgnt1kdataset, feature, batch_nmb]
                out=fun_save_model_performance(model,imgnt1kdataset,feature,...
                    training_performance,image0_perf);
            end
        end
        fprintf('Model %s imgnt1kdataset = %d, pf = %d\n',model,imgnt1kdataset,feature);
    end
end
%%%%%%%%%%%%%%%%%
% Combine the performances of all features for the aggregated model
%
for imgnt1kdataset=1:nmb_of_data_batches
    aa=[];
    for feature=1:nmb_of_feature_processes
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        reportname1 = sprintf('../Evaluation_Data/Model_Performance/training_performance_batch_%d_feature_%d_performance_%s_var.mat',...
            imgnt1kdataset, feature, model);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        saved_load=load(reportname1);
        bb=saved_load.training_performance;
        aa=cat(4,aa,bb);
    end
    training_performance=aa;
    reportname1 = sprintf('../Evaluation_Data/Model_Accuracy/training_data_batch_%d_feature_module_performance_%s_var.mat',...
        imgnt1kdataset, model);
    save(reportname1,'training_performance','-v7.3');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%% Step 4: Obtain the accuracy of the model
%
%%%%%%% Obtain the output of the model, i.e., the label prediction
%             of the input images
nmb_of_batches=12;
%%%%%%%%%%%%%% Key Input %%%%%%%%%%%%%
% five different feature-aggregates:
feature_sel=[1 2 3 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN
    1 2 3 4 5 6 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN
    1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN
    1 2 3 4 5 6 7 8 9 10 11 12 NaN NaN NaN NaN NaN
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 NaN NaN
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17];
nmb_of_ft_models=length(feature_sel(:,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for fs=1:nmb_of_ft_models
    aa=feature_sel(fs,:);
    idx=~isnan(aa);
    param.channel_sel=aa(idx);
    for imgnt1kdataset=1:nmb_of_data_batches
        reportname1 = sprintf('/work/mathbiology/lheath2/data/imagenet1k/mat/train_data_batch_%d.mat', imgnt1kdataset);
        data_load=load(reportname1,'labels');
        nmb_of_images=length(data_load.labels);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        reportname1 = sprintf('../Evaluation_Data/Model_Accuracy/training_data_batch_%d_feature_module_performance_%s_var.mat',...
            imgnt1kdataset, model);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        saved_load=load(reportname1);
        training_performance=saved_load.training_performance;
        classification_data=zeros(nmb_of_images,4);
        for batch_nmb=1:nmb_of_batches
            image_batch=fun_proc_batch_update(nmb_of_images,nmb_of_batches,batch_nmb);
            if ~isempty(image_batch)
                true_label=double(data_load.labels(image_batch));
                training_performance_temp=training_performance(:,:,:,:,image_batch);
                [predicted_label,likelyhood]=fun_majority_rule_prediction(training_performance_temp,param);
                top_1_majority=length(param.channel_sel);
                classification_data(image_batch,:)=[true_label;predicted_label;likelyhood;0*likelyhood+top_1_majority]';
            end
        end
        reportname1 = sprintf('../Evaluation_Data/Model_Accuracy/training_data_batch_%d_feature_module_performance_%s_var.mat', imgnt1kdataset,model);
        aa=sprintf('classification_data_%d',fs);
        str=struct(aa,classification_data);
        save(reportname1,"-fromstruct",str,'-append');
    end
    fprintf('Model %s feature = %d\n',model, fs);
end
%%
%%%%%%%%% Find the accuracy rates for featured models
% This part can be run anytime once the main scripts above are executed. 
%
% % nmb_of_ft_models=5;
% % nmb_of_image_set=zeros(1,10);
% % pr_set=zeros(1,10);
% % model_accuracy_comparison=zeros(1,nmb_of_ft_models);
% % for fs=1:nmb_of_ft_models
% %     for imgnt1kdataset=1:10
% %         reportname1 = sprintf('../Evaluation_Data/Model_Accuracy_1/training_data_batch_%d_feature_module_performance_%s_var.mat', imgnt1kdataset,model);
% %         aa=sprintf('classification_data_%d',fs);
% %         bb=load(reportname1,aa);
% %         c_data=bb.(aa);
% %         true_lab=c_data(:,1);
% %         pred_lab=c_data(:,2);
% %         nmb_of_images=length(true_lab);
% %         nmb_of_image_set(imgnt1kdataset)=nmb_of_images;
% %         idx=(abs(true_lab-pred_lab)==0);
% %         aa=sum(1*idx);
% %         pr=aa/nmb_of_images*100;
% %         pr_set(imgnt1kdataset)=pr;
% %     end
% %     model_accuracy_comparison(fs)=nmb_of_image_set*pr_set'/sum(nmb_of_image_set);
% % end
% % %%%% This shows the accuracy rates for the five featured models:
% % model_accuracy_comparison
%%