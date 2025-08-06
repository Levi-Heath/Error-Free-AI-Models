% clear all

proto_model={'T_h1_m1';'S_h1_m1';'T_h1_m2';'S_h1_m2';'T_h2_m1';'S_h2_m1'};
nmb_of_proto_models=length(proto_model);
featured_model={'model_1';'model_2';'model_3';'model_4';'model_5';'model_6'};
[nmb_of_ft_models,~]=size(featured_model);

nmb_of_lab=1000;
nmb_of_batches=10;
nmb_of_image_set=zeros(1,10);
pr_set=zeros(1,10);
model_accuracy_comparison=zeros(1,nmb_of_ft_models);
pm.nmb_of_lab=nmb_of_lab;
edges=1:1:(nmb_of_lab+1);
% top_n_labels=10;
confusion_matrixes=zeros(nmb_of_lab,nmb_of_lab,nmb_of_batches,nmb_of_ft_models,nmb_of_proto_models);
model_c_matrixes=zeros(nmb_of_lab,nmb_of_lab,nmb_of_ft_models,nmb_of_proto_models);

for ii=1:nmb_of_proto_models
    md=char(proto_model(ii));
%     confusion_matrixes=zeros(nmb_of_lab,nmb_of_lab,nmb_of_batches,nmb_of_ft_models);
%     model_c_matrixes=zeros(nmb_of_lab,nmb_of_lab,nmb_of_ft_models);
    top_100_pr_by_batch=zeros(nmb_of_batches,nmb_of_ft_models); % 100 postive rate
    top_100_pr_by_model=zeros(1,nmb_of_ft_models); % 100 postive rate
    batch_nmbs=zeros(1,nmb_of_batches);
    for fm=1:nmb_of_ft_models
        top_100_accnt=[];
        c_matrix=zeros(nmb_of_lab,nmb_of_lab);
        for imgnt1kdataset=1:nmb_of_batches
            reportname1 = sprintf('Model_%s/Evaluation_Data/Model_Accuracy/training_data_batch_%d_feature_module_performance_%s_var.mat', md,imgnt1kdataset,md);
            aa=sprintf('classification_data_%d',fm);
            bb=load(reportname1,aa);
            c_data=bb.(aa);
            true_lab=c_data(:,1);
            pred_lab=c_data(:,2);
            nmb_of_data=length(true_lab);
            c_mtx_output=fun_confusion_matrix(true_lab,pred_lab,pm);
            bb=c_mtx_output.conf_matrix;
            confusion_matrixes(:,:,imgnt1kdataset,fm,ii)=bb;
            c_matrix=c_matrix+nmb_of_data*bb;
            top_100_pr_by_batch(imgnt1kdataset,fm)=length(c_mtx_output.lab_100);
            top_100_accnt=[top_100_accnt,c_mtx_output.lab_100];
            batch_nmbs(imgnt1kdataset)=nmb_of_data;
        end
        model_c_matrixes(:,:,fm,ii)=c_matrix/sum(batch_nmbs);
        histN = histcounts(top_100_accnt,edges);
        idx=(histN==nmb_of_batches);
        top_100_pr_by_model(fm)=sum(1*idx);
    end
%     mean(top_100_pr_by_batch,1) % average 100-rate by data batch
%     top_100_pr_by_model
    %%%%%%%
    assignin('base',md, [mean(top_100_pr_by_batch,1);top_100_pr_by_model]')
end
%%
table(featured_model,T_h1_m1,T_h1_m2,T_h2_m1)

%% 
% Find the labels that are perfectly classified, by model
%
ii=1;
fm=6;
lab=1:1000;
A=model_c_matrixes(:,:,fm,ii);
dd=diag(A,0);
idx=(dd==100);
lab_100=lab(idx)
nmb_of_lab_100=sum(1*idx)

%% 
% Find the labels that are perfectly classified, by batch and model
%
ii=1;
fm=6;
imgnt1kdataset=1;
lab=1:1000;
A=confusion_matrixes(:,:,imgnt1kdataset,fm,ii);
dd=diag(A,0);
idx=(dd==100);
lab_100=lab(idx)
nmb_of_lab_100=sum(1*idx)


