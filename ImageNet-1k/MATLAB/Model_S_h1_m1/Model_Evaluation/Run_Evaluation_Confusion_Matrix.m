%clear all
%%%%% Find positive rate by confusion matrixes
%
model='S_h1_m1'; % or 'ANN'
nmb_of_lab=1000;
nmb_of_batches=10;
nmb_of_ft_models=6;
nmb_of_image_set=zeros(1,10);
pr_set=zeros(1,10);
model_accuracy_comparison=zeros(1,nmb_of_ft_models);
pm.nmb_of_lab=nmb_of_lab;
edges=1:1:(nmb_of_lab+1);
top_n_labels=10;

confusion_matrixes=zeros(nmb_of_lab,nmb_of_lab,nmb_of_batches,nmb_of_ft_models);
model_c_matrixes=zeros(nmb_of_lab,nmb_of_lab,nmb_of_ft_models);
top_100_pr_by_batch=zeros(nmb_of_batches,nmb_of_ft_models); % 100 postive rate
top_100_pr_by_model=zeros(1,nmb_of_ft_models); % 100 postive rate
batch_nmbs=zeros(1,nmb_of_batches);
for fs=1:nmb_of_ft_models
    top_100_accnt=[];
    c_matrix=zeros(nmb_of_lab,nmb_of_lab);
    for imgnt1kdataset=1:nmb_of_batches
        reportname1 = sprintf('../Evaluation_Data/Model_Accuracy/training_data_batch_%d_feature_module_performance_%s_var.mat', imgnt1kdataset,model);
        aa=sprintf('classification_data_%d',fs);
        bb=load(reportname1,aa);
        c_data=bb.(aa);
        true_lab=c_data(:,1);
        pred_lab=c_data(:,2);
        nmb_of_data=length(true_lab);
        c_mtx_output=fun_confusion_matrix(true_lab,pred_lab,pm);
        bb=c_mtx_output.conf_matrix;
        confusion_matrixes(:,:,imgnt1kdataset,fs)=bb;
        c_matrix=c_matrix+nmb_of_data*bb;
        top_100_pr_by_batch(imgnt1kdataset,fs)=length(c_mtx_output.lab_100);
        top_100_accnt=[top_100_accnt,c_mtx_output.lab_100];
        batch_nmbs(imgnt1kdataset)=nmb_of_data;
    end
    model_c_matrixes(:,:,fs)=c_matrix/sum(batch_nmbs);
    histN = histcounts(top_100_accnt,edges);
    idx=(histN==nmb_of_batches);
    top_100_pr_by_model(fs)=sum(1*idx);
end
mean(top_100_pr_by_batch,1) % average 100-rate by data batch
top_100_pr_by_model % 100-rate for the feature-aggreated models. 
%%
fs=6; % 1 to 6
[rates,labs]=fun_top_n_label_rate(model_c_matrixes(:,:,fs),top_n_labels);
[rates,labs]'

imgnt1kdataset=1; %1 to 10 by data batch
[rates,labs]=fun_top_n_label_rate(confusion_matrixes(:,:,imgnt1kdataset,fs),259);
[rates,labs]'
%%
