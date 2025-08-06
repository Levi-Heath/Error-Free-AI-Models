%%%%%%%% Find the accuracy rates for featured models
% This part can be run anytime once the main scripts above are executed. 
%
model='S_h1_m1';% or 'ANN'
nmb_of_ft_models=6;
nmb_of_image_set=zeros(1,10);
pr_set=zeros(1,10);
top_1_set=zeros(1,10);
model_accuracy_comparison=zeros(2,nmb_of_ft_models);
for fs=1:nmb_of_ft_models
    for imgnt1kdataset=1:10
        reportname1 = sprintf('../Evaluation_Data/Model_Accuracy/training_data_batch_%d_feature_module_performance_%s_var.mat',...
            imgnt1kdataset, model);
        aa=sprintf('classification_data_%d',fs);
        bb=load(reportname1,aa);
        c_data=bb.(aa);
        true_lab=c_data(:,1);
        pred_lab=c_data(:,2);
        likelyhood=c_data(:,3);
        top_1_majority=c_data(1,3);
        nmb_of_images=length(true_lab);
        nmb_of_image_set(imgnt1kdataset)=nmb_of_images;
        idx=(abs(true_lab-pred_lab)==0);
        aa=sum(1*idx);
        pr=aa/nmb_of_images*100;
        pr_set(imgnt1kdataset)=pr;
        %%%%%%%% Top-1 rate %%%%%%%%%%%
        bb=likelyhood(idx);
        cc=length(bb);
        idx1=(bb==top_1_majority);
        aa=sum(idx1*1);
        top_1=aa/cc*100;
        top_1_set(imgnt1kdataset)=top_1;
    end
    model_accuracy_comparison(1,fs)=nmb_of_image_set*pr_set'/sum(nmb_of_image_set);
    model_accuracy_comparison(2,fs)=nmb_of_image_set*top_1_set'/sum(nmb_of_image_set);
end
%%%% This shows the accuracy rates for the multiple featured models
%               together with the Top-1 rate:
model_accuracy_comparison