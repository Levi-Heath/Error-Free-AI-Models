% clear all

proto_model={'T_h1_m1';'S_h1_m1';'T_h1_m2';'S_h1_m2';'T_h2_m1';'S_h2_m1'};
nmb_of_proto_models=length(proto_model);
featured_model={'model_1';'model_2';'model_3';'model_4';'model_5';'model_6'};
[nmb_of_ft_models,~]=size(featured_model);
for ii=1:nmb_of_proto_models
    md=char(proto_model(ii));
    %%%%%%%
    nmb_of_image_set=zeros(1,10);
    pr_set=zeros(1,10);
    top_1_set=zeros(1,10);
    model_accuracy_comparison=zeros(2,nmb_of_ft_models);
    for fm=1:nmb_of_ft_models
        for imgnt1kdataset=1:10
            reportname1 = sprintf('Model_%s/Evaluation_Data/Model_Accuracy/training_data_batch_%d_feature_module_performance_%s_var.mat',...
                md,imgnt1kdataset, md);
            aa=sprintf('classification_data_%d',fm);
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
        model_accuracy_comparison(1,fm)=nmb_of_image_set*pr_set'/sum(nmb_of_image_set);
        model_accuracy_comparison(2,fm)=nmb_of_image_set*top_1_set'/sum(nmb_of_image_set);
%         assignin('base',featured_model(fm), model_accuracy_comparison')
    end
assignin('base',md, model_accuracy_comparison')
end

%%
tb1=table(featured_model,T_h1_m1,S_h1_m1,T_h1_m2,S_h1_m2,T_h2_m1,S_h2_m1)

%%
table(featured_model,T_h1_m1,T_h1_m2,T_h2_m1)

table(featured_model,S_h1_m1,S_h1_m2,S_h2_m1)

%%
for fm=1:nmb_of_ft_models
    %%%%%%%
    nmb_of_image_set=zeros(1,10);
    pr_set=zeros(1,10);
    top_1_set=zeros(1,10);
    model_accuracy_comparison_2=zeros(2,nmb_of_ft_models);
    for ii=1:nmb_of_proto_models
        md=char(proto_model(ii));
        for imgnt1kdataset=1:10
            reportname1 = sprintf('Model_%s/Evaluation_Data/Model_Accuracy/training_data_batch_%d_feature_module_performance_%s_var.mat',...
                md,imgnt1kdataset, md);
            aa=sprintf('classification_data_%d',fm);
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
        model_accuracy_comparison_2(1,ii)=nmb_of_image_set*pr_set'/sum(nmb_of_image_set);
        model_accuracy_comparison_2(2,ii)=nmb_of_image_set*top_1_set'/sum(nmb_of_image_set);
    end
    assignin('base',char(featured_model(fm)), round(model_accuracy_comparison_2,3)')
end

tb2=table(proto_model,model_1,model_2,model_3,model_4,model_5,model_6)
