function out=fun_training_evaulation(model,nmb_of_hidden_layers)
%
%
nmb_of_modules=40;
nmb_of_module_subsets=2;

channels_names={'R','G','B','RGg1','RBg1','GBg1','RGg2','RBg2','GBg2','RB','RG','GB','eRGB','BW','X','Y','Z'};
% feature_RGB=[1 0 0
%     0      1      0
%     0      0      1
%     0.618  0.382  0
%     0.618  0      0.382
%     0      0.618  0.382
%     0.382  0.618  0
%     0.382  0      0.618
%     0      0.382  0.618
%     0.5    0.5    0
%     0.5    0      0.5
%     0      0.5    0.5
%     1/3    1/3    1/3
%     0.299  0.587  0.114
%     0.4125 0.3576  0.1804
%     0.2126 0.7152  0.0722
%     0.0193 0.1192  0.9502];

nmb_of_colors=length(channels_names);
patch=0;
nmb_of_labs_per_module=25;

pstvrt_model=zeros(nmb_of_colors,nmb_of_modules,nmb_of_module_subsets);
%%
parfor module=1:nmb_of_modules
    %     pm=param;
    channels_names={'R','G','B','RGg1','RBg1','GBg1','RGg2','RBg2','GBg2','RB','RG','GB','eRGB','BW','X','Y','Z'};
    for subset=1:nmb_of_module_subsets
        reportname1 = sprintf('../../Data_Transformation/Transformed_IN1k_Data/Transformed_Data_for_SGD/train_data_patch_%d_module_%d_subset_%d_for_%d_labels_per_module.mat', ...
            patch,module,subset,nmb_of_labs_per_module);
        data_load=load(reportname1);
        % %         reportname1 = sprintf('../%s.mat',model);
        % %         temp_load=load(reportname1);
        % %         model_parameter=temp_load.model_parameters;
        for color=1:nmb_of_colors
            reportname1 = sprintf('../Model_Parameter/Trained_Parameter_patch_%d_module_%d_subset_%d_ch_%s.mat',...
                patch, module, subset, char(channels_names(color)));
            %      str=struct('W', W, 'b', b);
            temp_load=load(reportname1);
            W=temp_load.W;
            b=temp_load.b;
            % Assign the trained parameters
            %
            W1=W.LayerName1;
            W2=W.LayerName2;
            b1=b.LayerName1;
            b2=b.LayerName2;
            % %                 W1=model_parameter.W1(:,:,color,subset,module);
            % %                 W2=model_parameter.W2(:,:,color,subset,module);
            % %                 b1=model_parameter.b1(:,:,color,subset,module);
            % %                 b2=model_parameter.b2(:,:,color,subset,module);
            a_0=data_load.data(:,:,color);
            true_label=data_load.labels;
            dtsz=length(true_label);
            nmb_of_labels=length(b2);
            z1=W1*a_0+b1;
            [a1,~]=fun_activation(z1);
            z2=W2*a1+b2;

            if nmb_of_hidden_layers==1
                [a2,~]=fun_softmax(z2);
                predicted_vector=a2;
            else
                W3=W.LayerName3;
                b3=b.LayerName3;
                nmb_of_labels=length(b3);
                [a2,~]=fun_activation(z2);

                z3=W3*a2+b3;
                [a3,~]=fun_softmax(z3);
                predicted_vector=a3;
            end
            prediction=fun_predicted_vector_2_label(predicted_vector,nmb_of_labels);
            %%
            % Compute the error and positive rates.
            %
            v=abs(true_label-prediction.label);
            errt=sum(1.*(v>0))/dtsz;
            pstvrt=(1-errt)*100;
            pstvrt_model(color,module,subset)=pstvrt;
            aa=(module-1)*nmb_of_labs_per_module; %data_load.label_table(2,1)-1;
            training_performance=[true_label+aa;prediction.label+aa;prediction.distance];
            out=fun_save_result(model,patch, module, subset,color,channels_names,training_performance);
        end
    end
    module
end
reportname1 = sprintf('%s_1_performance.mat',model);
save(reportname1, 'pstvrt_model');
out=[];
end