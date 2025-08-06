function batch_training_performance=fun_ANN_Model(image,param)
%
% model=param.model;
nmb_of_hidden_layers=param.nmb_of_hidden_layers;

patch=param.patch;
nmb_of_modules=param.nmb_of_modules;
nmb_of_module_subsets=param.nmb_of_module_subsets;
nmb_of_colors=param.nmb_of_colors;
channels_names=param.channels_names;
% cross_entropy=param.cross_entropy;
nmb_of_labs_per_module=param.nmb_of_labs_per_module;
sz=size(image);
if length(sz)==2
    nmb_of_data=1;
else
    nmb_of_data=sz(2);
end

if nmb_of_data>1
    batch_training_performance=zeros(2,nmb_of_data,nmb_of_modules,nmb_of_module_subsets,nmb_of_colors);
else
    batch_training_performance=zeros(2,nmb_of_modules,nmb_of_module_subsets,nmb_of_colors);
end
for module=1:nmb_of_modules
    for subset=1:nmb_of_module_subsets
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

            if nmb_of_data>1
                a_0=image(:,:,color); 
            else
                a_0=image(:,color);
            end
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
            abs_label=prediction.label+(module-1)*nmb_of_labs_per_module;
            if nmb_of_data>1
                batch_training_performance(:,:,module,subset,color)=[abs_label;prediction.distance];
            else
                batch_training_performance(:,module,subset,color)=[abs_label;prediction.distance];
            end
        end
    end
end
if nmb_of_data>1
    batch_training_performance=permute(batch_training_performance,[1,3,4,5,2]);%
end
end