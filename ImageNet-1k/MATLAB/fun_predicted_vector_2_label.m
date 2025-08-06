function out=fun_predicted_vector_2_label(vector,nmb_of_labels)
%
% predicted vector of length 25 to digit label 1, ..., 25
% vector=predicted_vector;
[~,dtsz]=size(vector);
digit=zeros(1,dtsz);
distance=zeros(1,dtsz);
nn=nmb_of_labels;
v=vector(1:nn,:);
I=eye(nn);
vv=zeros(nn,1);
for j=1:dtsz
    for i=1:nn
        [cc,~]=loss_function(v(:,j),I(:,i));
        vv(i)=cc;
    end
    [aa,idx]=min(vv);
    digit(j)=idx;
    distance(j)=aa;
end
out.label=digit;
out.distance=distance;
%%%%%%%%%%
    function [loss,dloss]=loss_function(predicted_x,true_y)
        % cross_entropy loss
        %
        eps=1e-8;
        [n, N]=size(true_y);
        loss=sum(sum((true_y+eps).*log((true_y+eps)./(predicted_x+eps))))/N;
        %         ind_loss=sum((true_y+eps).*log((true_y+eps)./(predicted_x+eps)),1);
        bb=-(true_y(1:end-1,:)+eps)./(predicted_x(1:end-1,:)+eps);
        w=bb+ones(n-1,1)*(true_y(end,:)+eps)./(predicted_x(end,:)+eps);
        dloss=w'/N;
    end
%%%%%%%%%%
end
