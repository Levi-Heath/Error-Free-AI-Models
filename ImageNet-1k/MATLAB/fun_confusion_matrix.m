function output=fun_confusion_matrix(true_lab,pred_lab,pm)
%
nmb_of_images=length(true_lab);
nmb_of_lab=pm.nmb_of_lab;
% top_n=pm.top_n;
edges=1:1:(nmb_of_lab+1);
c_matrix=zeros(nmb_of_lab,nmb_of_lab);
seq_lab=1:nmb_of_lab;
% top_n_seq=1:top_n;
p_seq=(1:nmb_of_images)';
for lb=1:nmb_of_lab
    idx1=(true_lab==lb);
    aa=p_seq(idx1);
    nmb=length(aa);
    aa=pred_lab(aa);
    [N,~]=histcounts(aa,edges);
    rt=N/nmb*100;
    c_matrix(lb,:)=rt;
end
aa=diag(c_matrix);
% [~,cc]=sort(aa);
idx100=(aa==100);
output.lab_100=seq_lab(idx100);
% idx=flip(cc);
% c_matrix=c_matrix(idx, idx);
output.conf_matrix=c_matrix;
% top_n_lb=idx(top_n_seq);
% top_n_rt=diag(c_matrix(top_n_seq,top_n_seq));
% output.top_n_inf=[top_n_rt,top_n_lb];
end
