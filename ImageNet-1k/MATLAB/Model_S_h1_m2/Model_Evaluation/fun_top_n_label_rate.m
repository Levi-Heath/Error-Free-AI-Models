function [rates,labs]=fun_top_n_label_rate(c_mtrx,top_n)
% 
aa=c_mtrx;
bb=diag(aa);
[~,cc]=sort(bb);
cc=flip(cc);
dd=bb(cc);
seq=1:top_n;
rates=round(dd(seq),4);
labs=round(cc(seq),0);
end