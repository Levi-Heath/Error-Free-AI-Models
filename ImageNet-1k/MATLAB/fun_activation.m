function [out1,out2]=fun_activation(x)
%
%
%
[sz1,sz2]=size(x);
A=zeros(sz1,sz2);
B=zeros(sz1,sz1,sz2);
for zz=1:sz2
    [A(:,zz),B(:,:,zz)]=ReLU(x(:,zz));
end

out1=A;
out2=B;
%%%%%%%%
    function [y,dy]=ReLU(s)
        aa=(s>0).*1;
        y=aa.*s;
        dy=diag(aa);
    end
%%%%%%%%
end
