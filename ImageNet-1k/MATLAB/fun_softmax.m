function [out1,out2]=fun_softmax(S)
%
% Column-wise normaliztion
%
[sz1,sz2]=size(S);
A=zeros(sz1,sz2);
B=zeros(sz1-1,sz1,sz2);
for zz=1:sz2
    [aaa,bbb]=softmax(S(:,zz));
    A(:,zz)=aaa;
    B(:,:,zz)=bbb;
end
out1=A;
out2=B;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [y,dy]=softmax(S)
        %         bb=1e+2;
        bb=1;
        S=S/bb;
        aa=exp(S);
        sm=sum(exp(S))+1e-16;
        y=aa./sm;

        [n,~]=size(S);
        m=n-1;
        dy=zeros(m,n);
        for i=1:m
            for j=1:n
                if j~=i
                    dy(i,j)=-exp(S(i)).*exp(S(j));
                else
                    dy(i,j)=exp(S(i)).*sm-exp(S(i)).*exp(S(j));
                end
            end
        end
        dy=dy/sm^2/bb;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
