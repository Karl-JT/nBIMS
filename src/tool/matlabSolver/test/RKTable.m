function [a, b, c]=RKTable(n)
    [c, b]= gauleg(n);
    c = (c+1)/2;
    b = b/2;

    a = zeros(n);
    for i = 1:n
        for j = 1:n
            p = 1;
            for n = 1:n
                if (n~=j)
                    p = conv(p, [1, -c(n)])/(c(j)-c(n));
                end
            end
            q = polyint(p);
            a(i, j) = diff(polyval(q, [0, c(i)]));
        end
    end
	return
end

            
function [x,w]=gauleg(n)
    x=zeros(n,1);
    w=zeros(n,1);
    m=(n+1)/2;
    xm=0.0;
    xl=1.0;
    for i=1:m
    z=cos(pi*(i-0.25)/(n+0.5));
    while 1
              p1=1.0;
              p2=0.0;
              for j=1:n
               p3=p2;
                   p2=p1;
                   p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
              end
          pp=n*(z*p1-p2)/(z*z-1.0);
              z1=z;
    z=z1-p1/pp;
    if (abs(z-z1)<eps), break, end
    end
      x(i)=xm-xl*z;
    x(n+1-i)=xm+xl*z;
    w(i)=2.0*xl/((1.0-z*z)*pp*pp);
    w(n+1-i)=w(i);
    end
    x=x';
    w=w';
    return
end