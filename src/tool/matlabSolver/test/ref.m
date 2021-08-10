Obs_ref = -0.965649652758160;
Noise_Var = 1.0;
n = 16;
[x, w] = gauleg(n);
num = zeros(n, 1);
dem = zeros(n, 1);
parfor i = 1:n
    [QoI(i), Obs(i)] = run_NS_high_order(x(i));
    num(i) = QoI(i)*w(i)*exp(-0.5*(Obs(i)-Obs_ref)^2/Noise_Var);
    dem(i) = w(i)*exp(-0.5*(Obs(i)-Obs_ref)^2/Noise_Var);
    disp([i, x(i), QoI(i), Obs(i), num(i), dem(i)]);
end
reference(log2(n/2)) = sum(num)/sum(dem);
disp(reference);

% Obs_ref = -0.965649652758160;
% Noise_Var = 0.1;
% n = 128;
% [x, w] = gauher(n);
% num = zeros(n, 1);
% dem = zeros(n, 1);
% parfor i = 1:n
%     [QoI(i), Obs(i)] = run_NS_high_order(sqrt(2.0)*x(i));
%     num(i) = QoI(i)*w(i)*exp(-0.5*(Obs(i)-Obs_ref)^2/Noise_Var);
%     dem(i) = w(i)*exp(-0.5*(Obs(i)-Obs_ref)^2/Noise_Var);
%     disp([i, x(i), QoI(i), Obs(i), num(i), dem(i)]);
% end
% reference(log2(n/2)) = sum(num)/sum(dem);
% disp(reference);

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

function [x,w]=gauher(n)
    x=zeros(n,1);
    w=zeros(n,1);
	C1 = 0.9084064;
	C2 = 0.05214976;
	C3 = 0.002579930;
	C4 = 0.003986126;
	m = (n+1.0)/2.0;
	anu = 2.0*n + 1.0;
    for i=1:m
		rhs = (3+(i-1)*4)*pi/anu;
		r3 = rhs^(1.0/3.0);
		r2 = r3^(2);
		theta = r3*(C1+r2*(C2+r2*(C3+r2*C4)));
		z = sqrt(anu)*cos(theta);
        while 1
			p1=1.0/pi^(1.0/4.0);
			p2=0.0;
            for j=1:n
				p3=p2;
				p2=p1;
				p1=sqrt(2.0/j)*z*p2-sqrt((j-1.0)/j)*p3;
            end
			pp=sqrt(2.0*n)*p2;
			z1 = z;
			z = z1-p1/pp;
            if abs(z-z1) < 3.0e-13
				break;
            end
        end
		x(i) = z;
		x(n-i+1) = -z;
		w(i) = 2.0/(pp*pp);
		w(n-i+1) = w(i);
    end
end
