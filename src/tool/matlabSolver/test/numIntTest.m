format long

%%% Numerical parameters:
NC = 6;
w = CloseNewtonCotes(NC+1)

Nn = 64;
N = Nn*(NC+1) - 1;  
L = 1;

dx = L/(N+1);   		% distance between two physical points
x = (0:N)'*dx;      % physical space discretization
y = x;

x = [x; 1];
y = x;
[X, Y] = meshgrid(x,y);

vx = -cos(2*pi*X).*sin(2*pi*Y);
vy = sin(2*pi*X).*cos(2*pi*Y);

vInx = X.^0.5.*Y.^0.5.*vx;
vIny = X.^0.5.*Y.^0.5.*vy;

vIn = vInx - vIny;

% contourf(X, Y, vIn, 'LineColor', 'none');

h = dx;
w = w*h;

wtmp = zeros(N+2, 1);
for i = 0:(ceil((N+2)/(NC+1))-2)
    wtmp(i*(NC+1)+1:(i+1)*(NC+1)+1) = wtmp(i*(NC+1)+1:(i+1)*(NC+1)+1)+w;
end
w = wtmp;

w2d = kron(w, w');

QoI = 100*vIn.*w2d;
sum(sum(QoI))

reg(log2(Nn/2)+1) = sum(sum(QoI))


