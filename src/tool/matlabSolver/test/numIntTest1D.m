
N = 16;
L = 2*pi;

dx = L/(N+1);
x = (0:N)'*dx;

y = sin(x)+sin(2*x);
test1 = fft(y);
Y = fftshift(fft(y));

N2 = 64;
dx = L/(N2+1);
x2 = (0:N2)'*dx;

Y2 = padarray(Y, (N2-N)/2, 0, 'both');
test2 = fftshift(Y2)
y2 = ifft(ifftshift((N2+1)/(N+1)*Y2));
plot(x,y);
hold on;
plot(x2,y2,'x-');


