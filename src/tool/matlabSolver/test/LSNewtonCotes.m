function w =LSNewtonCotes(n)
    a = zeros(floor(sqrt(n/0.09)), n+1);
    b = zeros(floor(sqrt(n/0.09)), 1);
    x = linspace(-1, 1, n+1);
    P = {1, [1, 0]};
    a(1, :) = 1;
    a(2, :) = x;
    b(1) = 2;
    b(2) = 0;
    for i = 1:floor(sqrt(n/0.09))
        P{i+2} = (2*i+1)/(i+1)*conv([1,0], P{i+1}) - i/(i+1)*[0, 0, 1*P{i}];
        a(i+2, :) = polyval(P{i+2}, x);
        b(i+2) = diff(polyval(polyint(P{i+2}), [-1, 1]));
    end
    w = a \ b;
%     w = lsqnonneg(a, b);
    return
end
