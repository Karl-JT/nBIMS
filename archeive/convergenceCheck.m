numCoeff = 3;

chain0 = readmatrix("MCMCChain0.csv");
chain1 = readmatrix("MCMCChain1.csv");
chain2 = readmatrix("MCMCChain2.csv");
chain3 = readmatrix("MCMCChain3.csv");

chain0 = chain0(:, 1:numCoeff);
chain1 = chain1(:, 1:numCoeff);
chain2 = chain2(:, 1:numCoeff);
chain3 = chain3(:, 1:numCoeff);

size0 = size(chain0, 1);
size1 = size(chain1, 1);
size2 = size(chain2, 1);
size3 = size(chain3, 1);

n = min([size0, size1, size2, size3]);
b = 2*floor(n/100);
bk = b;
i = 1;
R = (1:10);
detW = (1:10);
detB = (1:10);

iteration = (1:b:n);

while(bk < n)
    W = zeros(numCoeff);
    B = 0;
    sequence = cat(3, chain0(bk/2:bk, 1:numCoeff), chain1(bk/2:bk, 1:numCoeff), chain2(bk/2:bk, 1:numCoeff), chain3(bk/2:bk, 1:numCoeff));
    average = mean(sequence, 1);
    for m = 1:4
        for t = 1:bk/2
            W = W + (sequence(t, :, m) - average(1, :, m)).'*(sequence(t, :, m) - average(1, :, m));
        end
    end 
    W = W/4/(bk/2-1); 
    
    totalAve = mean(average, 3);
    for m = 1:4
        B = B + (average(1, :, m)-totalAve).'*(average(1, :, m)-totalAve);
    end  
    B = B/3*(bk/2);
    V = (bk/2-1)/bk/2*W + (1+1/4)*B/(bk/2);
    tempMatrix = W\B/(bk/2);
    e = eig(tempMatrix);
    lamda = max(e);
    R(i) = (bk/2-1)/(bk/2) + 3/4*lamda;
    detW(i) = det(W); 
    detB(i) = det(B); 
    i = i+1;
    bk = bk + b;
end

figure(1);
plot(iteration(1:end-1), R);
title("RPSF");
xlabel("iterations");

figure(2);
plot(detW(end-10:end));
hold on;
plot(detB(end-10:end));
title("detW and detB");
legend("W", "B");

figure(3);
corrplot(chain0(bk/2:end, 1:end));
figure(4);
corrplot(chain1(bk/2:end, 1:end));
figure(5);
corrplot(chain2(bk/2:end, 1:end));
figure(6);
corrplot(chain3(bk/2:end, 1:end));

figure(7)
totalMix = [chain0(bk/2:end, 1:end);chain1(bk/2:end, 1:end);chain2(bk/2:end, 1:end);chain3(bk/2:end, 1:end)];
totalMixTable = array2table(totalMix);
for i = 1:numCoeff
    totalMixTable.Properties.VariableNames{i} = ['c', int2str(i-1)];
end
corrplot(totalMixTable);

sampleSize = 25;
sample = zeros(numCoeff, sampleSize);
for i = 1:numCoeff
    sample(i, :) = normrnd(mean(totalMix(:, i)), sqrt(var(totalMix(:, i))), sampleSize, 1);
end

% %yFile0 = readmatrix("yFile0.csv");
x = (-1:1/99:1);
figure(8);
hold all;
for i = 1:sampleSize
    test = sample(1, i);
    for j = 2:numCoeff
        test = test + sample(j, i)*cos(pi*x*(j-1));
    end
    plot(x, test);
end
title("function samples from target distribution");
xlabel("eta");
ylabel("epsilon");


sampleSize = 25;
sample = zeros(numCoeff, sampleSize);
for i = 1:numCoeff
    sample(i, :) = normrnd(mean(chain0(:, i)), sqrt(var(chain0(:, i))), sampleSize, 1);
end

% %yFile0 = readmatrix("yFile0.csv");
x = (-1:1/99:1);
figure(9);
hold all;
for i = 1:sampleSize
    test = sample(1, i);
    for j = 2:numCoeff
        test = test + sample(j, i)*cos(pi*x*(j-1));
    end
    plot(x, test);
end
title("function samples from target distribution");
xlabel("eta");
ylabel("epsilon");

sampleSize = 25;
sample = zeros(numCoeff, sampleSize);
for i = 1:numCoeff
    sample(i, :) = normrnd(mean(chain1(:, i)), sqrt(var(chain1(:, i))), sampleSize, 1);
end

% %yFile0 = readmatrix("yFile0.csv");
x = (-1:1/99:1);
figure(10);
hold all;
for i = 1:sampleSize
    test = sample(1, i);
    for j = 2:numCoeff
        test = test + sample(j, i)*cos(pi*x*(j-1));
    end
    plot(x, test);
end
title("function samples from target distribution");
xlabel("eta");
ylabel("epsilon");

sampleSize = 25;
sample = zeros(numCoeff, sampleSize);
for i = 1:numCoeff
    sample(i, :) = normrnd(mean(chain2(:, i)), sqrt(var(chain2(:, i))), sampleSize, 1);
end

% %yFile0 = readmatrix("yFile0.csv");
x = (-1:1/99:1);
figure(11);
hold all;
for i = 1:sampleSize
    test = sample(1, i);
    for j = 2:numCoeff
        test = test + sample(j, i)*cos(pi*x*(j-1));
    end
    plot(x, test);
end
title("function samples from target distribution");
xlabel("eta");
ylabel("epsilon");


sampleSize = 25;
sample = zeros(numCoeff, sampleSize);
for i = 1:numCoeff
    sample(i, :) = normrnd(mean(chain3(:, i)), sqrt(var(chain3(:, i))), sampleSize, 1);
end

% %yFile0 = readmatrix("yFile0.csv");
x = (-1:1/99:1);
figure(12);
hold all;
for i = 1:sampleSize
    test = sample(1, i);
    for j = 2:numCoeff
        test = test + sample(j, i)*cos(pi*x*(j-1));
    end
    plot(x, test);
end
title("function samples from target distribution");
xlabel("eta");
ylabel("epsilon");