modeNum = 4;
MCMCChain0 = readmatrix("./MCMCChain0.csv");

for i = 1:modeNum
    legendStr{i} = sprintf('C %d', i);
end

MCMC = MCMCChain0(:, 1:end-1);
MCMC = normcdf(MCMC);
MCMC = MCMC*2-1;
figure(1)
plot(MCMC);
title("trace plot")
xlabel("sample number")
ylabel("epsilon")
legend(legendStr)


figure(2);
subplot(2, 1, 2)
plot(movmean(MCMC, 50000));
ylim([-1, 1]);
title("running average with 50000 samples");
xlabel("sample number");
ylabel("epsilon")
legend(legendStr)

MCMCCumSum = cumsum(MCMC);
count = 1:size(MCMC, 1);
MCMCCumMean = MCMCCumSum./count';
subplot(2, 1, 1)
plot(MCMCCumMean);
title("Cumulative Average");
xlabel("sample number");
ylabel("epsilon")
legend(legendStr)


MCMCtable = array2table(MCMC(floor(end/2):end, :));
for i = 1:modeNum
MCMCtable.Properties.VariableNames{i} = ['C' num2str(i)];
end
figure(3)
corrplot(MCMCtable);

mu = mean(MCMC(floor(end/2):end, :));
covMatrix = cov(MCMC(floor(end/2):end, :));
R = mvnrnd(mu, covMatrix, 25);
x = (-1:2/198:1);
y = 1;
for i = 1:modeNum
    y = y + R(:, i)/i*cos(pi*i*x);
end
figure(4)
plot(x, y);
ylim([0 2])

y = 1;
for i = 1:modeNum
    y = y+mu(i)/i*cos(pi*i*x);
end
% figure(5)
% plot(x, y);
% ylim([0 2]);

figure(6)
y = 1;
for i = 1:modeNum
    y = y + MCMC(floor(end/2):end, i)/i*cos(pi*i*x);
end
muY = mean(y);
deviation = std(y);
upper = icdf('Normal', 0.95, muY, deviation);
lower = icdf('Normal', 0.05, muY, deviation);
plot(x, muY);
ylim([0 2])
hold on
% plot(x, upper);
% plot(x, lower);
fill([x flip(x)], [lower upper], 'k', 'LineStyle', 'none');
alpha(0.3);
title("Mean Results with 90% confidence level")
xlabel("eta")
ylabel("epsilon")

