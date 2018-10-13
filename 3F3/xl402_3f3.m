% 3F3 Random Variables and Random Numbers Generation
% Created by Tom Lu xl402
% Plot normal distribution


%{
subplot(141),
N = 100;
x2 = rand(N,1);
width=0.1;
axis= [0:width:1];
histogram(x2, axis)
hold on
mean_est=N*width*ones(size(axis));
plot(axis,mean_est, '-k')
std_est = sqrt(N*width*(1-width));
plot(axis, mean_est+3*std_est, '--b')
plot(axis, mean_est-3*std_est, '--r')
legend('U(0,1)','mean','+3\sigma', '-3\sigma')
title('N=10^2')

subplot(142),
N = 1000;
x2 = rand(N,1);
width=0.1;
axis= [0:width:1];
histogram(x2, axis)
hold on
mean_est=N*width*ones(size(axis));
plot(axis,mean_est, '-k')
std_est = sqrt(N*width*(1-width));
plot(axis, mean_est+3*std_est, '--b')
plot(axis, mean_est-3*std_est, '--r')
legend('U(0,1)','mean','+3\sigma', '-3\sigma')
title('N=10^3')

subplot(143),
N = 10000;
x2 = rand(N,1);
width=0.1;
axis= [0:width:1];
histogram(x2, axis)
hold on
mean_est=N*width*ones(size(axis));
plot(axis,mean_est, '-k')
std_est = sqrt(N*width*(1-width));
plot(axis, mean_est+3*std_est, '--b')
plot(axis, mean_est-3*std_est, '--r')
legend('U(0,1)','mean','+3\sigma', '-3\sigma')
title('N=10^4')

subplot(144),
N = 100000;
x2 = rand(N,1);
width=0.1;
axis= [0:width:1];
histogram(x2, axis)
hold on
mean_est=N*width*ones(size(axis));
plot(axis,mean_est, '-k')
std_est = sqrt(N*width*(1-width));
plot(axis, mean_est+3*std_est, '--b')
plot(axis, mean_est-3*std_est, '--r')
legend('U(0,1)','mean','+3\sigma', '-3\sigma')
title('N=10^5')

% Question 1
figure(1)
x1 = randn(1000,1);
subplot(211),
histogram(x1,20,'Normalization','pdf')
hold on
fplot(@(x) normpdf(x, 0, 1),[-3 3],'b')
ksdensity(x1, 'width', 1)
legend('p(x) Sampled','p(x)','kernel estimate, \sigma=1')
title('N(x|0,1)')

x2 = rand(1000,1);
subplot(212),
histogram(x2,[-0.45:0.1:1.45],'Normalization','pdf')
hold on
fplot(@(x) unifpdf(x, 0, 1),[-0.45 1.45],'b')
ksdensity(x2, 'width', 1)
[f,xi,bw] = ksdensity(x2, 'width', 0.1);
plot(xi,f,'-k')


legend('p(x) Sampled','p(x)','kernel estimate, \sigma=1', 'kernel estimate, \sigma=0.1')
title('U(0,1)')

% Question 2
figure(2)
a = 2;
b = 3;
y1 = a*x1 + b;
subplot(121),
histogram(x1,20,'Normalization','pdf')
hold on
histogram(y1,20,'Normalization','pdf')
title('y=ax+b')

fplot(@(x) normpdf(x, b, a),[-8 10],'b')
xlabel('random variable x or y')
ylabel('probability densities')
legend('p(x)','p(y)','predicted p(y)')

y2 = x1.^2;
subplot(122),
histogram(x1,20,'Normalization','pdf')
hold on
histogram(y2,20,'Normalization','pdf')
title('y=x^2')
fplot(@(x) exp(-x/2)/(sqrt(2*pi*x)),[0.1 10],'b')
xlabel('random variable x or y')
ylabel('probability densities')
legend('p(x)','p(y)','predicted p(y)')


% Question 3
figure(3)
fplot(@(x) exppdf(x,2), [-0.5 10],'b')
hold on
y3 = -log(1-x2);
histogram(y3,20,'Normalization','pdf')
ksdensity(x2, 'width', 0.1)
%}
% Question 4
figure(4)
alpha_array = [0.5,0.5,0.5,0.5,1.5,1.5,1.5,1.5];
beta_array = [-1.0,-0.25,0.25,1.0,-1.0,-0.25,0.25,1.0];
for plotId = 1 : 8
    subplot(2, 4, plotId) ;
    U = rand(20000,1)*pi-pi/2;
    V = exprnd(1, 20000,1);

    alpha = alpha_array(plotId);
    beta = beta_array(plotId);

    b = atan(beta* tan(pi*alpha/2));
    s = (1 + beta^2 * tan(pi * alpha / 2)^2).^(1/(2*alpha));
    X1 = (sin(alpha.*(U + b)))./(cos(U).^(1/alpha));
    X2 = ((cos(U - alpha.*(U+b)))./V).^((1-alpha)/alpha);
    X = s .* X1 .* X2;
    histogram(real(X),[-4:0.2:4],'Normalization','pdf')
    title(sprintf('α = %.2f, β = %.2f', alpha, beta) )
end



