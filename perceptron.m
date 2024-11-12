clear all
close all
load('DataSimulation/DataTrain_2Classes_Perceptron.mat');

% Parameters
N=length(data);

rho = 0.1;
fulldata=[ones(1,N);data]; # Adding bias
y=zeros(1,N);
nbItMax=800;
w=zeros(3,nbItMax);
z=zeros(1,N);
z = w(:,1) .'* fulldata;
y = 1./(1+exp(-z)); % phi(z)


% 2. Gradient Descent
% a. Initialization

% w(:,1) = rand(3,1) * 10 - 5; % Used to find the optimal initial values to take
% w(:,1)=[-3.1738,-2.5614,0.52567]; % initial values
v=(y-c).^2;
J(1) = (sum(v))/(2*N); % Criterion value for initial parameters




% b. Iterations
for ind = 2:nbItMax;
    % Gradient calculation
    deriv1=(y - c); % derivative of f w.r.t. y
    deriv2=y.*(1-y); % derivative of y w.r.t. z
    gradJ(1,ind-1) = sum(deriv1.*deriv2.*fulldata(1,:))/N;
    gradJ(2,ind-1) = sum(deriv1.*deriv2.*fulldata(2,:))/N;
    gradJ(3,ind-1) = sum(deriv1.*deriv2.*fulldata(3,:))/N;

    % updating parameters
    w(:,ind) = w(:,ind-1)-rho*gradJ(:,ind-1);

    for n= 1:N
        z(n)=w(:,ind-1).'*fulldata(:,n);
        y(n)=1/(1+exp(-z(n)));
    end

    J(ind) = (sum((y-c).^2))/(2*N);

    ## if J(ind)< J(ind-1)
    ##    rho= 2*rho;
    ## elseif J(ind)==J(ind-1)
    ##    rho=2*rho;
    ## else
    ##    rho=rho/2;
    ##    w(:,ind)=w(:,ind-1);
    ##    J(ind)=J(ind-1);
    ## end

    % Plotting the curves
end
figure;
subplot(4,1,1);
plot( [1:1:nbItMax],w(1,1:nbItMax));
xlabel("number of iterations");
ylabel("Value of w1");
title("Evolution of parameter w1 (weight 1)");

subplot(4,1,2);
plot( [1:1:nbItMax],w(2,1:nbItMax));
xlabel("number of iterations");
ylabel("Value of w2");
title("Evolution of parameter w2 (weight 2)");

subplot(4,1,3);
plot([1:nbItMax],w(3,1:nbItMax));
xlabel("number of iterations");
ylabel("Value of w3");
title("Evolution of parameter w3 (weight 3)");

subplot(4,1,4);
plot([1:nbItMax],J);
xlabel("number of iterations");
ylabel("Criterion value");
title("Evolution of the criterion");

% Test part
load('DataSimulation\DataTest_2Classes_Perceptron.mat');
xtest=[ones(1,N);dataTest];
ztest=zeros(1,N);
for n= 1:N
    ztest(n)=w(:,nbItMax)'*(xtest(:,n));
end
ytest=1./(1+exp(-ztest));

% Classification threshold to determine classes
classe_ypred=ones(1,N);

for i=1:N
    if ytest(i)<0.5
        classe_ypred(i)=0;
    end
end

% Calculating accuracy
n=0;
for i= [1:N]
    if classe_ypred(i)==cTest(i)
        n=n+1;
    endif
end

precision= (n/N)*100 % Accuracy of correct predictions obtained

