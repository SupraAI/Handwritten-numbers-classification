clear all
close all

% Initialization of image data
load('Data\DigitTrain_0.mat')
nb_imgs = size(imgs, 3); % number of images
taille = size(imgs, 1) * size(imgs, 2); % number of pixels

data = reshape(imgs, [taille, nb_imgs]); % Reshape the image into a vector
M(1, 1) = size(imgs, 3); % Number of images for class 0
c = labels.';

for i = 1:9 % Class 0 has already been loaded, start from 1
    load(['Data/DigitTrain_' num2str(i) '.mat']);
    nb_imgs = size(imgs, 3); % number of images
    taille = size(imgs, 1) * size(imgs, 2); % number of pixels

    X_reshape = reshape(imgs, [taille, nb_imgs]);
    M(1, i + 1) = size(imgs, 3);
    data = [data, X_reshape];
    c = [c, labels.'];
end

%---------------------------------------------------------
data = [ones(1, size(data, 2)); data]; % Add bias to the data matrix
N = size(data, 2); % Number of examples
P = 10; % Number of classes
MC = zeros(P, N); % Target matrix
NbItMax = 250; % Number of iterations

% Define the target matrix
% This will allow distinguishing the classes
a = 0;
for j = 1:P
    for i = 1:M(j)
        MC(j, a + i) = 1;
    end
    a = a + M(j);
end

% Initialize parameters
rho = 0.1;
w = zeros(size(data, 1), P); % Weights, size (n x P)
z = zeros(P, N); % z, size (P x N)
y = zeros(P, N); % y, size (P x N)
J = zeros(NbItMax, 1); % Cost function

% 1. Initial calculation of z and y
z = w' * data; % Calculate z for the initial iteration (size P x N)
y = 1 ./ (1 + exp(-z)); % Sigmoid activation for each output

% Initialize J
J(1) = sum(sum((y - MC).^2)) / (2 * N); % Cost function for initial parameters

% 2. Gradient descent
for ind = 2:NbItMax
    for p = 1:P
        % Calculate z and y for each class
        z(p, :) = w(:, p)' * data;
        y(p, :) = 1 ./ (1 + exp(-z(p, :)));

        % Calculate the gradient
        deriv1 = (y(p, :) - MC(p, :));
        deriv2 = y(p, :) .* (1 - y(p, :));
        gradJ = (data * (deriv1 .* deriv2)') / N;

        % Update the parameters
        w(:, p) = w(:, p) - rho * gradJ;
    end

    % Calculate the cost function (J) for this iteration
    J(ind) = sum(sum((y - MC) .^ 2)) / (2 * N);
end
printf('Training part OK \n')
%---------------------------------------------------------
% Test part

% Initialization of test data
load('Data\DigitTest_0.mat')
nb_imgs = size(imgs, 3); % number of images
taille = size(imgs, 1) * size(imgs, 2); % number of pixels

datatest = reshape(imgs, [taille, nb_imgs]);
Ntest = size(datatest, 2);
ctest = labels.';

for i = 1:9
    load(['Data/DigitTest_' num2str(i) '.mat']);
    nb_imgs = size(imgs, 3); % number of images
    taille = size(imgs, 1) * size(imgs, 2); % number of pixels

    X_reshape = reshape(imgs, [taille, nb_imgs]);
    M(1, i + 1) = size(imgs, 3);
    datatest = [datatest, X_reshape];
    ctest = [ctest, labels.'];
end

datatest = [ones(1, size(datatest, 2)); datatest]; % Add bias

% Initialize ztest and ytest
ztest = zeros(P, size(datatest, 2));
ytest = zeros(P, size(datatest, 2));
classe_ytest = ones(1, N);

% Calculate elements of ztest and ytest
for i = 1:P
    ztest(i, :) = w(:, i)' * datatest; % Calculate ztest for class p
    ytest(i, :) = 1 ./ (1 + exp(-ztest(i, :))); % Calculate ytest (sigmoid activation)
end

% Classification: assign each example to the class with the highest probability
[~, classe_ytest] = max(ytest);
classe_ytest = classe_ytest - 1; % Adjustment to make classes range from 0 to 9

%---------------------------------------------------------
% Generate the confusion matrix
matriceConf = confusionmat(ctest, classe_ytest, 'Order', 0:9);

% Display the confusion matrix
fprintf('Confusion matrix:\n');
disp(matriceConf);

% Create a figure for the display
figure('Name', 'Confusion Matrix', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);

% Create and display the confusion chart
confChart = confusionchart(matriceConf, 0:9);

% Customize the appearance
confChart.Title = 'Confusion Matrix for Handwritten Digit Classification';
confChart.XLabel = 'Predicted Class';
confChart.YLabel = 'True Class';
confChart.ColumnSummary = 'column-normalized';
confChart.RowSummary = 'row-normalized';

% Global accuracy
precision = sum(diag(matriceConf)) / sum(matriceConf(:)) * 100;
xlabel(sprintf('Global Accuracy: %.2f%%', precision));

