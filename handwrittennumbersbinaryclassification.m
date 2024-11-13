clear all
close all
clc
load('Data\DigitTest_1.mat')  % Load data for digit 1
c = zeros(1, length(imgs))';  % Initialize the target labels for digit 1
X_1 = reshape(imgs, [400, length(imgs)]);  % Reshape the digit 1 images
load('Data\DigitTest_7.mat')  % Load data for digit 7
X_3 = reshape(imgs, [400, length(imgs)]);  % Reshape the digit 7 images
X = [X_1, X_3];  % Combine the data for digits 1 and 7
X = [ones(1, length(X)); X];  % Adding a bias term
c = [c; ones(length(imgs), 1)]';  % Target labels: 0 for digit 1, 1 for digit 7
N = length(c);  % Number of examples

W = zeros(401, 1);  % Initializing weight vector (including bias term)
Z = zeros(1, N);  % Initializing Z values
Y = zeros(1, N);  % Initializing output values

% Parameters
rho = 10.^(-1);  % Learning rate
nbItMax = 250;  % Maximum number of iterations

% 2. Gradient Descent
% a. Initialization
J(1) = (1 / (2 * N)) * (sum(Y - c).^2);  % Initial cost function value

% b. Iterations (training loop)
for ind = 2:nbItMax
    for n = 1:401
        gradJ(n, ind - 1) = sum((Y - c) .* Y .* (1 - Y) .* X(n, :)) / N;  % Gradient computation
    end
    % Updating weights
    W(:, ind) = W(:, ind - 1) - rho * gradJ(:, ind - 1);

    % Compute output for each sample
    for n = 1:N
        Z(n) = W(:, ind - 1)' * X(:, n);  % Compute Z (input to the sigmoid)
        Y(n) = 1 / (1 + exp(-Z(n)));  % Sigmoid activation
    end
    % Updating cost function
    J(ind) = (1 / (2 * N)) * sum((Y - c).^2);
end

% Now performing the test phase

nbreBon = 0;  % Initializing the count of correctly classified samples
classeY = ones(1, N);  % Initializing predicted classes for training data

% Load test data for digit 1 and digit 7
load('Data\DigitTest_1.mat')
cTest = zeros(1, length(imgs))';  % Initializing target labels for the test set (digit 1)
labels8 = labels;  % Storing the labels for digit 1
Xt_1 = reshape(imgs, [400, length(imgs)]);  % Reshaping the test images for digit 1

load('Data\DigitTest_7.mat')
cTest = [cTest; ones(length(imgs), 1)]';
labels9 = labels;
Xt_3 = reshape(imgs, [400, length(imgs)]);

Xtest = [Xt_1, Xt_3];  % Combining the test data for digits 1 and 7
Xtest = [ones(1, length(X)); Xtest];  % Adding bias term to test data
Nt = length(cTest);  % Number of test samples
Ztest = zeros(1, Nt);  % Initialize test Z values
classeYt = ones(1, Nt);  % Initialize predicted test classes

% Perform predictions on the test data
for n = 1:Nt
    Ztest(n) = W(:, nbItMax)' * Xtest(:, n);  % Compute Z for test samples
end
Ytest = 1 ./ (1 + exp(-Ztest));  % Sigmoid activation for test output

% Classify based on the predicted probability
for n = 1:Nt
    if Ytest(n) < 0.5
        classeYt(n) = 0;  % Assign class 0 if the probability is less than 0.5 (digit 1)
        labelsPredict(n) = 8;  % Predicted label for digit 1
    end
end

% Counting correctly classified samples
for n = 1:Nt
    if classeYt(n) == cTest(n)  % Check if predicted class matches actual class
        nbreBon = nbreBon + 1;  % Increment correct count
    end
end

% Calculating success rate
tauxReusssite = nbreBon / Nt * 100;  % Success rate (accuracy) in percentage
fprintf('Success rate: %.2f%%\n', tauxReusssite);

% Confusion matrix
Co = confusionmat(cTest, classeYt);  % Generate confusion matrix
confusionchart(Co, [1, 7]);  % Plot confusion matrix with class labels 1 and 7

