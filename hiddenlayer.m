clear all
close all
load('DataSimulation/DataTrain_2Classes_Perceptron_2.mat');

N = length(data);
data = [ones(1, N); data]; % Adding the bias to the data matrix
P = 2; % Number of classes
MC = zeros(P, N); % Target matrix
nbItMax = 1500; % Number of iterations
L = 15; % Number of neurons in the hidden layer

c = c + 1;
% Filling the target matrix with a One-Hot encoding
for n = 1:N
    MC(c(n), n) = 1; % Setting 1 in the row corresponding to the class
    % where c(n) is the class of example n (assumed to be between 1 and P)
end

% Initializing the parameters
#w1 = randn(3, L) * sqrt(2 / 3); % Used to find the optimal initial values
#w2 = randn(L, P) * sqrt(2 / L); % Used to find the optimal initial values
rho = 12;
w1 = zeros(3, L);
w1(:, 1) = [7.7044, 1.9583, 0.16956];
z1 = zeros(L, N);
y1 = zeros(L, N);

w2 = zeros(L, P); % size 2x15
w2(:, 1) = [-4.1545, -6.165, 0.081463, -7.0899, 2.0582, 6.2843, 7.0354, 4.0291, 4.5846, -7.7424, -2.9155, 6.0034, 5.7462, 0.70189, -6.1575];
z2 = zeros(P, N); % size 15x2000
y2 = zeros(P, N); % size 15x2000

% Performing forward propagation for the hidden layer
z1 = w1.' * data; % Calculating z1 (L x N)
y1 = 1 ./ (1 + exp(-z1));

% Performing forward propagation for the output layer
z2 = w2.' * y1; % Calculating z2 (P x N)
y2 = 1 ./ (1 + exp(-z2)); % Sigmoid activation for the output layer (P x N)

% Initializing J
J2(1) = sum(sum((y2 - MC).^2)) / (2 * N); % Criterion for the initial parameters

% 2. Performing gradient descent
for ind = 2:nbItMax
    % Calculating z and y
    z1 = w1.' * data;
    y1 = 1 ./ (1 + exp(-z1));

    z2 = w2.' * y1;
    y2 = 1 ./ (1 + exp(-z2));

    % Calculating the error for the output layer
    delta2 = (y2 - MC) .* y2 .* (1 - y2);

    % Calculating the error for the hidden layer
    delta1 = (w2 * delta2) .* y1 .* (1 - y1);

    % Calculating the gradients
    gradJ2 = (y1 * delta2') / N;
    gradJ1 = (data * delta1') / N;

    % Updating the weights
    w1 = w1 - rho * gradJ1;
    w2 = w2 - rho * gradJ2;

    % Calculating the cost criterion (J) for the iteration
    J2(ind) = sum(sum((y2 - MC).^2)) / (2 * N);
end

printf('Training part OK\n')

% Adding after the training loop
figure;
plot(J2);
title('Cost evolution during training');
xlabel('Iteration');
ylabel('Cost');

% Testing Part
load('DataSimulation/DataTest_2Classes_Perceptron_2.mat');

% Initializing ztest and ytest
ztest1 = zeros(L, N);
ytest1 = zeros(L, N);

ztest2 = zeros(P, N);
ytest2 = zeros(P, N);

datatest = [ones(1, N); dataTest]; % Adding the bias

% Calculating ztest and ytest elements
ztest1 = w1.' * datatest; % Calculating ztest for class p
ytest1 = 1 ./ (1 + exp(-ztest1)); % Calculating ytest (sigmoid activation)

ztest2 = w2.' * ytest1; % Calculating ztest for class p
ytest2 = 1 ./ (1 + exp(-ztest2)); % Calculating ytest (sigmoid activation)

% Classification: assigning each example to the class with the highest probability
[~, classe_ytest] = max(ytest2);
classe_ytest = classe_ytest - 1;

%---------------------------------------------------------
% Generating the confusion matrix
matriceConf = confusionmat(cTest, classe_ytest, 'Order', [0 1]);

% Displaying the confusion matrix
fprintf('Confusion matrix:\n');
disp(matriceConf);

% Displaying the confusion matrix graphically
figure('Name', 'Confusion Matrix', 'NumberTitle', 'off');
imagesc(matriceConf);
colormap(jet);
colorbar;

% Annotating the cells
textStrings = num2str(matriceConf(:), '%d');  % Converting values to text
textStrings = strtrim(cellstr(textStrings));  % Removing unnecessary spaces
[x, y] = meshgrid(1:2);  % Coordinates for each cell
text(x(:), y(:), textStrings, 'HorizontalAlignment', 'center', 'Color', 'white');

% Display parameters
title('Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');
set(gca, 'XTick', 1:2, 'XTickLabel', {'Class 0', 'Class 1'}, 'YTick', 1:2, 'YTickLabel', {'Class 0', 'Class 1'});
axis square;

% Global precision
precision = sum(diag(matriceConf)) / sum(matriceConf(:)) * 100;
fprintf('Global precision: %.2f%%\n', precision);

