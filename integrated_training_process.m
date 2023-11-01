    %% Load Training Data & define class catalog & define input image size
disp('Loading training data...')
% download from MNIST-home page or import dataset from MATLAB
% https://www.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html
% http://yann.lecun.com/exdb/mnist/

% Specify training and validation data
% Recommended naming >>>
% Train: dataset for training a neural network
% Test: dataset for test a trained neural network after training process
% Valid: dataset for test a trained neural network during training process
% X: input / for Classification: image
% Y: output / for Classification: label
% for example: XTrain, YTrain, XTest, YTest, XValid, YValid

%% define network
%   most basic network
NN_layers = [
    % Analog zu VL-Bsp:

    % Input-Layer:
    % 28x28 Neuronen/Pixel, 1: Anzahl Werte pro Neuron
    imageInputLayer([28 28 1])

    % 1 Hidden-Layer:
    fullyConnectedLayer(20)
    % Wähle Aktivierungsfunktion sigma aus tanH, leakyReluLayer,...
    reluLayer

    % 2 Hidden-Layer:
    fullyConnectedLayer(20)
    reluLayer

    % Output-Layer:
    fullyConnectedLayer(10)
    % normalisiere Aktivierungs-Werte der Neuronen auf in Summe =1
    softmaxLayer
    classificationLayer
];

% visualize the neural network
analyzeNetwork(NN_layers)
%% Specify Training Options (define hyperparameters)

% miniBatchSize
% numEpochs
% learnRate 
% executionEnvironment
% numIterationsPerEpoch 
% solver "sgdm" "rmsprop" "adam"

% training on CPU or GPU(if available);
% 'auto': Use a GPU if one is available. Otherwise, use the CPU.
% 'cpu' : Use the CPU
% 'gpu' : Use the GPU.
% 'multi-gpu' :Use multiple GPUs
% 'parallel :


%%  Train neural network
% define "trainingOptions"
% training using "trainNetwork"



%% test neural network & visualization 
% Calculate accuracy

