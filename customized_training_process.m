%% Load Training Data & define class catalog & define input image size

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



% Datensatz laden:
disp('Loading training data...')
[XTrain,YTrain] = digitTrain4DArrayData;
disp('Loading training data done...')
%disp('Loading test data...')
%[XTest,YTest] = digitTest4DArrayData;
%disp('Loading test data done...')



% Datensatz aufteilen --> ???


%% define network (dlnet)

% Input-Layer:
% 28x28 Neuronen/Pixel, 1: Anzahl Werte pro Neuron, Normalization: none
layer1 =  imageInputLayer([28 28 1], Normalization="none");

% 1 Hidden-Layer:
layer2 = fullyConnectedLayer(392);
% Wähle Aktivierungsfunktion sigma aus tanH, leakyReluLayer,...
layer3 = reluLayer;

% Output-Layer:
layer4 = fullyConnectedLayer(10);
% erzeuge Wahrscheinlichkeitsverteiung aus layer6
layer5 = softmaxLayer;
% Loss-Funktion: ist hier nicht benötigt da benutzerdefiniertes Training

NN_layers = [
    layer1
    layer2
    layer3
    layer4
    layer5
    ];

% Bilde deep-learning-network:
% convert to a layer graph
lgraph = layerGraph(NN_layers);
% Create a dlnetwork object from the layer graph.
dlnet = dlnetwork(lgraph);

% visualize the neural network
analyzeNetwork(dlnet)

%% Specify Training Options (define hyperparameters)

% miniBatchSize: choose size so it can be devided by lenght(XTrain)
minibatchsize = 125;

% numEpochs: Maximale Epochenzahl:
numEpochs = 2;

% learnRate: Schrittweite des Gradientenabstiegsalgorithmus
initiallearnrate = 0.001;

% numIterationsPerEpoch: Anzahl Trainingselemente per Epoche:
numIterationsPerEpoch = length(XTrain)/minibatchsize;


%% Train neural network

% initialize the average gradients and squared average gradients
averageGrad = [];
averageSqGrad = [];

% initialize vector for indices for each epoch:
epoch_indices = 1:length(XTrain);

% initialize loss:
loss = [];

disp('Start training:')

iteration = 0;

% trainingloop
for epoch = 1:numEpochs

   epoch
   % update learnable parameters based on mini-batch of data
   % shuffle minibatches: shuffle indices
   epoch_indices = randperm(length(epoch_indices));

   for i = 1:numIterationsPerEpoch

        iteration= iteration + 1;

        % Read mini-batch of data and convert the labels to dummy variable:
        % determine current minibatch
        actual_minibatch = epoch_indices((i-1)*minibatchsize+1:i*minibatchsize);

        % convert labels to dummy variable:
        Y=zeros(10, minibatchsize);
        for j = 1:minibatchsize
            Y(YTrain(actual_minibatch(j)), j) = 1;
        end

        % Convert mini-batch of data to a dlarray:
        dlX = dlarray(single(XTrain(:,:,:,actual_minibatch)), 'SSCB');

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients helper function:
        % Calculate gradient, loss-Function and prediction for the currentobject:
        [grad,loss,dlYPred] = dlfeval(@modelGradients,dlnet,dlX,Y);

        % Update the network parameters using the optimizer, like SGD, Adam
        [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,grad, ...
            averageGrad,averageSqGrad,i,initiallearnrate);

        % Calculate accuracy & show the training progress.
        number_correct_labels = 0;
        for j = 1:minibatchsize
            % analize output of the NN:
            [max_value, prediction] = max(dlYPred(:, j));
       
            if single(prediction) == single(YTrain(actual_minibatch(j)))
              number_correct_labels = number_correct_labels + 1;
            end
        end
        accuracy = number_correct_labels/minibatchsize;

        % Plot training Process:
        subplot(2,1,1); plot(iteration, accuracy, 'k.');
        xlabel('iteration');
        ylabel('accuracy(iteration)');
        title('Accuracy = f(iteration)');
        hold on;
 
        subplot(2,1,2); plot(iteration, loss, 'b.');
        xlabel('iteration');
        ylabel('loss(iteration)');
        title('loss = f(iteration)');
        hold on;


        % option: validation
   end
end





%% test neural network & visualization 


%% Define Model Gradients Function

function [gradient,loss,dlYPred] = modelGradients(dlnet,dlX,Y)

    % forward propagation 
    dlYPred = forward(dlnet,dlX);
    % calculate loss -- varies based on different requirement
    loss = crossentropy(dlYPred,Y);
    % calculate gradients 
    gradient = dlgradient(loss,dlnet.Learnables);
    
end


