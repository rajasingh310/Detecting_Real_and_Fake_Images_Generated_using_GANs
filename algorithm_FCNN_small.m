%{

~ FCNN network model for training

Description: To classify real and fake images
Author: Rajababu Singh
Matriculation number: 1530580
Supervisor: Dr. Margret Keuper

%}

function net = algorithm_FCNN_small(x_train, x_val, y_train, y_val)

    layers = [
         imageInputLayer(size(x_train, 1, 2, 3),"Name","imageinput")

         dropoutLayer(0.3)

         convolution2dLayer(1,20)

         fullyConnectedLayer(numel(unique(y_train)),"Name","fc_3")
         softmaxLayer("Name","softmax")
         classificationLayer("Name","classoutput")
         ];
    
    options = trainingOptions('adam',...    
        'InitialLearnRate',0.2,... 
        'LearnRateSchedule', 'piecewise',... 
        'LearnRateDropFactor', 0.65,...
        'LearnRateDropPeriod', 1,...    
        'MaxEpochs',500,...           
        'MiniBatchSize', 75,...  
        'Shuffle', 'every-epoch',...
        'ValidationData', {x_val, y_val},...
        'ValidationFrequency', 200, ...
        'ValidationPatience', 5, ...
        'VerboseFrequency', 200, ...
        'plots','training-progress',...
        'ExecutionEnvironment','gpu');
    
    net = trainNetwork(x_train, y_train, layers, options);

end