% ML Project LSTM

%% House-keeping
clc
close all

%% Data import
% train_in;
% test_in;
% valid_in;
disp('Loading saved data from mat file ...');
load('data.mat')

%% View the distribution of the classes in the data using a histogram
f = figure;
f.Position(3) = 1.5*f.Position(3);

h = histogram(train.label);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")

%% Get the frequency counts of the classes and their names from the histogram
classCounts = h.BinCounts;
classNames = h.Categories;

%% Extract the text data and labels from the partitioned tables
disp('Pre-processing data ...');
textDataTrain = train.statement;
textDataTest = test.statement;
textDataValid = valid.statement;
YTrain = train.label;
YTest = test.label;
YValid = valid.label;

%% Visualize the training text data using a word cloud.
figure
wordcloud(textDataTrain);
title("Training Data")

%% Preprocess the training data
textDataTrain = erasePunctuation(textDataTrain);
textDataTrain = lower(textDataTrain);
documentsTrain = tokenizedDocument(textDataTrain);

%% Preprocess the Validation data
textDataValid = erasePunctuation(textDataValid);
textDataValid = lower(textDataValid);
documentsValid = tokenizedDocument(textDataValid);

%% Train Word Embedding
disp('Word Embedding Training ...');
embeddingDimension = 300;
embeddingEpochs = 50;

emb = trainWordEmbedding(documentsTrain, ...
    'Dimension',embeddingDimension, ...
    'NumEpochs',embeddingEpochs, ...
    'Verbose',1)

%% Convert Document to Sequences
documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")

%% Truncate the training documents to have length 18 using docfun
sequenceLength = 20;
documentsTruncatedTrain = docfun(@(words) words(1:min(sequenceLength,end)),documentsTrain);

%% Convert the documents to sequences of word vectors
XTrain = doc2sequence(emb,documentsTruncatedTrain);

%% To pad sequences of word vectors for LSTM networks
% Apply the example function leftPad, shown at the end of this example, to each of the sequences in XTrain. This function left-pads the sequences with zeros so that they have the same length.
for i = 1:numel(XTrain)
    XTrain{i} = leftPad(XTrain{i},sequenceLength);
end
XTrain(1:5);

%% Truncate the valid documents to have length 18 using docfun
sequenceLength = 20;
documentsTruncatedValid = docfun(@(words) words(1:min(sequenceLength,end)),documentsValid);

%% Convert the documents to sequences of word vectors
XValid = doc2sequence(emb,documentsTruncatedValid);

%% To pad sequences of word vectors for LSTM networks
% Apply the example function leftPad, shown at the end of this example, to each of the sequences in XValid. This function left-pads the sequences with zeros so that they have the same length.
for i = 1:numel(XValid)
    XValid{i} = leftPad(XValid{i},sequenceLength);
end
XValid(1:5);

%% Create and Train LSTM Network
disp('Defining Neural Network ...');
inputSize = embeddingDimension;
outputSize = 180;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(outputSize,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

%% Specify the training options
disp('Initializing training hyperparameters ...');

LearnRateDropFactor = 0.2;
LearnRateDropPeriod = 5;
LSTMEpochs = 50;
InitialLearnRate = 1;

options = trainingOptions('sgdm',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',LearnRateDropFactor,...
    'LearnRateDropPeriod',LearnRateDropPeriod,...'adam', ...
    ...'GradientThreshold',1, ...
    'Shuffle','every-epoch',...
    'MaxEpochs',LSTMEpochs,...
    'InitialLearnRate',InitialLearnRate, ...
    'Plots','training-progress', ...
    'Verbose',1);

%% Train the LSTM network using the trainNetwork function
disp('Training Network ...');
net = trainNetwork(XTrain,YTrain,layers,options);

%% Preprocess the test data using the same steps as the training documents.
disp('Preprocessing Test data ...');
textDataTest = erasePunctuation(textDataTest);
textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);

%% Convert the test documents to sequences using the same steps as the training documents
documentsTruncatedTest = docfun(@(words) words(1:min(sequenceLength,end)),documentsTest);
XTest = doc2sequence(emb,documentsTruncatedTest);
for i=1:numel(XTest)
    XTest{i} = leftPad(XTest{i},sequenceLength);
end
XTest(1:5);

%% Classify the Valid documents using the trained LSTM network.
disp('Predicting on Test data from trained network ...');
YValidPred = classify(net,XValid);

%% Calculate the classification accuracy
Valid_accuracy = sum(YValidPred == YValid)/numel(YValidPred);
disp(['Validation accuracy = ' num2str(Valid_accuracy*100) '%']);

%% Classify the test documents using the trained LSTM network.
YTestPred = classify(net,XTest);

%% Calculate the classification accuracy
Test_accuracy = sum(YTestPred == YTest)/numel(YTestPred);
disp(['Test accuracy = ' num2str(Test_accuracy*100) '%']);

%%
save(['mdl_emb' num2str(inputSize) '_' num2str(embeddingEpochs) '_lstm' num2str(embeddingEpochs) '.mat'],'net','Valid_accuracy','Test_accuracy');

%% Predict Using New Data
% % Classify the event type of three new weather reports. Create a string array containing the new weather reports.
% 
% reportsNew = [ ...
%     "Lots of water damage to computer equipment inside the office."
%     "A large tree is downed and blocking traffic outside Apple Hill."
%     "Damage to many car windshields in parking lot."
%     ];
% 
% %% Preprocess the text data using the same steps as the training documents.
% 
% reportsNew = lower(reportsNew);
% reportsNew = erasePunctuation(reportsNew);
% documentsNew = tokenizedDocument(reportsNew);
% 
% %% Convert the text data to sequences using the doc2sequence and leftPad example functions. Specify the sequence length to be the same as the training data.
% 
% documentsTruncatedNew = docfun(@(words) words(1:min(sequenceLength,end)),documentsNew);
% XNew = doc2sequence(emb,documentsTruncatedNew);
% for i=1:numel(XNew)
%     XNew{i} = leftPad(XNew{i},sequenceLength);
% end
% 
% %% Classify the new sequences using the trained LSTM network.
% 
% [labelsNew,score] = classify(net,XNew);
% 
% %% Show the weather reports with their predicted labels.
% 
% [reportsNew string(labelsNew)]
