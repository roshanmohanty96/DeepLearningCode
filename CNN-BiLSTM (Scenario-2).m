%% ============================================================
% Scenario-2: Monsoon Season CNN-BiLSTM MME
% Monsoon months: June to October (1970-2014)
%% ============================================================

clc; clear; close all;
excelFile = 'GCM data.xlsx';

%% ================= SCENARIO-2 SETTINGS =======================

% Monsoon season: June to October
monsoonMonths = 6:10;

% 75% training and 25% testing
trainRatio = 0.75;

%% ================= SCENARIO-2 CNN-BiLSTM HYPERPARAMETERS =====
kernelSize       = 2;
dropoutRate      = 0.20;
maxEpochs        = 120;
miniBatchSize    = 64;
initialLearnRate = 0.005;

% BiLSTM hidden units
numHiddenUnits = 12;

rng(42);

%% ================= LOAD EXCEL DATA ===========================

T = readtable(excelFile, 'VariableNamingRule', 'preserve');

disp('Excel columns detected:');
disp(T.Properties.VariableNames');

%% ================= READ DATE COLUMN ==========================

dateCol = 'Date';

if ~ismember(dateCol, T.Properties.VariableNames)
    error('Date column not found. Please check the Excel header.');
end

datesRaw = T.(dateCol);

if isdatetime(datesRaw)
    dates = datesRaw;
elseif isnumeric(datesRaw)
    dates = datetime(datesRaw, 'ConvertFrom', 'excel');
else
    dates = datetime(datesRaw, 'InputFormat', 'M/d/yyyy');
end

%% ================= EXTRACT IMD AND SELECTED GCM DATA =========

imdCol = 'IMD';

if ~ismember(imdCol, T.Properties.VariableNames)
    error('IMD column not found. Please check the Excel header.');
end

Y = T.(imdCol);

% Selected five GCMs for MME
gcmCols = { ...
    'EC-Earth3'
    'EC-Earth3-Veg'
    'MPI-ESM1-2-HR'
    'MPI-ESM1-2-LR'
    'INM-CM4-8'};

missingCols = setdiff(gcmCols, T.Properties.VariableNames);

if ~isempty(missingCols)
    disp('Missing GCM columns:');
    disp(missingCols');
    error('Some selected GCM columns are missing in the Excel file.');
end

fprintf('\nSelected GCMs used for Scenario-2 CNN-BiLSTM MME:\n');
disp(gcmCols');

X = T{:, gcmCols};

X = double(X);
Y = double(Y);

numGCM = size(X,2);

fprintf('Number of selected GCMs used = %d\n', numGCM);

%% ================= REMOVE MISSING VALUES =====================

validIdx = ~isnat(dates) & all(~isnan(X),2) & ~isnan(Y);

dates = dates(validIdx);
X = X(validIdx,:);
Y = Y(validIdx);

fprintf('\nTotal valid daily records before monsoon filtering = %d\n', length(Y));

%% ================= FILTER MONSOON SEASON ONLY ================

idxMonsoon = ismember(month(dates), monsoonMonths);

datesMonsoon = dates(idxMonsoon);
XMonsoon = X(idxMonsoon,:);
YMonsoon = Y(idxMonsoon);

fprintf('Total monsoon daily records = %d\n', length(YMonsoon));

%% ================= CREATE TEMPORAL SEQUENCES =================
% Look-back window for temporal learning
lookBack = 7;

% Each sample uses previous lookBack monsoon records.
% Input sequence size: [number of GCMs x lookBack]
% Output: IMD rainfall at current monsoon day.

XSeq = cell(length(YMonsoon)-lookBack+1, 1);
YSeq = nan(length(YMonsoon)-lookBack+1, 1);
dateSeq = NaT(length(YMonsoon)-lookBack+1, 1);

count = 0;

for i = lookBack:length(YMonsoon)

    count = count + 1;

    Xi = XMonsoon(i-lookBack+1:i, :)';   % [numGCM x lookBack]
    Yi = YMonsoon(i);                    % current-day IMD rainfall

    XSeq{count,1} = Xi;
    YSeq(count,1) = Yi;
    dateSeq(count,1) = datesMonsoon(i);

end

fprintf('Total monsoon sequence samples after look-back = %d\n', numel(YSeq));

%% ================= 75% TRAINING AND 25% TESTING ==============

numSamples = numel(YSeq);
numTrain = floor(trainRatio * numSamples);

idxTrain = 1:numTrain;
idxTest  = numTrain+1:numSamples;

XTrain = XSeq(idxTrain);
YTrain = YSeq(idxTrain);

XTest = XSeq(idxTest);
YTest = YSeq(idxTest);

dateTrain = dateSeq(idxTrain);
dateTest  = dateSeq(idxTest);

fprintf('\nScenario-2 chronological split applied:\n');
fprintf('Training samples = %d\n', numel(YTrain));
fprintf('Testing samples  = %d\n', numel(YTest));

%% ================= NORMALIZATION =============================

% Compute normalization parameters from training data only
XTrainAll = cat(2, XTrain{:});   % [numGCM x total_sequence_steps]

muX = mean(XTrainAll, 2, 'omitnan');
sigX = std(XTrainAll, 0, 2, 'omitnan');
sigX(sigX == 0) = 1;

muY = mean(YTrain, 'omitnan');
sigY = std(YTrain, 0, 'omitnan');

if sigY == 0
    sigY = 1;
end

% Normalize input sequences
for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - muX) ./ sigX;
end

for i = 1:numel(XTest)
    XTest{i} = (XTest{i} - muX) ./ sigX;
end

% Normalize target
YTrainN = (YTrain - muY) ./ sigY;

%% ================= CNN-BiLSTM ARCHITECTURE ===================
% Scenario-2 settings:

poolingSize = 1;

layers = [
    sequenceInputLayer(numGCM, "Name", "input")

    convolution1dLayer(kernelSize, 16, ...
        "Padding", "same", ...
        "Name", "conv1")
    reluLayer("Name", "relu1")

    maxPooling1dLayer(poolingSize, ...
        "Stride", 1, ...
        "Name", "pool1")

    convolution1dLayer(kernelSize, 32, ...
        "Padding", "same", ...
        "Name", "conv2")
    reluLayer("Name", "relu2")

    bilstmLayer(numHiddenUnits, ...
        "OutputMode", "last", ...
        "Name", "bilstm")

    dropoutLayer(dropoutRate, "Name", "dropout")

    fullyConnectedLayer(1, "Name", "fc_out")
    regressionLayer("Name", "regression")
];
%% ================= TRAINING OPTIONS ==========================

options = trainingOptions("adam", ...
    "MaxEpochs", maxEpochs, ...
    "MiniBatchSize", miniBatchSize, ...
    "InitialLearnRate", initialLearnRate, ...
    "Shuffle", "every-epoch", ...
    "Verbose", true, ...
    "Plots", "training-progress");

%% ================= TRAIN MODEL ===============================

fprintf('\nTraining Scenario-2 CNN-BiLSTM model for monsoon season...\n');

tic
net = trainNetwork(XTrain, YTrainN, layers, options);
trainingTime = toc;

fprintf('Training completed in %.2f seconds.\n', trainingTime);

%% ================= PREDICTION ================================

YPredTrainN = predict(net, XTrain, "MiniBatchSize", miniBatchSize);
YPredTestN  = predict(net, XTest,  "MiniBatchSize", miniBatchSize);

% Denormalize predictions
YPredTrain = YPredTrainN .* sigY + muY;
YPredTest  = YPredTestN  .* sigY + muY;

% Rainfall cannot be negative
YPredTrain(YPredTrain < 0) = 0;
YPredTest(YPredTest < 0)   = 0;


%% ================= PERFORMANCE METRICS =======================

trainMetrics = computeMetrics(YTrain, YPredTrain);
testMetrics  = computeMetrics(YTest, YPredTest);

fprintf('\n================ SCENARIO-2 TRAINING PERFORMANCE ================\n');
fprintf('Correlation (r) = %.3f\n', trainMetrics.r);
fprintf('RMSE            = %.3f mm/day\n', trainMetrics.RMSE);
fprintf('PBias           = %.3f %%\n', trainMetrics.PBias);
fprintf('KGE             = %.3f\n', trainMetrics.KGE);

fprintf('\n================ SCENARIO-2 TESTING PERFORMANCE =================\n');
fprintf('Correlation (r) = %.3f\n', testMetrics.r);
fprintf('RMSE            = %.3f mm/day\n', testMetrics.RMSE);
fprintf('PBias           = %.3f %%\n', testMetrics.PBias);
fprintf('KGE             = %.3f\n', testMetrics.KGE);


%% ================= PLOTS =====================================

% Training scatter plot
figure;
scatter(YTrain, YPredTrain, 20, 'filled'); hold on;
maxVal = max([YTrain; YPredTrain], [], 'omitnan');
plot([0 maxVal], [0 maxVal], 'k--', 'LineWidth', 1.2);
grid on;
xlabel('IMD observed monsoon rainfall (mm/day)');
ylabel('CNN-BiLSTM MME rainfall (mm/day)');
title('Scenario-2 Training: Observed vs CNN-BiLSTM MME');

% Testing scatter plot
figure;
scatter(YTest, YPredTest, 20, 'filled'); hold on;
maxVal = max([YTest; YPredTest], [], 'omitnan');
plot([0 maxVal], [0 maxVal], 'k--', 'LineWidth', 1.2);
grid on;
xlabel('IMD observed monsoon rainfall (mm/day)');
ylabel('CNN-BiLSTM MME rainfall (mm/day)');
title('Scenario-2 Testing: Observed vs CNN-BiLSTM MME');

%% ================= LOCAL FUNCTION ============================


function metrics = computeMetrics(obs, sim)

    obs = obs(:);
    sim = sim(:);

    valid = ~isnan(obs) & ~isnan(sim);
    obs = obs(valid);
    sim = sim(valid);

    metrics.r = corr(obs, sim, 'Rows', 'complete');

    metrics.RMSE = sqrt(mean((sim - obs).^2, 'omitnan'));

    % PBias formulation:
    % Negative PBias = overestimation
    % Positive PBias = underestimation
    metrics.PBias = 100 * sum(obs - sim, 'omitnan') / sum(obs, 'omitnan');

    if numel(obs) < 3 || std(obs) == 0 || mean(obs) == 0
        metrics.KGE = NaN;
    else
        r = metrics.r;
        alpha = std(sim) / std(obs);
        beta  = mean(sim) / mean(obs);

        metrics.KGE = 1 - sqrt((r - 1)^2 + ...
                                (alpha - 1)^2 + ...
                                (beta - 1)^2);
    end

end
