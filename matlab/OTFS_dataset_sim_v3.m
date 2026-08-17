
clear; close all; clc;
rng(20250911,'twister');

%% ===== Parameters =====
fc      = 2.15e9; c = 3e8;
v_kmh   = 360; v_ms = v_kmh/3.6;
fd_max  = v_ms/(c/fc);

M = 256; N = 8; Nfft = 1024;
delta_f = 15e3; Fs = Nfft*delta_f;
Lframe = Nfft * N;   % number of samples in one OTFS frame (time-domain)

Nt = 16; Nr = 1;
pilotOverhead = 0.25;
numSamples = 2; chunkSize = 1;
saveDir = fullfile(pwd,'dataset_chunks');
if ~exist(saveDir,'dir'), mkdir(saveDir); end

numDD = M * N;
numPilots = round(numDD * pilotOverhead);
assert(numPilots > 0);

%% ===== Create/configure CDL channel object =====
try
    cdl = nrCDLChannel;
catch
    cdl = nrTDLChannel;
end
cdl.DelayProfile = 'CDL-D';
cdl.CarrierFrequency = fc;
cdl.SampleRate = Fs;
try, cdl.MaximumDopplerShift = fd_max; end

% Antenna array / counts
if isprop(cdl,'TransmitAntennaArray')
    try
        cdl.TransmitAntennaArray = phased.URA('Size',[4 4], 'ElementSpacing', [0.5 0.5]);
    catch
        if isprop(cdl,'NumTransmitAntenna'), cdl.NumTransmitAntenna = Nt; end
    end
else
    if isprop(cdl,'NumTransmitAntenna'), cdl.NumTransmitAntenna = Nt; end
end
if isprop(cdl,'NumReceiveAntenna'), cdl.NumReceiveAntenna = Nr; end

%% ===== Compute paddingSamples =====
paddingSamples = 0;
try
    infoS = info(cdl); % may work
    if isfield(infoS,'PathDelays') && ~isempty(infoS.PathDelays)
        maxDelay = max(infoS.PathDelays);
        paddingSamples = ceil(maxDelay * Fs) + 2; % +2 margin
    end
catch
    try
        if isprop(cdl,'PathDelays')
            maxDelay = max(cdl.PathDelays);
            paddingSamples = ceil(maxDelay * Fs) + 2;
        end
    catch
        paddingSamples = 0;
    end
end
fprintf('Using paddingSamples = %d\n', paddingSamples);

%% ===== Pilot positions & pilot amplitudes =====
rng(12345,'twister');
pilotIdx = randsample(numDD, numPilots);
[pilot_rows, pilot_cols] = ind2sub([M, N], pilotIdx);

rng(98765,'twister');
pilotSymbolsAllTx = (randn(numPilots, Nt) + 1j*randn(numPilots, Nt)) ./ sqrt(2);

tx_dd_all = complex(zeros(M, N, Nt));
for t = 1:Nt
    for p = 1:numPilots
        tx_dd_all(pilot_rows(p), pilot_cols(p), t) = pilotSymbolsAllTx(p, t);
    end
end

tx_td_unpadded = complex(zeros(Lframe, Nt));
for t = 1:Nt
    tx_td_unpadded(:, t) = otfs_mod(tx_dd_all(:,:,t), M, N, Nfft);
end

tx_td_multi = complex(zeros(Lframe + paddingSamples, Nt)); % padded
for t = 1:Nt
    tx_td_multi(:,t) = [ zeros(paddingSamples,1); tx_td_unpadded(:,t) ];
end

%% ===== Build Phi using padded tx frames =====
fprintf('Building Phi (using padding alignment) ...\n');
Phi = complex(zeros(numDD, numPilots * Nt));
colIdx = 0;

for t = 1:Nt
    tx_td_padded = tx_td_multi(:, t);  % length Lframe + paddingSamples
    for p = 1:numPilots
        colIdx = colIdx + 1;

        H_unit = complex(zeros(M,N));
        l0 = pilot_rows(p);
        k0 = pilot_cols(p);
        H_unit(l0, k0) = 1;

        h_td = otfs_mod(H_unit, M, N, Nfft);   % length = Lframe

        y_conv = conv(tx_td_padded, h_td, 'same');     % length padding + 2*Lframe - 1
        startIdx = paddingSamples + 1;
        endIdx   = paddingSamples + Lframe;
        if length(y_conv) < endIdx
            y_conv(endIdx) = 0; 
        end
        rx_frame = y_conv(startIdx:endIdx);   

        y_dd = otfs_demod(rx_frame, M, N, Nfft);
        Phi(:, colIdx) = reshape(y_dd, numDD, 1);
    end
    if mod(t,4) == 0, fprintf('  Done Tx %d / %d\n', t, Nt); end
end

% Save Phi metadata
phiFile = fullfile(saveDir, 'Phi_pilotWeighted_padded.mat');
save(phiFile, 'Phi', 'pilotIdx', 'pilot_rows', 'pilot_cols', 'pilotSymbolsAllTx', 'paddingSamples', '-v7.3');
fprintf('Saved Phi (%d x %d) to %s\n', size(Phi,1), size(Phi,2), phiFile);

%% ===== Consistency test =====
fprintf('Running consistency test...\n');

set_cdl_seed(cdl, randi(2^28));
H_add_truth = extract_H_ADD_from_channel_withPadding(cdl, M, N, Nt, Nfft, Fs, paddingSamples);
h_full = reshape(H_add_truth, [numDD * Nt, 1]);

h_pilot = complex(zeros(numPilots * Nt, 1));
idx = 0;
for t = 1:Nt
    for p = 1:numPilots
        idx = idx + 1;
        lin = sub2ind([M,N], pilot_rows(p), pilot_cols(p));
        h_pilot(idx) = h_full((t-1)*numDD + lin);
    end
end

y_pred = Phi * h_pilot;

set_cdl_seed(cdl, randi(2^28));
rx_td_multi = channel_filter(cdl, tx_td_multi);  % outputs [paddingSamples + Lframe] or longer
if ndims(rx_td_multi) > 2
    rx_td_multi = squeeze(rx_td_multi(:,:,1));
end
rx_td = rx_td_multi(:,1);
if length(rx_td) < paddingSamples + Lframe
    rx_td(end+1 : paddingSamples + Lframe) = 0;
end
rx_frame = rx_td(paddingSamples + 1 : paddingSamples + Lframe);
y_dd_full = otfs_demod(rx_frame, M, N, Nfft);
y_full = reshape(y_dd_full, [], 1);


rel_err = norm(y_pred - y_full) / max(1e-12, norm(y_full));
fprintf('Consistency relative error (pred vs full) = %.6e\n', rel_err);

d = y_pred - y_full;
fprintf('rel_err = %.6e, max_abs_diff = %.6e, mean_abs_diff = %.6e\n', rel_err, max(abs(d)), mean(abs(d)));
%disp([ real(y_full(1:20)), imag(y_full(1:20)), real(y_pred(1:20)), imag(y_pred(1:20)), abs(y_full(1:20)), abs(y_pred(1:20)) ]);

%% ===== Dataset generation =====
numChunks = ceil(numSamples / chunkSize);
sampleCounter = 0;

for chunk = 1:numChunks
    thisChunk = min(chunkSize, numSamples - sampleCounter);
    HADD_chunk = complex(zeros(M, N, Nt, thisChunk));
    yDD_chunk = complex(zeros(numDD, thisChunk));
    feature_chunk = zeros(numPilots * Nt, thisChunk);

    for s = 1:thisChunk
        sampleCounter = sampleCounter + 1;

        set_cdl_seed(cdl, randi(2^28));

        H_add_truth = extract_H_ADD_from_channel_withPadding(cdl, M, N, Nt, Nfft, Fs, paddingSamples);
        H = H_add_truth;  
        energies = abs(H(:)).^2;        
        energies_sorted = sort(energies, 'descend');
        total_energy = sum(energies_sorted);

        num_total = numel(energies_sorted);
        num_top = ceil(0.10 * num_total);
        
        top_energy = sum(energies_sorted(1:num_top));
        
        percent_energy = 100 * top_energy / total_energy;
        
        fprintf('Total energy = %.3e\n', total_energy);
        fprintf('Energy in top 10%% elements = %.3e (%.2f%% of total)\n', top_energy, percent_energy);

        max_energy = max(abs(H_add_truth(:)).^2);
        threshold = 0.0005 * max_energy;
        
        mask = (abs(H_add_truth).^2 >= threshold);
        
        H_add_truth = H_add_truth .* mask;
        
        num_total = numel(H_add_truth);
        num_zero = nnz(H_add_truth == 0);
        percent_zero = 100 * num_zero / num_total;
        
        fprintf('max energy = %.3e\n', max_energy);
        fprintf('Threshold (1%% of total) = %.3e\n', threshold);
        fprintf('Fraction of entries set to zero = %.2f%%\n', percent_zero);

        H_slice = H_add_truth(:, :, 1);  
        
        csvwrite('H_slice_mag.csv', abs(H_slice));
        
        csvwrite('H_slice_real.csv', real(H_slice));
        csvwrite('H_slice_imag.csv', imag(H_slice));
        
        H_combined = [real(H_slice(:)), imag(H_slice(:))]; % flatten to 2 columns
        csvwrite('H_slice_complex.csv', H_combined);
        
        disp('Saved H_slice to CSV files.');


        HADD_chunk(:,:,:,s) = H_add_truth;
        
        h_full = reshape(H_add_truth, [numDD * Nt, 1]);
        h_pilot = complex(zeros(numPilots * Nt, 1));
        idx = 0;
        for t = 1:Nt
            for p = 1:numPilots
                idx = idx + 1;
                lin = sub2ind([M,N], pilot_rows(p), pilot_cols(p));
                h_pilot(idx) = h_full((t-1)*numDD + lin);
            end
        end

        y_noisefree = Phi * h_pilot;

        SNRdB = 10;
        sigP = mean(abs(y_noisefree).^2);
        noiseVar = sigP / (10^(SNRdB/10));
        noiseVec = sqrt(noiseVar/2) * (randn(size(y_noisefree)) + 1j*randn(size(y_noisefree)));
        y_obs = y_noisefree + noiseVec;

        yDD_chunk(:, s) = y_obs;
        feature_chunk(:, s) = abs(Phi' * y_obs);

        if mod(sampleCounter, 50) == 0
            fprintf('  Generated sample %d\n', sampleCounter);
        end
    end

    outFile = fullfile(saveDir, sprintf('chunk_%03d.mat', chunk));
    save(outFile, 'HADD_chunk', 'yDD_chunk', 'feature_chunk', '-v7.3');
    fprintf('Saved %s\n', outFile);
end

fprintf('Done: %d samples saved in %d chunks. Phi stored at %s\n', sampleCounter, numChunks, phiFile);


%% ===== Helper functions =====

function x_td = otfs_mod(x_dd, M, N, Nfft)
    X_tf = ifft(fft(x_dd, [], 2).', [], 2).';   % M x N -> freq-time
    s_mat = zeros(Nfft, N);
    s_mat(1:M, :) = X_tf;
    ofdm_time = ifft(s_mat, Nfft, 1);
    x_td = ofdm_time(:);
end

function y_dd = otfs_demod(y_td, M, N, Nfft)
    y_td = y_td(:);
    L = Nfft * N;
    if length(y_td) < L, y_td(end+1 : L) = 0; end
    y_td = y_td(1:L);
    rx_mat = reshape(y_td, Nfft, N);
    Y_freq = fft(rx_mat, Nfft, 1);
    X_tf = Y_freq(1:M, :);
    y_dd = fft(ifft(X_tf, [], 2).', [], 2).';
end



function rx_td_multi = channel_filter(chObj, tx_td_multi)
    try
        out = chObj(tx_td_multi);
    catch
        out = step(chObj, tx_td_multi);
    end
    if ndims(out) > 2
        out = squeeze(out(:,:,1));
    end
    rx_td_multi = out;
end

function H_add = extract_H_ADD_from_channel_withPadding(chObj, M, N, Nt, Nfft, Fs, padding)

    L = Nfft * N;
    txLen = L + padding;
    H_add = complex(zeros(M, N, Nt));

    pilot_delay = min(5, M);

    for t = 1:Nt
        tx_dd = complex(zeros(M, N));
        tx_dd(pilot_delay, :) = 1;         

        tx_td = otfs_mod(tx_dd, M, N, Nfft);   % length L
        tx_padded = [zeros(padding,1); tx_td]; % length padding+L

        tx_multi = zeros(txLen, Nt);
        tx_multi(:, t) = tx_padded;

        rx_multi = channel_filter(chObj, tx_multi);
        rx = squeeze(rx_multi(:,1));   

        if length(rx) < padding + L
            rx(padding + L) = 0;
        end
        rx_frame = rx(padding + (1:L));

        H_add(:,:,t) = otfs_demod(rx_frame, M, N, Nfft);
    end
end




function set_cdl_seed(chObj, seedVal)
    try
        release(chObj);
    catch
        % ignore
    end
    try
        chObj.Seed = seedVal;
    catch
        %ignore 
    end
    try
        reset(chObj);
    catch
        % ignore 
    end
end



%% 
disp(cdl)
try
  fprintf('MaximumDopplerShift = %g Hz\n', cdl.MaximumDopplerShift);
catch, warning('No MaximumDopplerShift property'); end
Tframe = (Nfft * N) / Fs;
fprintf('Frame duration Tframe = %.6f s\n', Tframe);
fprintf('Max doppler * Tframe = %.3e -> phase (rad) = %.3e\n', fd_max*Tframe, 2*pi*fd_max*Tframe);