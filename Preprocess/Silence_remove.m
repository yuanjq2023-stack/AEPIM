function success = vad_single_file(wavepath, savepath)
% VAD_SINGLE_FILE: Endpoint detection and trimming of a single audio file 
% based on the dual-threshold method.
%
% Input Parameters:
%   wavepath: Full path to the original audio file.
%   savepath: Destination path to save the trimmed audio file.
%
% Output Parameters:
%   success: Returns 1 if processing is successful, 0 otherwise.

    success = 0;
    try
        %1. Read audio file
        [y, Fs] = audioread(wavepath);
        
        % Convert to mono (take the first channel if multi-channel)
        if size(y, 2) > 1
            y = y(:, 1);
        end
        
        %%Normalization
        x = double(y);
        if max(abs(x)) > 0
            x = x / max(abs(x));
        end
        
        %2. Parameter settings (automatically calculated based on sample rate)
        FrameLen = round(0.025 * Fs);  %Frame length: 25ms
        FrameInc = round(0.010 * Fs);   %Frame shift: 10ms
        
        amp1 = 10;    %High energy threshold
        amp2 = 0.5;   %Low energy threshold
        zcr2 = 1;        %Zero-crossing rate threshold
        
        maxsilence = 120;    %Maximum allowable silence duration within speech (number of frames)
        minlen = 15;           %Minimum allowable speech duration
        
        status = 0;         %State machine: 0-Silence, 1-Possible onset, 2-Speech segment, 3-End
        count = 0;
        silence = 0;
        x1 = 1; 	%Start frame index
        x2 = 1; 	%End frame index
        
        % 3.Calculate Short-time Zero-Crossing Rate (ZCR)
        %%Note: Logic preserved from original code; ensure 'enframe' function is in the path.
        tmp1 = enframe(x(1:end-1), hamming(FrameLen), FrameInc);
        tmp2 = enframe(x(2:end)  , hamming(FrameLen), FrameInc);
        signs = (tmp1 .* tmp2) < 0;
        diffs = (tmp1 - tmp2) > 0.02;
        zcr = sum(signs .* diffs, 2);
        
        %4. Calculate Short-time Energy (STE)
        amp = sum(abs(enframe(x, FrameLen, FrameInc)), 2);
        
        %%Dynamically adjust energy thresholds
        amp1 = min(amp1, max(amp)/4);
        amp2 = min(amp2, max(amp)/8);
        
        %5. Search for endpoints using the state machine
        for n = 1:length(zcr)
            switch status
                case {0, 1} 		%Silence or Possible Onset state
                    if amp(n) > amp1 		%Confirmed entry into speech segment
                        x1 = max(n - count - 1, 1);
                        status = 2;
                        silence = 0;
                        count = count + 1;
                    elseif amp(n) > amp2 || zcr(n) > zcr2     %Possible entry into speech segment
                        status = 1;
                        count = count + 1;
                    else % Remaining in silence state
                        status = 0;
                        count = 0;
                    end
                case 2 %Speech Segment state
                    if amp(n) > amp2 || zcr(n) > zcr2 	%Remaining in speech segment
                        count = count + 1;
                        silence = 0;
                    else %Possible end of speech
                        silence = silence + 1;
                        if silence < maxsilence
                            count = count + 1;
                        elseif count < minlen 	%Identified as noise
                            status = 0;
                            silence = 0;
                            count = 0;
                        else %Confirmed end of speech
                            status = 3;
                            x2 = x1 + (count - silence);
                        end
                    end
                case 3
                    break;
            end
        end
        
        %6. Map frame indices back to sample indices and save
        %%Calculate offset: frame index * frame shift = sample start position
        start_sample = max(1, (x1-1) * FrameInc + 1);
        end_sample = min(length(x), (x2-1) * FrameInc + FrameLen);
        
        if end_sample > start_sample
            yy = x(start_sample : end_sample);
            
            %%Automatically create the destination directory if it doesn't exist
            [save_dir, ~, ~] = fileparts(savepath);
            if ~exist(save_dir, 'dir')
                mkdir(save_dir);
            end
            
            audiowrite(savepath, yy, Fs);
            success = 1;
        else
            fprintf('Warning: No speech detected in %s\n', wavepath);
        end
        
    catch ME
        fprintf('Error processing %s: %s\n', wavepath, ME.message);
        success = 0;
    end
end