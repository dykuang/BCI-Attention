% function [] = make_plots_Depth()
    
    locations = load('/mnt/HDD/Datasets/SEED/channel_loc.mat').locations;

    % load data

%     Data = load('/mnt/HDD/Datasets/SEED/benchmark_summary/ATT_DK_7models.mat');
    Data = load('/mnt/HDD/Datasets/SEED/benchmark_summary/ATT_SEED_7models_S15.mat');
%     Data = load('/mnt/HDD/Benchmarks/DEAP/ATT_DEAP_5models_S12.mat');
%     Data = load('/mnt/HDD/Benchmarks/DEAP/ATT_DEAP_QKV_S24.mat');
    Data = Data.CM;
    ch_list = 'all';
%     ch_list = {'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', ...
%                'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', ...
%                'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', ...
%                'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', ...
%                'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',...
%                'Fz', 'Cz'
%                };  % DEAP: 1-22

%     ch_list = {'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', ...
%                'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', ...
%                'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', ...
%                'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', ...
%                'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8',...
%                'PO4', 'O2'
%                };  % DEAP: 23-32

    % nomralize to same range [-1, 1]
    Data = (Data - min(Data))./(max(Data) - min(Data))*2 -1;
    
%     figure(1)
%     plot_topography(ch_list,  double(mean(Data(1,:,:),3)), true, locations, true, false, 500);
%     figure(2)
%     plot_topography(ch_list,  double(std(Data(1,:,:),[], 3)), true, locations, true, false, 500);

    % plot the mean
%     for i = 1:7
%        figure(i)
%        plot_topography(ch_list,  double(mean(Data(i,:,:),3)), true, locations, true, false, 500);
%        colorbar('ylim', [-1, 1])
%     end

    % plot the std
    for i = 8:14
       figure(i)
       plot_topography(ch_list,  double(std(Data(i-7,:,:),[],3)), true, locations, true, false, 500);
%        colormap('hot')
%        colorbar('ylim', [0, 1])
    end

    
% end

function h = subplottight(n,m,i)
    [c,r] = ind2sub([m n], i);
    ax = subplot('Position', [(c-1)/m, 1-(r)/n, 1/m, 1/n]);
    if(nargout > 0)
      h = ax;
    end
end
