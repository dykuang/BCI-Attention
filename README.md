# BCI-Attention
A repository including multiple attentional modules designs for brain signal recognition tasks.
Scripts should run with `tensorflow>=2.8.0`

------

### File Briefs  
  ##### Training
`cv_exp.sh`: Bash script for calling `exp_5CV_SEED.py` with data from different subjects in SEED dataset.  
`DEAP_benchmark.sh`: Bash script for calling `DEAP_test_gen.py` specifying different subjects, experiment types and models with DEAP dataset.  
`exp_5CV_SEED.py`: Main script for training with SEED dataset. One should manually change models used in the script.  
`DEAP_test_gen.py`: Main script for training with DEAP dataset.  

  ##### Explorations, Analysis and Visualizations 
  These are less organised scripts. They were written as our experiments goes. So you probably will not use all contents in a given script, but just some parts of it for your purpose. 
  
  `DEAP_Plots.py`: Gather metrics from trained models with DEAP for benchmarking, analysis and make some visualizations.  
  `SEED_extra_plot.py`: Gather metrics from trained models with SEED for benchmarking, analysis and make some visualizations.  
  `sigma_effect.py`: Scripts for exploring the Kernel Attention Module (KAM)'s effect on backbone network with SEED dataset.
  
  ##### Other supporting files  
  The folder `scalp_map` contains matlab files for making scalp maps.  
  `Modules.py`: Contains custom layers used for constructing network models.  
  `Models.py`: Contains different network models.
   `Utils.py`: Some utility functions
   `Visual.py`: Some function for general visualization purpose. 
   
------
### Demo Plots
These figures are:  
  * The prediction transition curve under different models given the same input samples from SEED dataset. 
  * Track of accuracies when different frequencies are filtered out from input samples.  
  * The map from deep gram matrix's element to attention matrix's element learned by MCAM module
 <!-- ![Image](demo_plots/PTC_SEED_KAM.png?raw=true "Title") -->
<img src="demo_plots/PTC_SEED_KAM.png" style=" width:960px ; height:240px "><img src="demo_plots/KAM_DEAP_freq_S24_rej.png" style=" width:360px ; height:240px " ><img src="demo_plots/mono_trace_S24.png" style=" width:480px ; height:200px "  >
