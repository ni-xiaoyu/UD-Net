Basic information of the code
1. Title: Identification and Spatio-Temporal Analysis of Ulva Prolifera in a Typical Coastal Area Using SAR Imageries
2. Key words: Ulva Prolifera, Coastal areas, Synthetic aperture radar, Deep learning
3. Author: Yanxia Wang, Xiaoyu Ni, Xiaoshuang Ma 
4. Contents of the main document: 1)  The Slove2.py is the master model, you can use the code to train all model; 2) The SegDataFolder.py is to processing data. Which include channel data, mean and std. If you want to train new data, you should pay attention on the channel data and the value of mean and std; 3) The getSetting.py is to change the hyper-parameter such as optimizer, scheduler, criterion, backbone et.al.