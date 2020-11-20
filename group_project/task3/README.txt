If you want to evaluate the model and get the test results using our trained model/weights, please first download: 
https://drive.google.com/file/d/1eQueyHfl1Dh22TLFGRnCj9ZzlllnYpaj/view?usp=sharing

Then use command line:  python evaluate.py  --dataDir  /path/to/test/set/   --outputDir  /path/to/output/results/ 
                                                                       --weightsPath  /path/to/best_1000.h5 




If you want to train a new model , please first download a pretrained weights file from:
https://drive.google.com/file/d/1lKlnGTJ78XLRQrzg5vzdU85pvoJvUJGq/view?usp=sharing

Then you need to divide your dataset into a format like:
DataDir
    train
        plant<num>_rgb_.png
        plant<num>_label.png
        plant000_rgb_.png
        plant000_label.png
        ...
    crossVal
        plant<num>_rgb_.png
        plant<num>_label.png
        plant002_rgb_.png
        plant002_label.png
        ...

Then use command line: python train_cvppp.py --dataDir /path/to/your/dataset/train --outputDir /path/to/your/training/directory 
                                                                         --name YourModelName --numEpochs 5  
                                                                         --init /path/to/pretrainedl.h5