# Deep Neural Network Prediction 
Step 1. Genernate random parameters and Run them sequentially :
    $ python3 collect_data.py -gp -ep -pp -pl pooling -num 10 -shuffle -d 1080ti

Step 2. Data  Timeline parser :
    $ python3 preprocess_data.py -pt -pl pooling -d 1080ti

Step 3. Combine all raw data :
    $ python3 preprocess_data.py -c -pl pooling -d 1080ti

Step 4. Split raw data to train and test data as performance prediction inputs
    $ python3 preprocess_data.py -sp -pl pooling -d 1080ti

<!-- python3 -m dnn_prediction.data_collection.remove_duplicate_params -ipfp dnn_prediction/golden_struct_values/convolution_parameters.csv -pl convolution -->

Step 5. Train Model :

    python3 train_model.py -ftf ./utils/Feature_Target/conv_pre.json -log2file 1 -e 1000 -st 400 -sg 0.5 -lf maple -n perfnetA -pd 1080ti -pl convolution -psl pre
        
    python3 train_model.py -ftf ./utils/Feature_Target/conv_exe.json -log2file 1 -e 1000 -st 400 -sg 0.5 -lf maple -n perfnetA -pd 1080ti -pl convolution -psl exe
            
    python3 train_model.py -ftf ./utils/Feature_Target/conv_post.json -log2file 1 -e 1000 -st 400 -sg 0.5 -lf maple -n perfnetA -pd 1080ti -pl convolution -psl post


Step 6. Generate model csv file : 

    python3 verify_model.py -gmc --model lenet -b 1

Step 7. Predict model csv file : 

    python3 verify_model.py -pdm -n perfnetA -d 1080ti -l -eva -lf maple --model lenet -b 1

  Or you can use script to predict all models with all batches:

    python3 predict_script.py -n perfnetA -d 1080ti -lf maple

<!-- model path : model/convolution_1080ti/exe/perfnetA_malpe_abse, perfnetA_malpe_r2, perfnetA_malpe_re, perfnetA_malpe_rmse -->

<!-- 
Collect Data guideline:    collect_data_script.py -> guideline()
Preprocess Data guideline: collect_data_script.py -> guideline()
Train midel guideline:     train_model_script.py  -> guideline()
Verify  guideline:         verify_guideline.py    -> guideline()
Run Full model: Please Check run_full_tfnetwork.py 
-->
