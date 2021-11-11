import os

def guideline():
    ### Verify model ###
    
    #Step1 Generate full model parameters:
    #$ python3 verify_model.py -mn lenet -b 1 -gm

    #Step1 Run Execute model parameters:
        # GPU: GPU my not run this step cause sess time is unuseful
        #$ python3 verify_model.py -mn lenet -b 1 -pd 1080ti -em 
    #CPU:
    #$ python3 verify_model.py -mn lenet -b 1 -pd E3-1275 -em -cpu

    #Step2 Run Profiler model parameters:
    #Only GPU is needed profile !!!:
    #$ python3 verify_model.py -mn lenet -b 1 -pd 1080ti -pm
    
    ### Step3: Combile All data
    #GPU:
    #$ python3 verify_model.py -mn lenet -b 1 -pd 1080ti -cm
    
    #CPU: Notice: Open -cpu tags!!
    #$ python3 verify_model.py -mn lenet -b 1 -pd E3-1275 -cm -cpu
    
    
    ### Step4: Verify model:
    # python3 verify_model.py -mn lenet -b 1 -pdm -magic_scaler 10 -pd 1080ti
    # python3 verify_model.py -mn lenet -b 1 -pdm -magic_scaler 10 -pd E3-1275
    #### Default Output Path is at: data_full_model/model_predict/xx_xxx_device.csv

    return 

def main():
    list_batch = [1, 2, 4, 8 , 16, 32, 64]
    list_model = ['lenet', 'alexnet', 'vgg16']
    for m in list_model:
        for b in list_batch:
            #str_ = 'python3 verify_model.py -mn {} -b {} -gm'.format(m, b)
            str_ = 'python3 verify_model.py -mn {} -b {} -pdm -tp clean_data_predict'.format(m, b)
            #print(str_)
            os.system(str_)    
    return 


if __name__ == '__main__':
    main()
