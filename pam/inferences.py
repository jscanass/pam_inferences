import os
import argparse
import yaml
#import librosa
import numpy as np
import pandas as pd
from maad import util
#from joblib import load
#from librosa.feature import melspectrogram, mfcc
#from tensorflow import keras
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from time import time
import torch
import torchaudio
from torchvision.transforms import Resize

from torch import nn
from pam.models.templates_models.torch_models import ResNetClassifier


def preprocessing_metadata(data_folder, 
                           window_size,
                           sliding_window, 
                           save=False, 
                           verbose=False):
    
    # Get audio metadata
    print('Extracting metadata in folder ', 'data/'+data_folder)
    df = util.get_metadata_dir('data/'+data_folder, verbose=verbose)
    print('Metadata obtained. Number of .wav files:',len(df))
    print('Count of lenghts:\n',
            df['length'].value_counts(dropna=False))
    # remove nan values
    df = df.loc[~df.sample_rate.isna(),:]
    print('Number of .wav files withot empty files:', len(df))
    
    # metadata with samples of fixed length and sliding window
    df_sw_all = pd.DataFrame()
    for value in df.length.unique():
        max_value = int(value-window_size)
        # generate interval of audio files
        interval = list(range(0, max_value,sliding_window)) + [value-window_size]
        df_val = df[df['length']==value]
        n_files = len(df_val)
        print(f'Preprocessing {n_files} .wav files  with lenght {value:.3f}')
        df_sw_val = pd.DataFrame(np.repeat(df_val.values,
                                            len(interval),
                                            axis=0),
                                            columns = df_val.columns )
        df_sw_val['min'] = interval*df_val.shape[0]
        df_sw_val['max'] = df_sw_val['min'] + window_size
        df_sw_all = pd.concat([df_sw_all, df_sw_val])  
    print('Inferences samples:', len(df_sw_all))
    
    if save:
        folder_name = data_folder.split('data/')[-1][:-1].replace('/', '_')
        if not os.path.exists('results/' + folder_name):
            os.makedirs('results/' + folder_name)
        df.to_parquet('results/' + folder_name +
                      'samples.parquet.gzip' ,
              compression='gzip')  
            
    return df_sw_all

'''
def inference_df_cnn(audio_path, trained_model, start_second, window_size):
    
    trained_model = keras.models.load_model(trained_model)

    try:
        s, fs = librosa.load(path=audio_path,offset=start_second,duration=window_size)
        S = melspectrogram(y=s,sr=fs, n_fft=1024, 
                            hop_length=256, n_mels=128, 
                            power=1.0, fmin = 50, fmax=4000)
        S = np.reshape(S, (1,S.shape[0],S.shape[1],1))
        y_prob = trained_model.predict(S)
        keras_inference = y_prob.argmax(axis=-1)
        return (keras_inference[0])
    except:
        return (None)
'''

'''
def inference_df_gbc(audio_path, trained_model, start_second, window_size):
    
    sklearn_clfs = load(trained_model)
    try:
        s, fs = librosa.load(path=audio_path,offset=start_second,duration=window_size)
       
        mfcc_feature = mfcc(y=s, sr=fs, n_mfcc=20, n_fft=1024, 
                        win_length=1024, hop_length=512, htk=True)                
        mfcc_feature = np.median(mfcc_feature, axis=1)  
        sklearn_inference = trained_model.predict(mfcc_feature.reshape(-1,1).T)
        return (sklearn_inference[0])
    except:
        return (None)
'''

def inference_df_torch(audio_path, 
                    trained_model, 
                    start_second, 
                    window_size,
                    device='cpu',
                    sample_rate=22050
                    # TO DO: hyperparameters_dict
                    ):
    
    try:
        waveform, sample_rate = torchaudio.load(audio_path, 
                                        frame_offset=sample_rate*start_second, 
                                        num_frames=sample_rate*window_size)


        device = device # from hyperparameters_dict
        sample_rate = sample_rate # from hyperparameters_dict

        sigmoid = nn.Sigmoid()

        # load back the model
        
        trained_model.eval()

        # test transformations
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=128,
            n_mels=128
        )
        test_transform = nn.Sequential(                           # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
                mel_spectrogram,                                            # convert to a spectrogram
                torchaudio.transforms.AmplitudeToDB(),
                Resize([224, 448]),
                                    )
                                    
        input = test_transform(waveform)
        # 2, 224, 448 -> 1, 1, 224, 448
        input = input[0].unsqueeze_(0).unsqueeze_(0)
        
        inference = sigmoid(trained_model(input))
        return inference.tolist()#.detach().numpy()
    except Exception as e: 
        #print('File:',start_second, audio_path)
        #print(e)
        return [None]*42

def main():
    
    # Read parameters of inference
    parser = argparse.ArgumentParser(description='Inferences')
    parser.add_argument('--config', help='Path to config file', default='configs/example.yaml')
    args = parser.parse_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    window_size = cfg['window_size']
    sliding_window = cfg['sliding_window']
    data_folder = cfg['data_folder']
    trained_model_path = cfg['trained_model']
    
    # Preprocecessing audio data to dataframe with inference samples

    audio_path = 'data/Local1(Orleans)/Visita1/1_gravador/memoria_A/INCT20955_20190830_230000.wav'
    device = 'cpu'
    model_instance = ResNetClassifier(model_type='resnet152',
                                ).to(device)
    state_dict = torch.load(trained_model_path)
    model_instance.load_state_dict(state_dict)
    
    df = preprocessing_metadata(data_folder,window_size,sliding_window)
    # df = df.sample(10000)

    ProgressBar().register()
    ddf = dd.from_pandas(df, npartitions=8)

    # Apply inferences over all samples using CNN model
    
    t0 = time()
    ddf['inference'] = ddf.apply(lambda x: inference_df_torch(x['path_audio'],
                                                                model_instance, 
                                                                x['min'],
                                                                window_size,
                                                                ),
                                                                axis=1,
                                                                #result_type='expand'
                                                                )
    # Convert Dask DataFrame back to Pandas DataFrame
    df = ddf.compute()
    t1 = time()    
    execution_time = str(round(t1-t0,1))
    print('-------------->>>>> Results for Dask ' + execution_time)     
    
    df['visita'] = df['path_audio'].apply(lambda x:x.split('/')[1])
    df['fname'] = df['path_audio'].apply(lambda x:x.split('/')[-1])

    df['fname'] = df['fname'].str.split(pat='.').str[0]
    df[['site','date']] = df['fname'].str.split(pat='_',n=1,expand=True)
    df['date'] = df['date'].str.split('_').apply(lambda x: x[0]+x[1])
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = df['date'].dt.time
    df['day'] = df['date'].dt.date

    folder_name = data_folder.split('data/')[-1][:-1].replace('/', '_')
    if not os.path.exists('results/' + folder_name):
        os.makedirs('results/' + folder_name)
    df.to_parquet('results/' + folder_name +
                    'inferences_torch.parquet.gzip' ,
                compression='gzip')
    print('Results saved in: results/' + folder_name +
                    'inferences_torch.parquet.gzip')

''' 
if __name__ == "__main__":
    
    main()
    
    """
    # Other functions for speed up inference
    
    ## joblib
    
    from joblib import Parallel, delayed, load

    dir_wav_files = []
    inference_results = []
    wrong = []

    pathlist = Path(folder).glob('**/*.wav')
    start = time()
    inference_results_all = Parallel(n_jobs=8,backend="threading")(delayed(chorus_inference)(i) for i in list(pathlist))

    ###
    with tqdm_joblib(tqdm(desc="My calculation", total=30007)) as progress_bar:
        pathlist = Path(folder).glob('**/*.wav')
        Parallel(n_jobs=8,backend="threading")(delayed(save_chorus)(i) for i in list(pathlist))
    end = time()
    ###
    
    inference_results_all = [item for sublist in inference_results_all for item in sublist]
    print(end-start)
    df = pd.DataFrame(inference_results_all,
                        columns=['dir','lengt','min','max',
                                    'random_inf','gbrt_inf','cnn_inf'])
    
    ###
    
    start = time()

    for i in tqdm(list(pathlist)):
        print(i)
        save_chorus(i)
        
    end = time()
    print(end-start)

    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object):
    # Context manager to patch joblib to report into tqdm progress bar given as argument
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()
    ###
    
    ## swifter
    import swifter
    df[['inference_gbc','inference_cnn']] = df.swifter.progress_bar(True).apply(lambda x: inference_df(x['path_audio'], 
                                                                                     x['min']), axis=1,result_type='expand')
    df.to_csv('results/inferences_Local1(Orleans)_s.csv',index=False)
    t2 = time()                                                                     
    print('-------------------------->>>>> Results for swifter:',round(t2-t1,3))

    tqdm.pandas()

    df[['inference_gbc','inference_cnn']] = df.progress_apply(lambda x: inference_df(x['path_audio'], 
                                                                                     x['min']), axis=1,result_type='expand')
    df.to_csv('results/inferences_Local1(Orleans)_p.csv',index=False)
    t3 = time()    
    print('-------------------------->>>>> Results for pandas',round(t3-t2,3))
'''