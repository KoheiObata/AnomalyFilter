import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import requests
import zipfile

from .dataset import Entity, Dataset


MACHINES = ['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-1-8',
            'machine-2-1', 'machine-2-2','machine-2-3','machine-2-4','machine-2-5','machine-2-6','machine-2-7','machine-2-8','machine-2-9',
            'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4','machine-3-5','machine-3-6','machine-3-7','machine-3-8', 'machine-3-9',
            'machine-3-10', 'machine-3-11']


# Data URIs
SMD_URL = 'https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset'
ANOMALY_ARCHIVE_URI = r'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip'
VALID_DATASETS = ['smd', 'anomaly_archive', 'iops', 'yahoo_real', 'yahoo_bench']

def download_file(filename:str, directory: str, source_url: str, decompress: bool = False) -> None:
    """Download data from source_ulr inside directory.
    Parameters
    ----------
    filename: str
        Name of file
    directory: str, Path
        Custom directory where data will be downloaded.
    source_url: str
        URL where data is hosted.
    decompress: bool
        Wheter decompress downloaded file. Default False.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    print('directory', directory)
    directory.mkdir(parents=True, exist_ok=True)

    filepath = Path(f'{directory}/{filename}')

    # Streaming, so we can iterate over the response.
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(source_url, stream=True, headers=headers)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte

    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filepath, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
            f.flush()
    t.close()

    size = filepath.stat().st_size

    if decompress:
        if '.zip' in filepath.suffix:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(directory)
        else:
            from patoolib import extract_archive
            extract_archive(str(filepath), outdir=directory)

def load_data(dataset: str, group: str, entities: Union[str, List[str]], downsampling: float=None, min_length: float=None, root_dir:str='./data', normalize:bool=True, verbose:bool=True, validation:bool=False):
    """Function to load TS anomaly detection datasets.
    Parameters
    ----------
    dataset: str
        Name of the dataset.
    group: str
        The train or test split.
    entities: Union[str, List[str]]
        Entities to load from the dataset.
    downsampling: Optional[float]
        Whether and the extent to downsample the data.
    root_dir: str
        Path to the directory where the datasets are stored.
    normalize: bool
        Whether to normalize Y.
    verbose: bool
        Controls verbosity
    """
    if dataset == 'smd':
        return load_smd(group=group, machines=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'anomaly_archive':
        return load_anomaly_archive(group=group, datasets=entities, downsampling=downsampling, min_length=min_length, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset == 'iops':
        return load_iops(group=group, filename=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    elif dataset in ['yahoo_real','yahoo_bench']:
        return load_yahoo(group=group, filename=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose, validation=validation)
    else:
        raise ValueError(f'Dataset must be one of {VALID_DATASETS}, but {dataset} was passed!')

def load_smd(group, machines=None, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    # NOTE: The SMD dataset is normalized and therefore we do not need normalize it further. The normalize parameter is for input compatibility.
    if machines is None:
        machines = MACHINES

    if isinstance(machines, str):
        machines = [machines]

    root_dir = f'{root_dir}/ServerMachineDataset'

    # Download data
    for machine in machines:

        if not os.path.exists(f'{root_dir}/train/{machine}.txt'):

            print('downloading SMD train')
            download_file(filename=f'{machine}.txt',
                          directory=f'{root_dir}/train',
                          source_url=f'{SMD_URL}/train/{machine}.txt')

            print('downloading SMD test')
            download_file(filename=f'{machine}.txt',
                          directory=f'{root_dir}/test',
                          source_url=f'{SMD_URL}/test/{machine}.txt')

            print('downloading SMD test label')
            download_file(filename=f'{machine}.txt',
                          directory=f'{root_dir}/test_label',
                          source_url=f'{SMD_URL}/test_label/{machine}.txt')

    # Load train data
    if group=='train':
        entities, entities_val = [], []
        for machine in machines:
            name = 'smd-train'
            name_val = 'smd-val'
            train_file = f'{root_dir}/train/{machine}.txt'
            Y = np.loadtxt(train_file, delimiter=',').T

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape

                right_padding = downsampling - n_t%downsampling
                Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)

            if validation:
                train_length = int(Y.shape[1]*0.9)
                entity = Entity(Y=Y[:, :train_length], name=machine, verbose=verbose)
                entities.append(entity)
                entity_val = Entity(Y=Y[:, train_length:], name=machine, verbose=verbose)
                entities_val.append(entity_val)
            else:
                entity = Entity(Y=Y, name=machine, verbose=verbose)
                entities.append(entity)

        if validation:
            smd = Dataset(entities=entities, name=name, verbose=verbose)
            smd_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)
            return smd, smd_val
        else:
            smd = Dataset(entities=entities, name=name, verbose=verbose)
            return smd

    # Load test data
    elif group=='test':
        entities = []
        for machine in machines:
            name = 'smd-test'
            test_file = f'{root_dir}/test/{machine}.txt'
            label_file = f'{root_dir}/test_label/{machine}.txt'

            Y = np.loadtxt(test_file, delimiter=',').T
            labels = np.loadtxt(label_file, delimiter=',')

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape
                right_padding = downsampling - n_t%downsampling

                Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                labels = np.pad(labels, (right_padding, 0))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
                labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)

            labels = labels[None, :]
            entity = Entity(Y=Y, name=machine, labels=labels, verbose=verbose)
            entities.append(entity)

        smd = Dataset(entities=entities, name=name, verbose=verbose)
        return smd

def download_anomaly_archive(root_dir='./data'):
    """Convenience function to download the Timeseries Anomaly Archive datasets
    """
    # Download the data
    download_file(filename=f'AnomalyArchive',
                directory=root_dir,
                source_url=ANOMALY_ARCHIVE_URI,
                decompress=True)

    # Reorganising the data
    shutil.move(src=f'{root_dir}/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData',
                dst=root_dir)
    os.remove(os.path.join(root_dir, 'AnomalyArchive'))
    shutil.rmtree(os.path.join(root_dir, 'AnomalyDatasets_2021'))
    shutil.move(src=f'{root_dir}/UCR_Anomaly_FullData',
                dst=f'{root_dir}/AnomalyArchive')

def load_anomaly_archive(group, datasets=None, downsampling=None, min_length=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    if not os.path.exists(f'{root_dir}/AnomalyArchive/'): download_anomaly_archive(root_dir=root_dir)

    ANOMALY_ARCHIVE_ENTITIES = ['_'.join(e.split('_')[:4]) for e in os.listdir(os.path.join(root_dir, 'AnomalyArchive'))]
    ANOMALY_ARCHIVE_ENTITIES = sorted(ANOMALY_ARCHIVE_ENTITIES)

    if datasets is None: datasets = ANOMALY_ARCHIVE_ENTITIES
    if verbose: print(f'Number of datasets: {len(datasets)}')

    entities, entities_val = [], []
    for file in os.listdir(os.path.join(root_dir, 'AnomalyArchive')):

        downsampling_entity = downsampling
        if '_'.join(file.split('_')[:4]) in datasets or file.split('_')[0]==datasets or file.split('_')[2]==datasets:
            with open(os.path.join(root_dir, 'AnomalyArchive',  file)) as f:
                Y = f.readlines()
                if len(Y) == 1:
                    Y = Y[0].strip()
                    Y = np.array([eval(y) for y in Y.split(" ") if len(y) > 1]).reshape((1, -1))
                elif len(Y) > 1:
                    Y = np.array([eval(y.strip()) for y in Y]).reshape((1, -1))

            fields = file.split('_')
            meta_data = {
                    'name': '_'.join(fields[:4]),
                    'train_end': int(fields[4]),
                    'anomaly_start_in_test': int(fields[5])-int(fields[4]),
                    'anomaly_end_in_test': int(fields[6][:-4])-int(fields[4]),
                }
            if verbose:
                print(f'Entity meta-data: {meta_data}')

            if normalize:
                Y_train = Y[0, 0:meta_data['train_end']].reshape((-1, 1))
                scaler = MinMaxScaler()
                scaler.fit(Y_train)
                Y = scaler.transform(Y.T).T

            n_time = Y.shape[-1]
            len_train = meta_data['train_end']
            len_test = n_time - len_train

            # No downsampling if n_time < min_length
            if (downsampling_entity is not None) and (min_length is not None):

                if (len_train//downsampling_entity < min_length) or (len_test//downsampling_entity < min_length):
                    downsampling_entity = None

            if group == 'train':
                name = f"{meta_data['name']}-train"
                name_val = f"{meta_data['name']}-val"
                Y = Y[0, 0:meta_data['train_end']].reshape((1, -1))

                # Downsampling
                if downsampling_entity is not None:
                    n_features, n_t = Y.shape

                    right_padding = downsampling_entity - n_t%downsampling_entity
                    Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                    Y = Y.reshape(n_features, Y.shape[-1]//downsampling_entity, downsampling_entity).max(axis=2)

                if validation:
                    train_length = int(Y.shape[1]*0.9)
                    entity = Entity(Y=Y.reshape((1, -1))[:, :train_length], name=meta_data['name'], verbose=verbose)
                    entities.append(entity)
                    entity_val = Entity(Y=Y.reshape((1, -1))[:, train_length:], name=meta_data['name'], verbose=verbose)
                    entities_val.append(entity_val)
                else:
                    entity = Entity(Y=Y.reshape((1, -1)), name=meta_data['name'], verbose=verbose)
                    entities.append(entity)


            elif group == 'test':
                name = f"{meta_data['name']}-test"
                Y = Y[0, meta_data['train_end']+1:].reshape((1, -1))

                # Label the data
                labels = np.zeros(Y.shape[1])
                labels[meta_data['anomaly_start_in_test']:meta_data['anomaly_end_in_test']] = 1

                # Downsampling
                if downsampling_entity is not None:
                    n_features, n_t = Y.shape
                    right_padding = downsampling_entity - n_t%downsampling_entity

                    Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                    labels = np.pad(labels, (right_padding, 0))

                    Y = Y.reshape(n_features, Y.shape[-1]//downsampling_entity, downsampling_entity).max(axis=2)
                    labels = labels.reshape(labels.shape[0]//downsampling_entity, downsampling_entity).max(axis=1)

                labels = labels[None, :]
                entity = Entity(Y=Y.reshape((1, -1)), name=meta_data['name'], labels=labels, verbose=verbose)
                entities.append(entity)


    if validation:
        data = Dataset(entities=entities, name=name, verbose=verbose)
        data_val = Dataset(entities=entities_val, name=name_val, verbose=verbose)
        return data, data_val
    else:
        data = Dataset(entities=entities, name=name, verbose=verbose)
        return data

def load_iops(group, filename, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    root_dir = f'{root_dir}/IOPS/{filename}'

    if group == 'train':
        df = pd.read_csv(f'{root_dir}.train.out', header=None, names=['Value', 'Label'])
        Y = np.array(df['Value']).reshape(1,-1)

        name = f'{filename}-train'
        name_val = f'{filename}-val'
        if normalize:
            scaler = MinMaxScaler()
            scaler.fit(Y.T)
            Y = scaler.transform(Y.T).T

        # Downsampling
        if downsampling is not None:
            n_features, n_t = Y.shape
            right_padding = downsampling - n_t%downsampling
            Y = np.pad(Y, ((0,0), (right_padding, 0) ))
            Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)


        if validation:
            train_length = int(Y.shape[1]*0.9)
            entity = Entity(Y=Y[:, :train_length], name=name, verbose=verbose)
            entity_val = Entity(Y=Y[:, train_length:], name=name_val, verbose=verbose)
            data = Dataset(entities=[entity], name=name, verbose=verbose)
            data_val = Dataset(entities=[entity_val], name=name_val, verbose=verbose)
            return data, data_val
        else:
            entity = Entity(Y=Y, name=name, verbose=verbose)
            data = Dataset(entities=[entity], name=name, verbose=verbose)
            return data


    elif group == 'test':
        df = pd.read_csv(f'{root_dir}.test.out', header=None, names=['Value', 'Label'])
        Y = np.array(df['Value']).reshape(1,-1)
        if normalize:
            df_train = pd.read_csv(f'{root_dir}.train.out', header=None, names=['Value', 'Label'])
            Y_train = np.array(df_train['Value']).reshape(1,-1)
            scaler = MinMaxScaler()
            scaler.fit(Y_train.T)
            Y = scaler.transform(Y.T).T

        name = f'{filename}-test'

        # Label the data
        labels = np.array(df['Label'])

        # Downsampling
        if downsampling is not None:
            n_features, n_t = Y.shape
            right_padding = downsampling - n_t%downsampling

            Y = np.pad(Y, ((0,0), (right_padding, 0) ))
            labels = np.pad(labels, (right_padding, 0))

            Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
            labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)

        labels = labels[None, :]
        entity = Entity(Y=Y, name=name, labels=labels, verbose=verbose)
        data = Dataset(entities=[entity], name=name, verbose=verbose)
        return data

def load_yahoo(group, filename, downsampling=None, root_dir='./data', normalize=True, verbose=True, validation=False):
    real_entity_list = ['A1real_10', 'A1real_11', 'A1real_12', 'A1real_13', 'A1real_15', 'A1real_16', 'A1real_17', 'A1real_19', 'A1real_1', 'A1real_20', 'A1real_21', 'A1real_22', 'A1real_23', 'A1real_24', 'A1real_25', 'A1real_26', 'A1real_27', 'A1real_28', 'A1real_29', 'A1real_2', 'A1real_30', 'A1real_31', 'A1real_32', 'A1real_33', 'A1real_34', 'A1real_37', 'A1real_38', 'A1real_39', 'A1real_3', 'A1real_40', 'A1real_41', 'A1real_42', 'A1real_43', 'A1real_45', 'A1real_46', 'A1real_47', 'A1real_4', 'A1real_50', 'A1real_51', 'A1real_52', 'A1real_53', 'A1real_55', 'A1real_56', 'A1real_57', 'A1real_58', 'A1real_60', 'A1real_61', 'A1real_62', 'A1real_63', 'A1real_65', 'A1real_66', 'A1real_67', 'A1real_6', 'A1real_7', 'A1real_8', 'A1real_9']
    bench_entity_list = ['A3Benchmark-TS100', 'A3Benchmark-TS10', 'A3Benchmark-TS11', 'A3Benchmark-TS12', 'A3Benchmark-TS13', 'A3Benchmark-TS14', 'A3Benchmark-TS15', 'A3Benchmark-TS16', 'A3Benchmark-TS18', 'A3Benchmark-TS19', 'A3Benchmark-TS1', 'A3Benchmark-TS20', 'A3Benchmark-TS21', 'A3Benchmark-TS22', 'A3Benchmark-TS23', 'A3Benchmark-TS24', 'A3Benchmark-TS25', 'A3Benchmark-TS26', 'A3Benchmark-TS27', 'A3Benchmark-TS28', 'A3Benchmark-TS29', 'A3Benchmark-TS2', 'A3Benchmark-TS30', 'A3Benchmark-TS31', 'A3Benchmark-TS32', 'A3Benchmark-TS33', 'A3Benchmark-TS35', 'A3Benchmark-TS36', 'A3Benchmark-TS37', 'A3Benchmark-TS39', 'A3Benchmark-TS3', 'A3Benchmark-TS40', 'A3Benchmark-TS41', 'A3Benchmark-TS42', 'A3Benchmark-TS43', 'A3Benchmark-TS44', 'A3Benchmark-TS45', 'A3Benchmark-TS46', 'A3Benchmark-TS47', 'A3Benchmark-TS48', 'A3Benchmark-TS49', 'A3Benchmark-TS4', 'A3Benchmark-TS50', 'A3Benchmark-TS51', 'A3Benchmark-TS53', 'A3Benchmark-TS54', 'A3Benchmark-TS55', 'A3Benchmark-TS56', 'A3Benchmark-TS58', 'A3Benchmark-TS59', 'A3Benchmark-TS5', 'A3Benchmark-TS60', 'A3Benchmark-TS61', 'A3Benchmark-TS62', 'A3Benchmark-TS63', 'A3Benchmark-TS64', 'A3Benchmark-TS65', 'A3Benchmark-TS66', 'A3Benchmark-TS67', 'A3Benchmark-TS68', 'A3Benchmark-TS69', 'A3Benchmark-TS6', 'A3Benchmark-TS70', 'A3Benchmark-TS71', 'A3Benchmark-TS72', 'A3Benchmark-TS73', 'A3Benchmark-TS75', 'A3Benchmark-TS76', 'A3Benchmark-TS77', 'A3Benchmark-TS78', 'A3Benchmark-TS79', 'A3Benchmark-TS7', 'A3Benchmark-TS80', 'A3Benchmark-TS81', 'A3Benchmark-TS82', 'A3Benchmark-TS83', 'A3Benchmark-TS84', 'A3Benchmark-TS85', 'A3Benchmark-TS86', 'A3Benchmark-TS87', 'A3Benchmark-TS88', 'A3Benchmark-TS89', 'A3Benchmark-TS8', 'A3Benchmark-TS90', 'A3Benchmark-TS91', 'A3Benchmark-TS92', 'A3Benchmark-TS93', 'A3Benchmark-TS94', 'A3Benchmark-TS95', 'A3Benchmark-TS96', 'A3Benchmark-TS97', 'A3Benchmark-TS98', 'A3Benchmark-TS99', 'A3Benchmark-TS9', 'A4Benchmark-TS100', 'A4Benchmark-TS10', 'A4Benchmark-TS11', 'A4Benchmark-TS12', 'A4Benchmark-TS13', 'A4Benchmark-TS14', 'A4Benchmark-TS15', 'A4Benchmark-TS17', 'A4Benchmark-TS19', 'A4Benchmark-TS1', 'A4Benchmark-TS20', 'A4Benchmark-TS21', 'A4Benchmark-TS22', 'A4Benchmark-TS23', 'A4Benchmark-TS24', 'A4Benchmark-TS25', 'A4Benchmark-TS26', 'A4Benchmark-TS27', 'A4Benchmark-TS28', 'A4Benchmark-TS29', 'A4Benchmark-TS2', 'A4Benchmark-TS30', 'A4Benchmark-TS31', 'A4Benchmark-TS32', 'A4Benchmark-TS33', 'A4Benchmark-TS34', 'A4Benchmark-TS35', 'A4Benchmark-TS36', 'A4Benchmark-TS37', 'A4Benchmark-TS38', 'A4Benchmark-TS39', 'A4Benchmark-TS3', 'A4Benchmark-TS40', 'A4Benchmark-TS41', 'A4Benchmark-TS42', 'A4Benchmark-TS43', 'A4Benchmark-TS44', 'A4Benchmark-TS45', 'A4Benchmark-TS46', 'A4Benchmark-TS47', 'A4Benchmark-TS48', 'A4Benchmark-TS49', 'A4Benchmark-TS4', 'A4Benchmark-TS50', 'A4Benchmark-TS52', 'A4Benchmark-TS55', 'A4Benchmark-TS56', 'A4Benchmark-TS57', 'A4Benchmark-TS58', 'A4Benchmark-TS59', 'A4Benchmark-TS5', 'A4Benchmark-TS60', 'A4Benchmark-TS61', 'A4Benchmark-TS62', 'A4Benchmark-TS63', 'A4Benchmark-TS64', 'A4Benchmark-TS65', 'A4Benchmark-TS67', 'A4Benchmark-TS68', 'A4Benchmark-TS69', 'A4Benchmark-TS6', 'A4Benchmark-TS70', 'A4Benchmark-TS71', 'A4Benchmark-TS72', 'A4Benchmark-TS73', 'A4Benchmark-TS74', 'A4Benchmark-TS75', 'A4Benchmark-TS76', 'A4Benchmark-TS77', 'A4Benchmark-TS78', 'A4Benchmark-TS79', 'A4Benchmark-TS7', 'A4Benchmark-TS80', 'A4Benchmark-TS81', 'A4Benchmark-TS82', 'A4Benchmark-TS84', 'A4Benchmark-TS85', 'A4Benchmark-TS86', 'A4Benchmark-TS87', 'A4Benchmark-TS88', 'A4Benchmark-TS89', 'A4Benchmark-TS8', 'A4Benchmark-TS90', 'A4Benchmark-TS91', 'A4Benchmark-TS92', 'A4Benchmark-TS93', 'A4Benchmark-TS94', 'A4Benchmark-TS95', 'A4Benchmark-TS97', 'A4Benchmark-TS98', 'A4Benchmark-TS99']
    head = 'Yahoo_' if filename in [*real_entity_list] else 'Yahoo'
    root_dir = f'{root_dir}/YAHOO'

    if group == 'train':
        df = pd.read_csv(f'{root_dir}/{head}{filename}_data.out', header=None, names=['Value', 'Label'])
        Y = np.array(df['Value']).reshape(1,-1)
        total_length = Y.shape[1]
        train_length = int(total_length*0.6)
        Y = Y[:, :train_length]

        name = f'{filename}-train'
        name_val = f'{filename}-val'
        if normalize:
            scaler = MinMaxScaler()
            scaler.fit(Y.T)
            Y = scaler.transform(Y.T).T

        # Downsampling
        if downsampling is not None:
            n_features, n_t = Y.shape
            right_padding = downsampling - n_t%downsampling
            Y = np.pad(Y, ((0,0), (right_padding, 0) ))
            Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)


        if validation:
            valid_length = int(total_length*0.1)
            entity = Entity(Y=Y[:, :train_length-valid_length], name=name, verbose=verbose)
            entity_val = Entity(Y=Y[:, train_length-valid_length:], name=name_val, verbose=verbose)
            data = Dataset(entities=[entity], name=name, verbose=verbose)
            data_val = Dataset(entities=[entity_val], name=name_val, verbose=verbose)
            return data, data_val
        else:
            entity = Entity(Y=Y, name=name, verbose=verbose)
            data = Dataset(entities=[entity], name=name, verbose=verbose)
            return data


    elif group == 'test':
        df = pd.read_csv(f'{root_dir}/{head}{filename}_data.out', header=None, names=['Value', 'Label'])
        Y = np.array(df['Value']).reshape(1,-1)
        total_length = Y.shape[1]
        train_length = int(total_length*0.6)
        Y = Y[:, train_length:]

        if normalize:
            df_train = pd.read_csv(f'{root_dir}/{head}{filename}_data.out', header=None, names=['Value', 'Label'])
            Y_train = np.array(df_train['Value']).reshape(1,-1)
            Y_train = Y_train[:, :train_length]
            scaler = MinMaxScaler()
            scaler.fit(Y_train.T)
            Y = scaler.transform(Y.T).T

        name = f'{filename}-test'

        # Label the data
        labels = np.array(df['Label'])
        labels = labels[train_length:]

        # Downsampling
        if downsampling is not None:
            n_features, n_t = Y.shape
            right_padding = downsampling - n_t%downsampling

            Y = np.pad(Y, ((0,0), (right_padding, 0) ))
            labels = np.pad(labels, (right_padding, 0))

            Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
            labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)

        labels = labels[None, :]
        entity = Entity(Y=Y, name=name, labels=labels, verbose=verbose)
        data = Dataset(entities=[entity], name=name, verbose=verbose)
        return data