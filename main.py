import argparse
import numpy as np
import os
import glob
import time

import torch
import torch.nn as nn


from diffwave_csdi import diff_CSDI
import utils
import datautils


def create_mask(size, way='random'):
    if way=='random':
        mask = torch.rand(size)
        mask = mask>0.5
    elif way=='all_true':
        mask = torch.rand(size)
        mask = mask>-1
    elif way=='all_false':
        mask = torch.rand(size)
        mask = mask<-1
    elif 'bernoulli' in way:
        # 'bernoulli_0.5'
        try:
            p = float(way.split('_')[1]) #the larger p, the larger the noise (ratio of True(1))
        except:
            p = 0.5
        if len(size)==3:
            B, W = size[0], size[1]*size[2]
        mask = generate_binomial_mask(B, W, p=p)
        if len(size)==3: mask = mask.reshape(B,size[1],size[2])
    return mask

def generate_binomial_mask(B, W, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, W))).to(torch.bool)


def time_embedding(pos, d_model=128):
    pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
    position = pos.unsqueeze(2)
    div_term = 1 / torch.pow(
        10000.0, torch.arange(0, d_model, 2) / d_model
    )
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return pe

def get_side_info(base_shape, params):
    B, K, L = base_shape
    observed_tp = torch.arange(L).unsqueeze(0).repeat(B,1) # (B,L)

    time_embed = time_embedding(observed_tp, params.emb_time_dim)  # (B,L,emb)
    time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
    embed_layer = nn.Embedding( num_embeddings=params.n_features, embedding_dim=params.emb_feature_dim)
    feature_embed = embed_layer( torch.arange(params.n_features))  # (K,emb)
    feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

    side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
    side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)
    return side_info

class AnomalyFilter:
    def __init__(
        self,
        model_dir = "./training",
        params = None,
        device = 'cpu',
    ):

        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir

        self.params = params
        self.epoch = params.epoch
        self.device = device

        self.autocast = torch.cuda.amp.autocast()
        self.scaler = torch.cuda.amp.GradScaler()

        self.model = diff_CSDI(self.params).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.params.lr)

        self.beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - self.beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.noise_level = self.noise_level.to(self.device)
        self.loss_fn = nn.L1Loss()


    def train(self, train_dataloader, val_dataloader=None):
        stop_counter = 0
        best_val_loss = np.inf
        time_list, train_loss_list, val_loss_list = [],[],[]

        for epoch in range(self.epoch):
            self.model.train()
            starttime = time.time()
            cum_loss, step_count = 0, 0
            for step, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                inputs = batch['Y']
                inputs = inputs.to(self.device)

                # ------ make noise-------------
                B, D, W = inputs.shape #(batch, n_features, window)

                t_diffusions = torch.randint(0, self.params.diffusion_steps, [B], device=self.device)

                noise_scale = self.noise_level[t_diffusions].unsqueeze(1).unsqueeze(2).to(self.device) # select t_diffusions level noise
                noise = torch.randn_like(inputs).to(self.device) #Gaussian with shape of inputs
                mask = create_mask(inputs.size(), way=self.params.mask).to(self.device) #add noise to True(1)
                noise = noise*mask

                noisy_input = (noise_scale**0.5) * inputs + (1.0 - noise_scale)**0.5 * noise
                # original signal is slowly "fading out". The noise is slowly "fading in".
                noisy_input = noisy_input.to(self.device)

                with self.autocast:
                    cond_info = get_side_info((B,D,W), self.params).to(self.device)
                    predicted = self.model(x=noisy_input, cond_info=cond_info, diffusion_step=t_diffusions)
                    loss = self.loss_fn(noise, predicted)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at epoch {epoch}')

                cum_loss += loss.item()
                step_count += 1
            epoch_loss = cum_loss/step_count
            epoch_t = time.time() - starttime
            print('Epoch:', epoch, '     loss: ', str(epoch_loss)[0:6], '     time: ', str(epoch_t)[0:4], 'sec')
            time_list.append(epoch_t)
            train_loss_list.append(epoch_loss)

            # early stop
            if val_dataloader:
                val_loss = self.validation(val_dataloader)
                val_loss_list.append(val_loss.item())
                if val_loss < best_val_loss:
                    stop_counter = 0
                    best_val_loss = val_loss
                    print("best validation loss is updated", str(best_val_loss.item())[:6])
                    torch.save(self.model.state_dict(),f'{self.model_dir}/bestmodel.pkl')
                else:
                    stop_counter += 1

            np.savetxt(f'{self.model_dir}/time.txt',np.array(time_list),fmt='%.4e')
            np.savetxt(f'{self.model_dir}/train_loss.txt',np.array(train_loss_list),fmt='%.4e')
            np.savetxt(f'{self.model_dir}/valid_loss.txt',np.array(val_loss_list),fmt='%.4e')

            if val_dataloader and stop_counter > 10:
                break
        #################################################################################

    def validation(self, val_dataloader):
        loss = torch.tensor([0.0], requires_grad=False, device=self.device)
        diffusion_steps = self.params.diffusion_steps

        #validate each diffusion steps at least 10 times
        ideal_sample = diffusion_steps*10
        current_sample = self.params.batch_size*len(val_dataloader)
        valid_iter = 1 if ideal_sample < current_sample else min([10, int(ideal_sample/current_sample)])

        with torch.no_grad():
            self.model.eval()
            for t in range(valid_iter):
                for batch in val_dataloader:
                    inputs = batch['Y']
                    inputs = inputs.to(self.device)
                    B, D, W = inputs.shape #(batch, n_features, window)
                    original_inputs = inputs.clone().detach()

                    t_diffusions = torch.randint(0, diffusion_steps, [B], device=self.device)
                    noise_scale = self.noise_level[t_diffusions].unsqueeze(1).unsqueeze(2).to(self.device) # select t_diffusions level noise
                    noise = torch.randn_like(inputs).to(self.device) #Gaussian with shape of inputs
                    mask = create_mask(inputs.size(), way=self.params.mask).to(self.device) #add noise to True(1)
                    noise = noise*mask
                    noisy_input = (noise_scale**0.5) * inputs + (1.0 - noise_scale)**0.5 * noise
                    noisy_input = noisy_input.to(self.device)

                    cond_info = get_side_info((B,D,W), self.params).to(self.device)
                    predicted = self.model(x=noisy_input, cond_info=cond_info, diffusion_step=t_diffusions)
                    loss += self.loss_fn(noise, predicted)
            return loss


def test(test_dataloader, model_dir, params, device='cpu'):
    model = diff_CSDI(params).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/bestmodel.pkl'))

    diffusion_steps_i = params.diffusion_steps_i
    inputs_list = []
    prediction_list = []

    training_noise_schedule = np.array(params.noise_schedule)
    beta = training_noise_schedule
    alpha = 1 - beta #signal remaining after noise is added during INFERENCE
    alpha_cum = np.cumprod(alpha) #cumulative proportion of signal remaining after adding noise at each step during inference.

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = batch['Y']
            inputs = inputs.to(device)
            B, D, W = inputs.shape #(batch, n_features, window)
            original_inputs = inputs.clone().detach()

            T = np.array(np.arange(diffusion_steps_i), dtype=np.float32)

            if not params.reverse: #naive inference
                noise_scale = alpha_cum[diffusion_steps_i-1]*torch.ones((B,1,1)).to(device)
                noise = torch.randn_like(inputs).to(device) #Gaussian with shape of inputs
                inputs = (noise_scale**0.5) * inputs + (1.0 - noise_scale)**0.5 * noise #noisy input
            else: #noiseless inference
                noise_scale = alpha_cum[diffusion_steps_i-1]*torch.ones((B,1,1)).to(device)
                noise = torch.zeros_like(inputs).to(device) #Zero with shape of inputs
                inputs = (noise_scale**0.5) * inputs + (1.0 - noise_scale)**0.5 * noise #scaled noiseless input

            cond_info = get_side_info((B,D,W), params).to(device)
            for n in range(diffusion_steps_i-1, -1, -1):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                inputs = c1 * (inputs - c2 * model(x=inputs, cond_info=cond_info, diffusion_step=torch.tensor([T[n]], device=device)))
                if n > 0: # if not first step
                    if not params.reverse: #naive inference
                        noise = torch.randn_like(inputs) #use noise
                    else: #noiseless inference
                        noise = torch.zeros_like(inputs) #without noise
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    inputs += sigma * noise

            inputs_list.append(original_inputs)
            prediction_list.append(inputs)

        inputs_list = torch.cat(inputs_list, dim=0)
        inputs_list = inputs_list.to('cpu').detach().numpy().copy()
        prediction_list = torch.cat(prediction_list, dim=0)
        prediction_list = prediction_list.to('cpu').detach().numpy().copy()

    return inputs_list, prediction_list


def get_meta_data(entity):
    anomaly_data_dir = './dataset/AnomalyArchive'
    if not os.path.exists(f'{anomaly_data_dir}/'):
        import loaders
        loaders.load.download_anomaly_archive(root_dir='./dataset')
    for file in os.listdir(anomaly_data_dir):
        if '_'.join(file.split('_')[:4]) in entity or file.split('_')[0]==entity or file.split('_')[2]==entity:
            fields = file.split('_')
            meta_data = {
                    'name': '_'.join(fields[:4]),
                    'train_end': int(fields[4]),
                    'anomaly_start_in_test': int(fields[5])-int(fields[4]),
                    'anomaly_end_in_test': int(fields[6][:-4])-int(fields[4]),
                }
            print(meta_data)
            return meta_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default='anomaly_archive', help='The dataset name, [anomaly_archive, beatgan_ecg, smd]')
    parser.add_argument('--entities', type=str, default='0', help='[machine-1-1, ...]')

    parser.add_argument('--downsampling', type=int, default=1, help='(defaults to 1)')
    parser.add_argument('--batch-size', type=int, default=128, help='The batch size (defaults to 128)')
    parser.add_argument('--window-size', type=int, default=100, help='The window size (defaults to 100)')
    parser.add_argument('--window-step', type=int, default=1, help='The sliding window (defaults to 1)')

    # Learning
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--epoch', type=int, default=1, help='The number of epochs')

    # Diffusion Model
    parser.add_argument('--diffusion-steps', type=int, default=50, help='forward diffusion steps(defaults to 50)')
    parser.add_argument('--diffusion-steps-i', type=int, default=50, help='reverse diffusion steps(defaults to 50)')
    parser.add_argument('--beta-start', type=float, default=0.0001, help='(defaults to 1e-4)')
    parser.add_argument('--beta-end', type=float, default=0.01, help='(defaults to 0.01)')

    # Architecture
    parser.add_argument('--residual-layers', type=int, default=8, help='(defaults to 8)')
    parser.add_argument('--residual-channels', type=int, default=64, help='(defaults to 64)')
    parser.add_argument('--nheads', type=int, default=8, help='(defaults to 8)')

    # Module
    parser.add_argument('--mask', type=str, default='bernoulli', help='all_true, all_false, bernoulli_x (x is the ratio of noise)')
    parser.add_argument('--reverse', action="store_false", default=True, help='if True: Noiseless Inference; if False: Naive Inference')

    # Computer
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--seed', type=int, default=0, help='The random seed')
    parser.add_argument('--run_name', type=str, default='test', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')

    args = parser.parse_args()
    print("Arguments:", str(args))

    device = utils.init_dl_program(args.gpu, seed=args.seed)
    print('Device', device)

    if args.dataset == 'anomaly_archive':
        n_features = 1
        args.batch_size = 128
        args.window_size = 100
        args.window_step = 1
        #for all subdataset
        entity_list = [str(i).zfill(3) for i in range(1,251)]
        #for convenience
        entity_list = ['028']
    elif args.dataset == 'smd':
        n_features = 38
        args.batch_size = 128
        args.window_size = 100
        args.window_step = 10
        entity_list = ["1-1","1-2","1-3","1-4","1-5","1-6","1-7","1-8","2-1","2-2","2-3","2-4","2-5","2-6","2-7","2-8","2-9","3-1","3-2","3-3","3-4","3-5","3-6","3-7","3-8","3-9","3-10","3-11"]
        entity_list = [f'machine-{entity}' for entity in entity_list]
    if args.dataset == 'iops':
        n_features = 1
        args.batch_size = 128
        args.window_size = 100
        args.window_step = 10
        entity_list = ['KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3', 'KPI-1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0', 'KPI-301c70d8-1630-35ac-8f96-bc1b6f4359ea', 'KPI-42d6616d-c9c5-370a-a8ba-17ead74f3114', 'KPI-43115f2a-baeb-3b01-96f7-4ea14188343c', 'KPI-431a8542-c468-3988-a508-3afd06a218da', 'KPI-4d2af31a-9916-3d9f-8a8e-8a268a48c095', 'KPI-54350a12-7a9d-3ca8-b81f-f886b9d156fd', 'KPI-55f8b8b8-b659-38df-b3df-e4a5a8a54bc9', 'KPI-57051487-3a40-3828-9084-a12f7f23ee38', 'KPI-6a757df4-95e5-3357-8406-165e2bd49360', 'KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd', 'KPI-6efa3a07-4544-34a0-b921-a155bd1a05e8', 'KPI-7103fa0f-cac4-314f-addc-866190247439', 'KPI-847e8ecc-f8d2-3a93-9107-f367a0aab37d', 'KPI-8723f0fb-eaef-32e6-b372-6034c9c04b80', 'KPI-9c639a46-34c8-39bc-aaf0-9144b37adfc8', 'KPI-a07ac296-de40-3a7c-8df3-91f642cc14d0', 'KPI-a8c06b47-cc41-3738-9110-12df0ee4c721', 'KPI-ab216663-dcc2-3a24-b1ee-2c3e550e06c9', 'KPI-adb2fde9-8589-3f5b-a410-5fe14386c7af', 'KPI-ba5f3328-9f3f-3ff5-a683-84437d16d554', 'KPI-c02607e8-7399-3dde-9d28-8a8da5e5d251', 'KPI-c69a50cf-ee03-3bd7-831e-407d36c7ee91', 'KPI-da10a69f-d836-3baa-ad40-3e548ecf1fbd', 'KPI-e0747cad-8dc8-38a9-a9ab-855b61f5551d', 'KPI-f0932edd-6400-3e63-9559-0a9860a1baa9', 'KPI-ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa']
    if args.dataset in ['yahoo_real','yahoo_bench']:
        n_features = 1
        args.batch_size = 128
        args.window_size = 100
        args.window_step = 1
        if args.dataset == 'yahoo_real': entity_list = ['A1real_10', 'A1real_11', 'A1real_12', 'A1real_13', 'A1real_15', 'A1real_16', 'A1real_17', 'A1real_19', 'A1real_1', 'A1real_20', 'A1real_21', 'A1real_22', 'A1real_23', 'A1real_24', 'A1real_25', 'A1real_26', 'A1real_27', 'A1real_28', 'A1real_29', 'A1real_2', 'A1real_30', 'A1real_31', 'A1real_32', 'A1real_33', 'A1real_34', 'A1real_37', 'A1real_38', 'A1real_39', 'A1real_3', 'A1real_40', 'A1real_41', 'A1real_42', 'A1real_43', 'A1real_45', 'A1real_46', 'A1real_47', 'A1real_4', 'A1real_50', 'A1real_51', 'A1real_52', 'A1real_53', 'A1real_55', 'A1real_56', 'A1real_57', 'A1real_58', 'A1real_60', 'A1real_61', 'A1real_63', 'A1real_65', 'A1real_66', 'A1real_67', 'A1real_6', 'A1real_7', 'A1real_8', 'A1real_9']
        if args.dataset == 'yahoo_bench': entity_list = ['A3Benchmark-TS100', 'A3Benchmark-TS10', 'A3Benchmark-TS11', 'A3Benchmark-TS12', 'A3Benchmark-TS13', 'A3Benchmark-TS14', 'A3Benchmark-TS15', 'A3Benchmark-TS16', 'A3Benchmark-TS18', 'A3Benchmark-TS19', 'A3Benchmark-TS1', 'A3Benchmark-TS20', 'A3Benchmark-TS21', 'A3Benchmark-TS22', 'A3Benchmark-TS23', 'A3Benchmark-TS24', 'A3Benchmark-TS25', 'A3Benchmark-TS26', 'A3Benchmark-TS27', 'A3Benchmark-TS28', 'A3Benchmark-TS29', 'A3Benchmark-TS2', 'A3Benchmark-TS30', 'A3Benchmark-TS31', 'A3Benchmark-TS32', 'A3Benchmark-TS33', 'A3Benchmark-TS35', 'A3Benchmark-TS36', 'A3Benchmark-TS37', 'A3Benchmark-TS39', 'A3Benchmark-TS3', 'A3Benchmark-TS40', 'A3Benchmark-TS41', 'A3Benchmark-TS42', 'A3Benchmark-TS43', 'A3Benchmark-TS44', 'A3Benchmark-TS45', 'A3Benchmark-TS46', 'A3Benchmark-TS47', 'A3Benchmark-TS48', 'A3Benchmark-TS49', 'A3Benchmark-TS4', 'A3Benchmark-TS50', 'A3Benchmark-TS51', 'A3Benchmark-TS53', 'A3Benchmark-TS54', 'A3Benchmark-TS55', 'A3Benchmark-TS56', 'A3Benchmark-TS58', 'A3Benchmark-TS59', 'A3Benchmark-TS5', 'A3Benchmark-TS60', 'A3Benchmark-TS61', 'A3Benchmark-TS62', 'A3Benchmark-TS63', 'A3Benchmark-TS64', 'A3Benchmark-TS65', 'A3Benchmark-TS66', 'A3Benchmark-TS67', 'A3Benchmark-TS68', 'A3Benchmark-TS69', 'A3Benchmark-TS6', 'A3Benchmark-TS70', 'A3Benchmark-TS71', 'A3Benchmark-TS72', 'A3Benchmark-TS73', 'A3Benchmark-TS75', 'A3Benchmark-TS76', 'A3Benchmark-TS77', 'A3Benchmark-TS78', 'A3Benchmark-TS79', 'A3Benchmark-TS7', 'A3Benchmark-TS80', 'A3Benchmark-TS81', 'A3Benchmark-TS82', 'A3Benchmark-TS83', 'A3Benchmark-TS84', 'A3Benchmark-TS85', 'A3Benchmark-TS86', 'A3Benchmark-TS87', 'A3Benchmark-TS88', 'A3Benchmark-TS89', 'A3Benchmark-TS8', 'A3Benchmark-TS90', 'A3Benchmark-TS91', 'A3Benchmark-TS92', 'A3Benchmark-TS93', 'A3Benchmark-TS94', 'A3Benchmark-TS95', 'A3Benchmark-TS96', 'A3Benchmark-TS97', 'A3Benchmark-TS98', 'A3Benchmark-TS99', 'A3Benchmark-TS9', 'A4Benchmark-TS100', 'A4Benchmark-TS10', 'A4Benchmark-TS11', 'A4Benchmark-TS12', 'A4Benchmark-TS13', 'A4Benchmark-TS14', 'A4Benchmark-TS15', 'A4Benchmark-TS17', 'A4Benchmark-TS19', 'A4Benchmark-TS1', 'A4Benchmark-TS20', 'A4Benchmark-TS21', 'A4Benchmark-TS22', 'A4Benchmark-TS23', 'A4Benchmark-TS24', 'A4Benchmark-TS25', 'A4Benchmark-TS26', 'A4Benchmark-TS27', 'A4Benchmark-TS28', 'A4Benchmark-TS29', 'A4Benchmark-TS2', 'A4Benchmark-TS30', 'A4Benchmark-TS31', 'A4Benchmark-TS32', 'A4Benchmark-TS33', 'A4Benchmark-TS34', 'A4Benchmark-TS35', 'A4Benchmark-TS36', 'A4Benchmark-TS37', 'A4Benchmark-TS38', 'A4Benchmark-TS39', 'A4Benchmark-TS3', 'A4Benchmark-TS40', 'A4Benchmark-TS41', 'A4Benchmark-TS42', 'A4Benchmark-TS43', 'A4Benchmark-TS44', 'A4Benchmark-TS45', 'A4Benchmark-TS46', 'A4Benchmark-TS47', 'A4Benchmark-TS48', 'A4Benchmark-TS49', 'A4Benchmark-TS4', 'A4Benchmark-TS50', 'A4Benchmark-TS52', 'A4Benchmark-TS55', 'A4Benchmark-TS56', 'A4Benchmark-TS57', 'A4Benchmark-TS58', 'A4Benchmark-TS59', 'A4Benchmark-TS5', 'A4Benchmark-TS60', 'A4Benchmark-TS61', 'A4Benchmark-TS62', 'A4Benchmark-TS63', 'A4Benchmark-TS64', 'A4Benchmark-TS65', 'A4Benchmark-TS67', 'A4Benchmark-TS68', 'A4Benchmark-TS69', 'A4Benchmark-TS6', 'A4Benchmark-TS70', 'A4Benchmark-TS71', 'A4Benchmark-TS72', 'A4Benchmark-TS73', 'A4Benchmark-TS74', 'A4Benchmark-TS75', 'A4Benchmark-TS76', 'A4Benchmark-TS77', 'A4Benchmark-TS78', 'A4Benchmark-TS79', 'A4Benchmark-TS7', 'A4Benchmark-TS80', 'A4Benchmark-TS81', 'A4Benchmark-TS82', 'A4Benchmark-TS84', 'A4Benchmark-TS85', 'A4Benchmark-TS86', 'A4Benchmark-TS87', 'A4Benchmark-TS88', 'A4Benchmark-TS89', 'A4Benchmark-TS8', 'A4Benchmark-TS90', 'A4Benchmark-TS91', 'A4Benchmark-TS92', 'A4Benchmark-TS93', 'A4Benchmark-TS94', 'A4Benchmark-TS95', 'A4Benchmark-TS97', 'A4Benchmark-TS98', 'A4Benchmark-TS99']

    for entity in entity_list:

        if args.dataset == 'anomaly_archive':
            meta_data = get_meta_data(entity)
            train_end = int(meta_data['train_end'])
            if train_end<10000:
                args.window_step = 1
            elif train_end>=10000 and train_end<100000:
                args.window_step = 10
            elif train_end>=100000:
                args.window_step = 100


        params = utils.AttrDict(
            # Training params
            batch_size=args.batch_size,
            lr=args.lr,
            epoch=args.epoch,
            max_grad_norm=1.0,
            seed=args.seed,

            # Model params
            n_features = n_features,
            mask = args.mask,
            reverse = args.reverse,

            emb_time_dim=128,
            emb_feature_dim=16,
            diffusion_embedding_dim=128,
            nheads=args.nheads,
            residual_layers=args.residual_layers,
            residual_channels=args.residual_channels,
            noise_schedule = np.linspace(args.beta_start, args.beta_end, args.diffusion_steps).tolist() if args.diffusion_steps>1 else [args.beta_end],
            diffusion_steps = args.diffusion_steps,
            diffusion_steps_i = args.diffusion_steps_i,
        )

        dataparams = utils.AttrDict(
            dataset=args.dataset,
            entities=entity,
            downsampling=args.downsampling,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_step=args.window_step,
        )


        base_dir = f'./result/{args.run_name}'
        data_dir = f'{args.dataset}/{entity}/d{dataparams.downsampling}_b{dataparams.batch_size}_w{dataparams.window_size}_s{dataparams.window_step}'
        param_dir = f'{params.diffusion_steps}_{args.beta_start}_{args.beta_end}/'
        param_dir += f'/{params.residual_layers}_{params.residual_channels}_{params.nheads}/{params.mask}'
        model_dir = f'{base_dir}/{data_dir}/{param_dir}/{args.seed}'


        train_dataloader, val_dataloader = datautils.load_dataloader(dataparams, group='train')
        test_dataloader = datautils.load_dataloader(dataparams, group='test')
        print('# of train',len(train_dataloader))
        print('# of valid',len(val_dataloader) if val_dataloader else None)
        print('# of test', len(test_dataloader))

        args.Train=True
        if os.path.isdir(f'{model_dir}/test'):
            args.Train=False
        if args.Train:
            print('Train')
            print(model_dir)
            model = AnomalyFilter(model_dir = model_dir, params = params, device = device)
            model.train(train_dataloader, val_dataloader)


        #noiseless inference
        params.reverse=True
        args.Test=True
        test_save_dir = f'{model_dir}/test'
        if os.path.isfile(f'{test_save_dir}/input.npy'):
            print(f'{test_save_dir}/input.npy', 'exists')
            args.Test=False
        if args.Test:
            print('Test')
            test_inputs, test_prediction = test(test_dataloader, model_dir, params, device)
            os.makedirs(test_save_dir, exist_ok=True)
            np.save(f'{test_save_dir}/input.npy',test_inputs)
            np.save(f'{test_save_dir}/pred.npy',test_prediction)

            print('Anomaly Score')
            B,D,W = test_inputs.shape

            input_sequence = np.zeros((D,B*W))
            pred_sequence = np.zeros((D,B*W))
            for b in range(B):
                input_sequence[:,W*b:W*(b+1)] = test_inputs[b]
                pred_sequence[:,W*b:W*(b+1)] = test_prediction[b]
            half_window = int(args.window_size/2)
            input_sequence[:,:half_window] = 0
            pred_sequence[:,:half_window] = 0

            anomaly_score = np.sum((input_sequence - pred_sequence)**2, axis=0)
            anomaly_score = np.convolve(anomaly_score, np.ones(half_window)/half_window, mode='same')
            np.save(f'{test_save_dir}/anomaly_score.npy',anomaly_score)


            if args.dataset == 'anomaly_archive':
                def min_max(data):
                    return (data - min(data))/(max(data)-min(data))
                print('Plot')
                import matplotlib.pyplot as plt
                window = 300
                window_e = 300
                meta_data = get_meta_data(entity)
                anomaly_start = meta_data['anomaly_start_in_test']
                anomaly_end = meta_data['anomaly_end_in_test']
                if anomaly_start==anomaly_end:
                    anomaly_end += 1
                anomaly_length = anomaly_end - anomaly_start

                plt.figure()
                if window>len(anomaly_score):
                    window = anomaly_start
                    window_e = len(anomaly_score)-anomaly_end
                anomaly_score = min_max(anomaly_score)
                plt.plot(anomaly_score[anomaly_start-window:anomaly_end+window_e], label='anomaly_score')
                plt.plot(input_sequence[0, anomaly_start-window:anomaly_end+window_e], label='input')
                plt.plot(pred_sequence[0, anomaly_start-window:anomaly_end+window_e], label='pred')
                plt.axvspan(window, window+anomaly_length, color='r', alpha=0.3)
                plt.legend()
                plt.savefig(f'{test_save_dir}/fig_{entity}.png')
                plt.close()

            print('Finish')


