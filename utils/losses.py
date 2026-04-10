import torch.nn as nn
import torch
import numpy as np
import mir_eval
import scipy
import auraloss
import torch, torch.nn as nn, numpy as np
from kymatio.torch import TimeFrequencyScattering
from Utils import JTFSLoss
import psyacloss.psyacloss_torch as psyacloss
import librosa
import scipy.spatial.distance

class LossApplicator:
    def __init__(self):
        super().__init__()
        self.loss_funcs = {}

    def add_losses(self, losses_):
        for loss in losses_:
            self.add_loss(*loss)

    def add_loss(self, name, args):
        lf_name = args.pop('func_name')
        assert name not in self.loss_funcs.keys()
        self.loss_funcs[name] = eval(lf_name)(**args)

    def get_loss(self, input, target, loss_name):
        if input.ndim == 1:
            input = np.expand_dims(input, 0)
            target = np.expand_dims(target, 0)
        return self.loss_funcs[loss_name].forward(input, target)

    def get_losses(self, input, target):
        losses = {}
        if input.ndim == 1:
            input = np.expand_dims(input, 0)
            target = np.expand_dims(target, 0)
        for each in self.loss_funcs:
            losses[each] = self.loss_funcs[each].forward(input, target)

        return losses


class ESR:
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        loss = np.add(target, -output)
        loss = np.square(loss)
        loss = np.mean(loss, axis=1)
        energy = np.mean(np.square(target), axis=1) + self.epsilon
        loss = np.divide(loss, energy)

        return loss

class JTFS():
    def __init__(self, fs, dur):
        super().__init__()
        self.jtfs_kwargs = {
            "shape": (fs*dur,),
            "Q": (8, 2),
            "J": 12,
            "J_fr": 5,
            "F": 0,
            "Q_fr": 2,
            "format": "time",
        }
        #self.jtfs = TimeFrequencyScattering(**self.jtfs_kwargs)
        self.specloss = JTFSLoss.TimeFrequencyScatteringLoss(**self.jtfs_kwargs)

    def forward(self, output, target):
        return self.specloss(torch.tensor(output).float(), torch.tensor(target).float()).numpy()


class MSE:

    @staticmethod
    def forward(output, target):
        loss = np.add(target, -output)
        loss = np.square(loss)
        if output.ndim == 3:
            loss = np.mean(np.abs(loss), axis=(1, 2))
        else:
            loss = np.mean(np.abs(loss), axis=1)
        return loss


class MAE:

    @staticmethod
    def forward(output, target):
        loss = np.add(target, -output)
        if output.ndim == 3:
            loss = np.mean(np.abs(loss), axis=(1, 2))
        else:
            loss = np.mean(np.abs(loss), axis=1)
        return loss


class BSSEval:

    @staticmethod
    def forward(output, target):
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(target, output, compute_permutation=False)
        return sdr


class STFT_Loss:
    def __init__(self, fft_size=1024,
                 hop_size=256,
                 window='hann',
                 lam_sc=1.0,
                 lam_log=0,
                 lam_lin=0,
                 dist='MAE',
                 scale_inv=False,
                 energy_norm=False,
                 fs=44100
                 ):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window = scipy.signal.windows.get_window(window, fft_size, fftbins=False)
        self.lam_sc = lam_sc
        self.lam_log = lam_log
        self.lam_lin = lam_lin
        self.dist = dist
        self.scale_inv = scale_inv
        self.energy_norm = energy_norm
        self.fs = fs

        self.stft = scipy.signal.ShortTimeFFT(win=self.window, hop=hop_size, fs=self.fs, fft_mode='onesided')

    def forward(self, output, target):
        if energy_norm:
            output, target = energy_norm(output, target)

        out_mag, out_phs = self.get_stft(output)
        tgt_mag, tgt_phs = self.get_stft(target)

        # compute loss terms
        spec_conv_l = SpecConvLoss(out_mag, tgt_mag) if self.lam_sc else 0.0
        log_mag_loss = stftloss(out_mag, tgt_mag, log=True, dist=self.dist) if self.lam_log else 0.0
        lin_mag_loss = stftloss(out_mag, tgt_mag, log=False, dist=self.dist) if self.lam_lin else 0.0

        return spec_conv_l + log_mag_loss + lin_mag_loss

    def get_stft(self, x):
        x_stft = self.stft.stft(x)
        x_mag = np.abs(x_stft)
        x_phs = np.angle(x_stft)
        return x_mag, x_phs

class AuraSTFT():
    def __init__(self, args):
        self.loss = auraloss.freq.STFTLoss(**args)

    def forward(self, output, target):
        if output.ndim == 2:
            output = np.expand_dims(output, 0)
            target = np.expand_dims(target, 0)
        return self.loss(torch.tensor(output).float(), torch.tensor(target).float()).numpy()


class AuraMRSTFT():
    def __init__(self, args):
        self.loss = auraloss.freq.MultiResolutionSTFTLoss(**args)

    def forward(self, output, target):
        if output.ndim == 2:
            output = np.expand_dims(output, 0)
            target = np.expand_dims(target, 0)
        return self.loss(torch.tensor(output).float(), torch.tensor(target).float()).numpy()


class AuraMelSTFT():
    def __init__(self, args):
        self.loss = auraloss.freq.MelSTFTLoss(**args)

    def forward(self, output, target):
        if output.ndim == 2:
            output = np.expand_dims(output, 0)
            target = np.expand_dims(target, 0)
        return self.loss(torch.tensor(output).float(), torch.tensor(target).float()).numpy()


def energy_norm(output, target):
    alpha = np.sqrt(np.mean(np.square(target), axis=1))
    return output/alpha, target/alpha


def SpecConvLoss(output, target):
    return np.linalg.norm(output - target, ord='fro', axis=(1, 2)) / np.linalg.norm(target, ord='fro', axis=(1, 2))


def stftloss(output, target, log=True, dist='MAE'):
    if log:
        output = np.log10(output)
        target = np.log10(target)
    if dist == 'MAE':
        return MAE.forward(output, target)
    elif dist == 'MSE':
        return MSE.forward(output, target)
    else:
        return None
    
class psy_acu():
    # psychoacoustic loss by G. Schuller
    def __init__(self, fs):
        self.fs = fs

    def forward(self, output, target):
        
        # but for the torch version to work, the signals should be in mono 
        # as of now the loss oesn't suppoert batch processing (it doesn't differentiate between channel and batch dimension)
        assert output.ndim < 3, "Only mono signals are supported"

        ploss = 0
        for i in range(output.shape[0]):
            ploss += psyacloss.percloss(
                torch.tensor(target[i,:]).float(), 
                torch.tensor(output[i,:]).float(), self.fs)
        return (ploss/output.shape[0]).numpy() # mean over batch


class mfcc():
    def __init__(self, args):
        self.dist = args['dist']

    def forward(self, output, target):
        mfcc_out = librosa.feature.mfcc(y=output.squeeze(), sr=44100)
        mfcc_tgt = librosa.feature.mfcc(y=target.squeeze(), sr=44100)

        if self.dist == 'l1':
            return np.mean(np.abs(mfcc_out-mfcc_tgt))
        elif self.dist == 'l2':
            return np.mean(np.square(mfcc_out - mfcc_tgt))


class DeepFeatLoss():
    def __init__(self, args):
        self.feat = args['feat']
        self.dist = args['dist']
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output, target):
        if self.dist == 'l1':
            return np.mean(np.abs(output-target))
        elif self.dist == 'l2':
            return np.mean(np.square(output - target))
        elif self.dist == 'cosine':
            return 1 - np.mean(self.cos(torch.tensor(output), torch.tensor(target)).numpy())

    
if __name__ == '__main__':

    fs = 44100
    dur = 2

    out = np.random.randn(1, fs*dur)*100
    tgt = np.random.randn(1, fs*dur)*100
    esr = ESR()

    lone = esr.forward(out, tgt)

    out, tgt = energy_norm(out, tgt)
    ltwo = esr.forward(out, tgt)

    losses_to_apply = [['JTFS', {'func_name': 'JTFS', 'fs': fs, 'dur': dur}],
                       ['esr_basic', {'func_name': 'ESR'}],
                       ['mse', {'func_name': 'MSE'}],
                       ['mae', {'func_name': 'MAE'}],
                       ['bss_eval', {'func_name': 'BSSEval'}],
                       ['specconv', {'func_name': 'STFT_Loss', 'lam_sc': 1.0}],
                       ['logstft', {'func_name': 'STFT_Loss', 'lam_log': 1.0, 'lam_sc': 0}],
                       ['linstft', {'func_name': 'STFT_Loss', 'lam_lin': 1.0, 'lam_sc': 0}],
                       ['psyacu', {'func_name': 'psy_acu', 'fs': fs}],
                       ]

    l = LossApplicator()
    l.add_losses(losses_to_apply)

    inp = np.random.randn(1, fs*dur)
    tgt = np.random.randn(1, fs*dur)

    losses = l.get_losses(inp, tgt)
    
    ''' not sure how this was meant to be used
    inp_b = np.random.randn(5, fs*dur)
    inp_b[0, :] = inp
    tgt_b = np.random.randn(5, fs*dur)
    tgt_b[0, :] = tgt

    losses_2 = l.get_losses(inp_b, tgt_b)

    for each in l.loss_funcs.keys():
        assert losses[each] == losses_2[each][0:1]
    '''
    print(3)