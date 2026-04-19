##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: process_outputs.py
#
# Description: Read in sim_outs.h5 and process (integrate, low-cut, low-pass, etc.)
# This gets called from command line with cmdline arguments (run after simulation)
#
##############################################################################

import numpy as np
from numpy import array as npa
from pathlib import Path
import time
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter,bilinear_zpk,zpk2sos,sosfilt,lfilter
from numpy import pi,sqrt
from numpy.fft import rfft
from numpy import log10,log2
from resampy import resample
from air_abs.visco_filter import apply_visco_filter
from air_abs.modal_filter import apply_modal_filter
from air_abs.ola_filter import apply_ola_filter
from common.myfuncs import wavwrite,iceil,iround

FREQ_MIN_HZ = 20.0
FREQ_MAX_HZ = 20e3

#class to process sim_outs.h5 file
class ProcessOutputs: 
    def __init__(self,data_dir):
        self.print('loading...')

        #get some integers from comms_out
        self.data_dir = data_dir
        h5f = h5py.File(data_dir / Path('comms_out.h5'),'r')
        out_alpha = h5f['out_alpha'][...]
        Nr = h5f['Nr'][()]
        Nt = h5f['Nt'][()]
        diff = h5f['diff'][()]
        h5f.close()

        #get some sim constants (floats) from sim_consts
        h5f = h5py.File(data_dir / Path('sim_consts.h5'),'r')
        Ts = h5f['Ts'][()]
        c = h5f['c'][()]
        Tc = h5f['Tc'][()]
        rh = h5f['rh'][()]
        h5f.close()

        #read the raw outputs from sim_outs
        h5f = h5py.File(data_dir / Path('sim_outs.h5'),'r')
        u_out  = h5f['u_out'][...]
        h5f.close()
        self.print('loading done...')

        assert out_alpha.size == Nr
        assert u_out.size == Nr*Nt
        assert out_alpha.ndim == 2

        self.r_out = None #for recombined raw outputs (corresponding to Rxyz)
        self.r_out_f = None #r_out filtered (and diffed)

        self.Ts = Ts
        self.Fs = 1/Ts
        self.Nt = Nt
        self.Ts_f = Ts
        self.Fs_f = 1/Ts
        self.Nt_f = Nt
        self.Nr = Nr
        #self.Nmic = out_alpha.shape[0]
        self.u_out =  u_out
        self.diff = diff
        self.out_alpha = out_alpha
        self.data_dir = data_dir

        self.Tc = Tc
        self.rh = rh

    def print(self,fstring):
        print(f'--PROCESS_OUTPUTS: {fstring}')

    #initial process: consolidate receivers with linterp weights, and integrate/low-cut
    def initial_process(self,fcut=10.0,N_order=4):
        self.print('initial process...')
        u_out = self.u_out
        out_alpha = self.out_alpha
        data_dir = self.data_dir
        apply_int = self.diff
        Ts = self.Ts

        #just recombine outputs (from trilinear interpolation)
        r_out = np.sum((u_out*out_alpha.flat[:][:,None]).reshape((*out_alpha.shape,-1)),axis=1)

        h5f = h5py.File(data_dir / Path('sim_outs.h5'),'r+')
        try:
            del h5f['r_out']
            self.print('overwrite r_out dataset (native sample rate)')
        except:
            pass
        h5f.create_dataset('r_out', data=r_out)
        h5f.close()

        if fcut>0:
            if apply_int:
                #design combined butter filter with integrator
                z,p,k=butter(N_order,fcut*2*pi,btype='high',analog=True,output='zpk') 
                assert(np.all(z==0.0))
                z = z[1:] #remove one zero
                #p = np.r_[p,0] #adds integrator (bilinear routine handles p-z cancellation)
                zd,pd,kd=bilinear_zpk(z,p,k,1/Ts)
                sos = zpk2sos(zd,pd,kd)
                self.print('applying lowcut-integrator')
            else:
                #design digital high-pass
                sos=butter(N_order,2*Ts*fcut,btype='high',output='sos') 
                self.print('applying lowcut')
            r_out_f = sosfilt(sos,r_out)
        elif apply_int: #shouldn't really use this without lowcut, but here in case
            b = Ts/2*npa([1,1])
            a = npa([1,1])
            r_out_f = lfilter(b,a,r_out)
            self.print('applying integrator')
        else:
            r_out_f = np.copy(r_out)
        self.print('initial process done')

        self.r_out = r_out
        self.r_out_f = r_out_f

    #lowpass filter for fmax (to remove freqs with too much numerical dispersion)
    def apply_lowpass(self,fcut,N_order=8,symmetric=True):
        Ts_f = self.Ts_f
        r_out_f = self.r_out_f
        nyquist_hz = 0.5/self.Ts_f
        fcut = min(fcut,FREQ_MAX_HZ,0.999*nyquist_hz)

        if symmetric: #will be run twice
            assert N_order%2==0
            N_order= int(N_order//2)
            self.print(f'{N_order=} for symmetric IIR filtering')

        #design digital high-pass
        sos=butter(N_order,2*Ts_f*fcut,btype='low',output='sos') 
        self.print('applying lowpass to filtered output')
        r_out_f = sosfilt(sos,r_out_f)
        if symmetric: #runs again, time reversed
            self.print('applying second time in reverse to remove phase shift')
            r_out_f = sosfilt(sos,r_out_f[:,::-1])[:,::-1]
            
        self.r_out_f = r_out_f

    #resample with resampy, 48kHz default 
    def resample(self,Fs_f=48e3):
        Fs = self.Fs #raw Fs
        if Fs==Fs_f:
            return
        r_out_f = self.r_out_f

        self.print(f'resampling')
        r_out_f = resample(r_out_f, Fs, Fs_f, filter='kaiser_best')

        self.Fs_f = Fs_f
        self.Ts_f = 1/Fs_f
        self.Nt_f = r_out_f.shape[-1]
        self.r_out_f = r_out_f

    #to apply Stokes' filter (see DAFx2021 paper)
    def apply_stokes_filter(self,NdB=120):
        Fs_f = self.Fs_f
        Tc = self.Tc
        rh = self.rh
        r_out_f = self.r_out_f
        self.print(f'applying Stokes'' air absorption filter')
        r_out_f = apply_visco_filter(r_out_f, Fs_f, Tc=Tc,rh=rh, NdB=NdB)
        self.Nt_f = r_out_f.shape[-1] #gets lengthened by filter

        self.r_out_f = r_out_f
        Nt_f = self.Nt_f

    #to apply Stokes' filter (see I3DA 2021 paper)
    def apply_modal_filter(self):
        Fs_f = self.Fs_f
        Tc = self.Tc
        rh = self.rh
        r_out_f = self.r_out_f
        self.print(f'applying modal air absorption filter')
        r_out_f = apply_modal_filter(r_out_f, Fs_f, Tc=Tc,rh=rh)
        self.Nt_f = r_out_f.shape[-1] #gets lengthened by filter

        self.r_out_f = r_out_f
        Nt_f = self.Nt_f

    #to apply air absorption through STFT (overlap-add) framework
    def apply_ola_filter(self): #default settings for 48kHz
        Fs_f = self.Fs_f
        Tc = self.Tc
        rh = self.rh
        r_out_f = self.r_out_f
        self.print(f'applying OLA air absorption filter')
        r_out_f = apply_ola_filter(r_out_f, Fs_f, Tc=Tc,rh=rh)
        self.Nt_f = r_out_f.shape[-1] #maybe lengthened by filter

        self.r_out_f = r_out_f
        Nt_f = self.Nt_f

    #plot the raw outputs (just to debug)
    def plot_raw_outputs(self):
        Nt = self.Nt
        Ts = self.Ts
        Fs = self.Fs
        tv = np.arange(Nt)*Ts
        u_out = self.u_out
        r_out = self.r_out

        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)
        #for out in u_out:
            #ax.plot(tv,out,linestyle='-')
        #ax.set_title('raw grid outputs')
        #ax.margins(0, 0.1)
        #ax.set_xlabel('time (s)')
        #ax.grid(which='both', axis='both')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(r_out.shape[0]):
            ax.plot(tv,r_out[i],linestyle='-',label=f'{i}')
        ax.set_title('r_out')
        ax.margins(0, 0.1)
        ax.set_xlabel('time (s)')
        ax.grid(which='both', axis='both')
        ax.legend()

    #plot the final processed outputs 
    def plot_filtered_outputs(self):
        #possibly resampled
        r_out_f = self.r_out_f
        Nt_f = self.Nt_f
        Ts_f = self.Ts_f
        Fs_f = self.Fs_f
        tv = np.arange(Nt_f)*Ts_f
        Nfft = 2**iceil(log2(Nt_f))
        fv = np.arange(np.int_(Nfft//2)+1)/Nfft*Fs_f

        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        for i in range(r_out_f.shape[0]):
            ax.plot(tv,r_out_f[i],linestyle='-',label=f'R{i+1}')
        ax.set_title('r_out filtered')
        ax.margins(0, 0.1)
        ax.set_xlabel('time (s)')
        ax.grid(which='both', axis='both')
        ax.legend()

        ax = fig.add_subplot(2, 1, 2)
        r_out_f_fft_dB = 20*log10(np.abs(rfft(r_out_f,Nfft,axis=-1))+np.spacing(1))
        freq_mask = (fv>=FREQ_MIN_HZ) & (fv<=FREQ_MAX_HZ)
        if not np.any(freq_mask):
            raise ValueError('No FFT bins in the 20 Hz to 20 kHz range')
        fv_plot = fv[freq_mask]
        r_out_f_fft_dB = r_out_f_fft_dB[:,freq_mask]
        dB_max = np.max(r_out_f_fft_dB)
        for i in range(r_out_f.shape[0]):
            ax.plot(fv_plot,r_out_f_fft_dB[i],linestyle='-',label=f'R{i+1}')
        ax.set_title('r_out filtered')
        ax.margins(0, 0.1)
        ax.set_xlabel('freq (Hz)')
        ax.set_ylabel('dB')
        ax.set_xscale('log')
        ax.set_ylim((dB_max-80,dB_max+10))
        ax.set_xlim((FREQ_MIN_HZ,min(FREQ_MAX_HZ,Fs_f/2)))
        ax.grid(which='both', axis='both')
        ax.legend()

    def benchmark_frequency_bounded_calculation(self,repeats=25):
        if self.r_out_f is None:
            raise RuntimeError('Run initial_process() before benchmarking')
        if repeats < 1:
            raise ValueError('repeats must be >= 1')

        r_out_f = self.r_out_f
        Nt_f = self.Nt_f
        Fs_f = self.Fs_f
        Nfft = 2**iceil(log2(Nt_f))
        fv = np.arange(np.int_(Nfft//2)+1)/Nfft*Fs_f
        freq_mask = (fv>=FREQ_MIN_HZ) & (fv<=FREQ_MAX_HZ)
        if not np.any(freq_mask):
            raise ValueError('No FFT bins in the 20 Hz to 20 kHz range')

        # Legacy: full-band FFT dB calculation used before the 20-20k bound.
        t0 = time.perf_counter()
        for _ in range(repeats):
            legacy_fft_db = 20*log10(np.abs(rfft(r_out_f,Nfft,axis=-1))+np.spacing(1))
            np.max(legacy_fft_db)
        t_legacy = time.perf_counter()-t0

        # New: same FFT followed by restricting calculations to 20Hz-20kHz bins.
        t1 = time.perf_counter()
        for _ in range(repeats):
            bounded_fft_db = 20*log10(np.abs(rfft(r_out_f,Nfft,axis=-1))+np.spacing(1))
            bounded_fft_db = bounded_fft_db[:,freq_mask]
            np.max(bounded_fft_db)
        t_bounded = time.perf_counter()-t1

        legacy_avg = t_legacy/repeats
        bounded_avg = t_bounded/repeats
        speedup = legacy_avg/max(bounded_avg,np.finfo(float).tiny)
        ratio = freq_mask.sum()/freq_mask.size

        self.print('benchmark: original vs bounded frequency calculation')
        self.print(f'repeats={repeats}, bins_kept={freq_mask.sum()}/{freq_mask.size} ({100*ratio:.2f}%)')
        self.print(f'legacy total={t_legacy:.6f}s, avg={legacy_avg:.6e}s')
        self.print(f'bounded total={t_bounded:.6f}s, avg={bounded_avg:.6e}s')
        self.print(f'speedup(legacy/bounded)={speedup:.3f}x')

        return {
            'repeats': repeats,
            'legacy_total_s': t_legacy,
            'legacy_avg_s': legacy_avg,
            'bounded_total_s': t_bounded,
            'bounded_avg_s': bounded_avg,
            'speedup_legacy_over_bounded': speedup,
            'bins_kept': int(freq_mask.sum()),
            'bins_total': int(freq_mask.size),
        }

    def show_plots(self):
        plt.show()

    #save in WAV files, with native scaling and normalised across group of receivers
    def save_wav(self):
        #saves processed outputs
        Fs_f = self.Fs_f
        data_dir = self.data_dir
        r_out_f = self.r_out_f
        r_out_f = np.atleast_2d(r_out_f) 
        n_fac = np.max(np.abs(r_out_f.flat[:])) 
        self.print(f'headroom = {-20*np.log10(n_fac):.1}dB')
        for i in range(r_out_f.shape[0]):
            fname = Path(data_dir / Path(f'R{i+1:03d}_out_normalised.wav')) #normalised across receivers
            wavwrite(fname,int(Fs_f),r_out_f[i]/n_fac) 
            if n_fac<1.0:
                fname = Path(data_dir / Path(f'R{i+1:03d}_out_native.wav')) #not scaled, direct sound amplitude ~1/4πR
                wavwrite(fname,int(Fs_f),r_out_f[i]) 

    #saw processed outputs in .h5 (with native scaling) 
    def save_h5(self):
        #saves processed outputs
        self.print('saving H5 data..')
        h5f = h5py.File(self.data_dir / Path('sim_outs_processed.h5'),'w')
        h5f.create_dataset('r_out_f', data=self.r_out_f)
        h5f.create_dataset('Fs_f', data=self.Fs_f)
        h5f.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,help='run directory')
    parser.add_argument('--plot', action='store_true',help='plot filtered outputs')
    parser.add_argument('--plot_raw', action='store_true',help='plot raw outputs')
    parser.add_argument('--resample_Fs', type=float,help='output Fs for processed outputs')
    parser.add_argument('--fcut_lowcut', type=float,help='')
    parser.add_argument('--fcut_lowpass', type=float,help='')
    parser.add_argument('--N_order_lowcut', type=int,help='butter order lowcut (eg. 10Hz)')
    parser.add_argument('--N_order_lowpass', type=int,help='butter order lowpass')
    parser.add_argument('--symmetric_lowpass',action='store_true',help='make symmetric FIR out of IIR (N_order even)')
    parser.add_argument('--air_abs_filter', type=str,help='stokes, modal, OLA, or none  ')
    parser.add_argument('--save_wav', action='store_true',help='save WAV files of processed outputs')
    parser.add_argument('--benchmark_freq_bounds', action='store_true',help='benchmark legacy full-band vs 20Hz-20kHz bounded calc')
    parser.add_argument('--benchmark_repeats', type=int,help='number of repeated runs for benchmark')
    parser.set_defaults(plot=False)
    parser.set_defaults(plot_raw=False)
    parser.set_defaults(data_dir=None)
    parser.set_defaults(resample_Fs=48e3)
    parser.set_defaults(air_abs_filter='none')
    parser.set_defaults(save_wav=False)
    parser.set_defaults(N_order_lowpass=8)
    parser.set_defaults(N_order_lowcut=8)
    parser.set_defaults(fcut_lowcut=FREQ_MIN_HZ)
    parser.set_defaults(fcut_lowpass=FREQ_MAX_HZ)
    parser.set_defaults(symmetric_lowpass=False)
    parser.set_defaults(benchmark_freq_bounds=False)
    parser.set_defaults(benchmark_repeats=25)

    args = parser.parse_args()

    po = ProcessOutputs(args.data_dir)

    fcut_lowcut = max(args.fcut_lowcut,FREQ_MIN_HZ)
    po.initial_process(fcut=fcut_lowcut,N_order=args.N_order_lowcut)

    if args.resample_Fs:
        po.resample(args.resample_Fs)

    fcut_lowpass = min(args.fcut_lowpass,FREQ_MAX_HZ)
    if fcut_lowpass>0:
        po.apply_lowpass(fcut=fcut_lowpass,N_order=args.N_order_lowpass,symmetric=args.symmetric_lowpass)

    #these are only needed if you're simulating with fmax >1kHz, but generally fine to use
    if args.air_abs_filter.lower() == 'modal': #best, but slowest
        po.apply_modal_filter()
    elif args.air_abs_filter.lower() == 'stokes': #generally fine for indoor air
        po.apply_stokes_filter()
    elif args.air_abs_filter.lower() == 'ola': #fastest, but not as recommended
        po.apply_ola_filter()

    if args.benchmark_freq_bounds:
        po.benchmark_frequency_bounded_calculation(repeats=args.benchmark_repeats)

    po.save_h5()

    if args.save_wav:
        po.save_wav()
        
    if args.plot_raw:
        po.plot_raw_outputs()

    if args.plot or args.plot_raw:
        po.plot_filtered_outputs()
        po.show_plots()

if __name__ == '__main__':
    main()
