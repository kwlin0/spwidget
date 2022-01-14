import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import fsps
import ipywidgets as widgets
from pyphot import LickLibrary, unit
from bqplot import DateScale, LinearScale, Axis, Figure, Lines
from ipywidgets import FloatSlider, VBox, HBox

t = np.linspace(0, 13, 150)
stellarmass_array_delayedtau = np.zeros(len(t))

def sfh_dec(t, tau):
    
    def sfh(t, tau):
        return np.exp(-t/tau)
   
    totalstellarmass = 1e12
    norm = totalstellarmass / integrate.quad(sfh, 0, 13, args=(tau))[0]
    
    return norm * np.exp(-t/tau)

def sfh_inc(t, tau):
    
    def sfh(t, tau):
        return np.exp(t/tau)
   
    totalstellarmass = 1e12
    norm = totalstellarmass / integrate.quad(sfh, 0, 13, args=(tau))[0]
    
    return norm * np.exp(t/tau)

def sfh_delay_dec(t, tau):
    
    def sfh_delay(t, tau):
        return t * np.exp(-t/tau)
   
    totalstellarmass = 1e12
    norm = totalstellarmass / integrate.quad(sfh_delay, 0, 13, args=(tau))[0]
    
    return norm * t * np.exp(-t/tau)

def sfr_linexp(t, tau, t_truncate, lin_slope):
    if t <= t_truncate:
        return t * np.exp(-t/tau)
    else:
        return lin_slope

def stellarmass_tau(t, tau):
    #stellarmass_array = np.zeros(len(t))
    stellarmass_array = [integrate.quad(sfh_dec, 0, time, args=(tau))[0] for time in t]
    #for idx, time in enumerate(t):
    #    stellarmass_array[idx] = integrate.quad(sfh_dec, 0, time, args=(tau))[0]
    return stellarmass_array

def stellarmass_delayedtau(t, tau):
    #stellarmass_array = np.zeros(len(t))
    stellarmass_array = [integrate.quad(sfh_delay_dec, 0, time, args=(tau))[0] for time in t]
    #for idx, time in enumerate(t):
    #    stellarmass_array[idx] = integrate.quad(sfh_delay_dec, 0, time, args=(tau))[0]
    return stellarmass_array

sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=0, sfh=1, tau=1.0, logzsol=0.0, dust_type=2, dust2=0.0)

def Mgb_tau(time, tau, logzsol):
    sp.params['sfh'] = 1
    sp.params['tau'] = tau
    sp.params['logzsol'] = logzsol
    lickindex = np.zeros(len(time[1:]))
    for i, tage in enumerate(time[1:]):
        wave, spec = sp.get_spectrum(tage=tage, peraa=True)
        lickindex[i] = LickLibrary()['Mg_b'].get(wave*unit['AA'], spec)
    return lickindex

def Mv_tau(time, tau, logzsol):
    sp.params['sfh'] = 1
    sp.params['tau'] = tau
    sp.params['logzsol'] = logzsol
    Mv_array = np.zeros(len(time[1:]))
    for i, tage in enumerate(time[1:]):
        Mv_array[i] = sp.get_mags(tage=tage, bands=['v'])
    return Mv_array

def Mv_delayedtau(time, tau, logzsol):
    sp.params['sfh'] = 4
    sp.params['tau'] = tau
    sp.params['logzsol'] = logzsol
    Mv_array = np.zeros(len(time[1:]))
    for i, tage in enumerate(time[1:]):
        Mv_array[i] = sp.get_mags(tage=tage, bands=['v'])
    return Mv_array

def BVcolor_tau(time, tau, logzsol):
    sp.params['sfh'] = 1
    sp.params['tau'] = tau
    sp.params['logzsol'] = logzsol
    BVcolor_array = np.zeros(len(time[1:]))
    for i, tage in enumerate(time[1:]):
        bmag, vmag = sp.get_mags(tage=tage, bands=['b', 'v'])
        BVcolor_array[i] = bmag - vmag
    return BVcolor_array

def BVcolor_delayedtau(time, tau, logzsol):
    sp.params['sfh'] = 4
    sp.params['tau'] = tau
    sp.params['logzsol'] = logzsol
    BVcolor_array = np.zeros(len(time[1:]))
    for i, tage in enumerate(time[1:]):
        bmag, vmag = sp.get_mags(tage=tage, bands=['b', 'v'])
        BVcolor_array[i] = bmag - vmag
    return BVcolor_array


def show_plots(tau = 1., logzsol = 0.0):
    
    """
    Optional arguments:
    
    tau : e-folding time scale (default = 1.)
    logzsol : metallicity of stellar population in units of logarithmic solar Z (default = 0.0)
    
    """
    default_tau = tau
    default_logzsol = logzsol
    
    # Make sliders

    tau_slider = FloatSlider(value=default_tau, min = 0.1, max = 15, step = .1, description = 'Tau')
    Z_slider = FloatSlider(value=default_logzsol, min = -5, max = 5, step = .1, description = 'log Metallicity')
    
    # Define scales
    
    x_sfh = LinearScale()
    y_sfh = LinearScale()

    x_mass = LinearScale()
    y_mass = LinearScale()
    
    x_bv = LinearScale()
    y_bv = LinearScale()
    
    # Define Axis objects
    
    ax_x_sfh = Axis(label='Time [Gyr]', scale=x_sfh, grid_lines='solid')
    ax_y_sfh = Axis(label='SFR [solar mass/yr]', scale=y_sfh, orientation='vertical', grid_lines='solid', label_offset='-50')
    
    ax_x_mass = Axis(label='Time [Gyr]', scale=x_mass, grid_lines='solid')
    ax_y_mass = Axis(label='Stellar Mass [solar mass]', scale=y_mass, orientation='vertical', grid_lines='solid', label_offset='-50')
    
    ax_x_bv = Axis(label='Time [Gyr]', scale=x_bv, grid_lines='solid')
    ax_y_bv = Axis(label='B-V', scale=y_bv, orientation='vertical', grid_lines='solid', label_offset='-50')
    
    # Make Lines
    
    sfh_tau = Lines(y=sfh_dec(t, tau=default_tau),x=t , scales={'x': x_sfh, 'y': y_sfh}, display_legend=True, labels=['Tau ='+str(default_tau)], colors = ['#FF0000'])
    sfh_dtau = Lines(y=sfh_delay_dec(t, tau=default_tau), x=t , scales={'x': x_sfh, 'y': y_sfh}, display_legend=True, labels=['Delayed Tau'])
    #sfh_inctau = Lines(y=sfh_inc(t, tau=default_tau), x=t , scales={'x': x_sfh, 'y': y_sfh}, display_legend=True, labels=['Rising Tau'], colors = ['#00A300'])

    smass_tau = Lines(y=stellarmass_tau(t, tau=default_tau), x=t, scales={'x': x_mass, 'y': y_mass}, colors = ['#FF0000'])
    smass_dtau = Lines(y=stellarmass_delayedtau(t, tau=default_tau), x=t, scales={'x': x_mass, 'y': y_mass})
    
    bv_tau = Lines(y=BVcolor_tau(t, default_tau, default_logzsol), x=t[1:],  scales={'x': x_bv, 'y': y_bv}, colors = ['#FF0000'])
    bv_dtau = Lines(y=BVcolor_delayedtau(t, default_tau, default_logzsol), x=t[1:],  scales={'x': x_bv, 'y': y_bv})
    
    # Make Figures
    
    fig1 = Figure(axes=[ax_x_sfh, ax_y_sfh], marks=[sfh_tau, sfh_dtau], title='Star Formation History', legend_location='top-right')
    fig1.layout.width = '50%'
    
    fig2 = Figure(axes=[ax_x_mass, ax_y_mass], marks=[smass_tau, smass_dtau], title='Stellar Mass')
    fig2.layout.width = '50%'
    
    fig3 = Figure(axes=[ax_x_bv, ax_y_bv], marks=[bv_tau, bv_dtau], title='B-V Color (Z= '+str(default_logzsol)+')')
    fig3.layout.width = '50%'
    
    # Update Lines from slider callback
    
    def update_plot(change):
        sfh_tau.y = sfh_dec(t, tau=tau_slider.value)
        sfh_tau.labels = ['Tau ='+str(tau_slider.value)]
        sfh_dtau.y = sfh_delay_dec(t, tau=tau_slider.value)
        #sfh_inctau.y = sfh_inc(t, tau=tau_slider.value)
        smass_tau.y = stellarmass_tau(t, tau=tau_slider.value)
        smass_dtau.y = stellarmass_delayedtau(t, tau=tau_slider.value)
        bv_tau.y = BVcolor_tau(t, tau_slider.value, Z_slider.value)
        bv_dtau.y = BVcolor_delayedtau(t, tau_slider.value, Z_slider.value)
        fig3.title = 'B-V Color (Z= '+str(Z_slider.value)+')'
    
    tau_slider.observe(update_plot,'value')
    
    Z_slider.observe(update_plot,'value')
    
    return VBox([tau_slider,Z_slider, HBox([fig1,fig2], align_content = 'stretch'), fig3])