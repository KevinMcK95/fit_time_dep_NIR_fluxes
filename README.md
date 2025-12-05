# fit_time_dep_NIR_fluxes

When fitting fluxes for near-infrared (NIR) detectors, it is common to assume the fluxes are not changing over the course of the exposure. This a good assumption for space-based data (e.g. JWST, Roman), but is not true for ground-based observations because the atmosphere is likely changing on the same timescale as detector readouts. 

We build upon a previous approach by Tim Brandt (see [fitramp.py]([https://example.com](https://github.com/t-brandt/fitramp/blob/main/fitramp.py))) to include a time-variable component to fluxes that shares information between neighbouring pixels. We discuss the statistical underpinings of this work in Li et al. (in prep, arxiv link soon). We test this process using synthetic data as well as real data from the APOGEE spectrograph. 
