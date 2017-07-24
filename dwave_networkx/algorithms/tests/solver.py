sampler_found = True

try:
    from dwave_sapi_dimod import SAPISampler as Sampler
except ImportError:
    _sampler_found = False
    raise
