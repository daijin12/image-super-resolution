import runway
import numpy as np
from ISR.models import RDN, RRDN


@runway.setup(options={'checkpoint': runway.file(extension='.hdf5')})
def setup(opts):
    print("Checkpoint: ", opts["checkpoint"])
    if "C4" in opts['checkpoint']:
        rdn  = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':4}, patch_size=40)
    elif "C6" in opts["checkpoint"]:
        rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
    else:
        rdn = RDN(arch_params={'C':3, 'D':10, 'G':64, 'G0':64, 'x':2})
    rdn.model.load_weights(opts['checkpoint'])
    return rdn
    

@runway.command('upscale', inputs={'image': runway.image}, outputs={'upscaled': runway.image})
def upscale(rdn, inputs):
    width, height = inputs['image'].size
    if width >= 1000 or height >= 1000:
        return rdn.predict(np.array(inputs['image']), batch_size=10, by_patch_of_size=40, padding_size=5)
    else:
        return rdn.predict(np.array(inputs['image']), batch_size=1)


if __name__ == '__main__':
    runway.run(port=4231)
