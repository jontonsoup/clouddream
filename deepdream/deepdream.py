# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
import json
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import shutil
import caffe

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

with open("settings.json") as json_file:
    json_data = json.load(json_file)

img = PIL.Image.open('inputs/input.jpg')
if (img == None):
    quit()

model_path = '../caffe/models/bvlc_googlenet/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is storred in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimiation objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normaized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, jitter=32, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, jitter=jitter, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            #showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


maxwidth = json_data['maxwidth']
width = img.size[0]

if width > maxwidth:
    wpercent = (maxwidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((maxwidth,hsize), PIL.Image.ANTIALIAS)

img = np.float32(img)

frame = img

def process(net, frame):
    layers = [
            "inception_3b/5x5_reduce",
            ]

    for octave_n in xrange(1,10,1):
        for octave_scale in [0.5,1.5,2,2.2]:
            for iterations in xrange(75,100,10):
                for layer in layers:
                    output = deepdream(net, frame, iter_n=iterations, octave_n=octave_n, octave_scale=octave_scale, end=layer)
                    name = layer.replace("/", "") + "_itr_" + str(iterations) + "_octs_"
                    name2 = name + str(octave_n) + "_scl_" + str(octave_scale) + "_jt_"
                    name3 = name2 + "32"

                    PIL.Image.fromarray(np.uint8(output)).save("outputs/"+ name3 + ".jpg", dpi=(600,600))

    shutil.move("inputs/input.jpg", "done/input.jpg")

process(net, frame)
