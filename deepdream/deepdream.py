import os
os.environ['GLOG_minloglevel'] = '2'
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
import json
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import shutil
import caffe
from multiprocessing import Process

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    output = np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
    return output
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step_normalized(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
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

def objective_L2(dst, guide_features):
    dst.diff[:] = dst.data

def make_step(net, step_size=1.5, end='inception_4c/output',
              jitter=32, clip=True, guide_features=None, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    objective(dst, guide_features)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(filename, net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, model="", jitter=32, guide_features=None,guide=None,guide_file="",**step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    print("OCTS:" + str(octave_n))

    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail

        for i in xrange(iter_n):
            make_step(net, end=end, jitter=jitter, clip=clip, guide_features=guide_features, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            #showarray(vis)
            if ((i > 10) and (i % 10 == 0)):
                save_file(vis, end, i, octave, octave_scale, filename, model,guide_file)
            print filename, octave, i, end, vis.shape, guide_file
            clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

def objective_guide(dst, guide_features):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best


def process2(net, frame, filename, model, guide, guide_file):
    layers = net.blobs.keys()
    #print layers

    #googlenet best
    layers = [
            "inception_3b/5x5_reduce",
            "inception_3b/5x5",
            ]


    for octave_n in [4]:
        for octave_scale in [2.5]:
            for iterations in [50]:
                for layer in layers:
                    if layer != "data":
                        if guide != None:
                            print("================USING GUIDE=========================")
                            h, w = guide.shape[:2]
                            src, dst = net.blobs['data'], net.blobs[layer]
                            src.reshape(1,3,h,w)
                            src.data[0] = preprocess(net, guide)
                            net.forward(end=layer)
                            guide_features = dst.data[0].copy()
                            output = deepdream(filename, net, frame, iter_n=iterations, octave_n=octave_n, octave_scale=octave_scale, end=layer, model=model,objective=objective_guide,guide_features=guide_features,guide=guide, guide_file=guide_file)
                            print("####################DONE GUIDE#############################")
                            save_file(output, layer, iterations, octave_n, octave_scale, "_____final_____" + filename, model, guide_file)
                        else:
                            output = deepdream(filename, net, frame, iter_n=iterations, octave_n=octave_n, octave_scale=octave_scale, end=layer, model=model)
                            save_file(output, layer, iterations, octave_n, octave_scale, "_____final_____" + filename, model)

def save_file(output, layer, iterations, octave_n, octave_scale, filename, model,guide=""):
    name = guide + "____" + model + "_" + layer.replace("/", "") + "_itr_" + str(iterations) + "_octs_"
    name2 = name + str(octave_n) + "_scl_" + str(octave_scale) + "_jt_"
    name3 = name2 + "32__nonlin" + filename
    PIL.Image.fromarray(np.uint8(output)).save("outputs/" + name3, dpi=(600,600))


def start(filename, guide_file):
    print("Starting async file processing")
    with open("settings.json") as json_file:
        json_data = json.load(json_file)

    try:
        guide = np.float32(PIL.Image.open(os.getcwd() + "/guides/" + guide_file))
    except:
        guide = None

    img = PIL.Image.open(os.getcwd() + "/inputs/" + filename)
    if (img == None):
        quit()

    model_name = "googlenet"
    model_path = '../caffe/models/bvlc_googlenet/'
    param_fn = model_path + 'bvlc_googlenet.caffemodel'

    # model_name = "alexnet"
    # model_path = '../caffe/models/bvlc_alexnet/'
    # param_fn = model_path + 'bvlc_alexnet.caffemodel'

    # model_name = "rcnn"
    # model_path = '../caffe/models/bvlc_reference_rcnn_ilsvrc13/'
    # param_fn = model_path + 'bvlc_reference_rcnn_ilsvrc13.caffemodel'

    # model_name = "flickr"
    # model_path = '../caffe/models/finetune_flickr_style/'
    # param_fn = model_path + 'finetune_flickr_style.caffemodel'

    # model_name = "bvlc_reference"
    # model_path = '../caffe/models/bvlc_reference_caffenet/'
    # param_fn = model_path + 'bvlc_reference_caffenet.caffemodel'


    net_fn   = model_path + 'deploy.prototxt'

    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    maxwidth = json_data['maxwidth']

    width = img.size[0]

    if width > maxwidth:
        wpercent = (maxwidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((maxwidth,hsize), PIL.Image.ANTIALIAS)

    img = np.float32(img)

    frame = img
    print("++++++++++++++++++++++++++Processing File++++++++++++++++++++++++++")
    process2(net, frame, filename, model_name,guide, guide_file)
    shutil.move(os.getcwd() + "/inputs/" + guide_file, "guide_done/")


###############################################################################
count = 1
for filename in os.listdir(os.getcwd() + "/inputs/"):
    for guide in os.listdir(os.getcwd() + "/guides/"):
        if filename == ".DS_Store" or filename == ".gitkeep":
            continue
        if guide == ".DS_Store" or filename == ".gitkeep":
            continue
        print filename, guide
        p = Process(target=start, args=(filename,guide,))
        p.start()
        if count == 1:
            print("BLOCKING.......")
            p.join()
            count = 1
        else:
            count = count + 1
    if filename != ".DS_Store" and filename != ".gitkeep":
        shutil.move(os.getcwd() + "/inputs/" + filename, "done/")


