from keras.models import Model
import keras.layers as layers
from keras.activations import tanh

def make_net(input_shape = 256,filter_sizes = [512,512,256,128,64,32,16]):
    #input_shape = 256
    #filter_sizes = [512,512,256,128,64,32,16]
    layerdict = {}
    layerdict['input'] = layers.Input(shape=(input_shape,))

    layerdict['block0_dense'] = layers.Dense(4*4*512)(layerdict['input'])

    layerdict['block0_add'] = layers.Reshape([4,4,512])(layerdict['block0_dense'])
    for num in range(1,len(filter_sizes)+1):
        layerdict['block'+str(num)+'_BN_1'] = layers.BatchNormalization(epsilon=1e-3,momentum=0.999,
                                    name='block'+str(num)+'_BN_1')(layerdict['block'+str(num-1)+'_add'])

        layerdict['block'+str(num)+'_relu_1'] = layers.ReLU(name='block'+str(num)+'_relu_1')(layerdict['block'+str(num)+'_BN_1'])

        layerdict['block'+str(num)+'_conv2d_1'] = layers.Conv2DTranspose(filters=filter_sizes[num-1],kernel_size=(3,3),strides=(2,2),padding='same', use_bias=False,
                                    name='block'+str(num)+'_conv2d_1',kernel_initializer='normal')(layerdict['block'+str(num)+'_relu_1'])

        layerdict['block'+str(num)+'_BN_2'] = layers.BatchNormalization(epsilon=1e-3,momentum=0.999,
                                    name='block'+str(num)+'_BN_2')(layerdict['block'+str(num)+'_conv2d_1'])

        layerdict['block'+str(num)+'_relu_2'] = layers.ReLU(name='block'+str(num)+'_relu_2')(layerdict['block'+str(num)+'_conv2d_1'])

        layerdict['block'+str(num)+'_out'] = layers.Conv2DTranspose(filters=filter_sizes[num-1],kernel_size=(3,3),strides=(1,1),padding='same', use_bias=False,
                                    name='block'+str(num)+'_out',kernel_initializer='normal')(layerdict['block'+str(num)+'_relu_2'])
                                    
        layerdict['block'+str(num)+'_add_fitter'] = layers.Conv2DTranspose(filters=filter_sizes[num-1],kernel_size=(3,3),strides=(2,2),padding='same', use_bias=False,
                                    name='block'+str(num)+'_add_fitter',kernel_initializer='normal')(layerdict['block'+str(num-1)+'_add'])
        
        layerdict['block'+str(num)+'_add'] = layers.add([layerdict['block'+str(num)+'_add_fitter'],layerdict['block'+str(num)+'_out']])

    num += 1

    final_BN = layers.BatchNormalization(epsilon=1e-3,momentum=0.999,
                                    name='block'+str(num)+'_BN_1')(layerdict['block'+str(num-1)+'_add'])

    final_relu = layers.ReLU(name='final_relu')(final_BN)


    final_conv = layers.Conv2D(filters=3,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=False,
                                name='final_conv',kernel_initializer='normal')(final_relu)

    final_layer = tanh(final_conv)
    model = Model(layerdict['input'],final_layer)
    return model

if __name__ == '__main__':
    model = make_net()
    model.summary()