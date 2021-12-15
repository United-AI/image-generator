from keras.models import Model
import keras.layers as layers


def make_net(input_shape = [512,512,3], output_shape = 1,filter_sizes = [16,32,64,128,256,512,512]):
    #input_shape = [512,512,3]
    #output_shape = 1
    #filter_sizes = [16,32,64,128,256,512,512]
    layerdict = {}
    layerdict['block0_add'] = layers.Input(shape=(input_shape[0],input_shape[1],input_shape[2]))

    for num in range(1,len(filter_sizes)+1):
        layerdict['block'+str(num)+'_relu_1'] = layers.ReLU(name='block'+str(num)+'_relu_1')(layerdict['block'+str(num-1)+'_add'])

        layerdict['block'+str(num)+'_conv2d_1'] = layers.Conv2D(filters=filter_sizes[num-1],kernel_size=(3,3),strides=(2,2),padding='same', use_bias=False,
                                    name='block'+str(num)+'_conv2d_1',kernel_initializer='normal')(layerdict['block'+str(num)+'_relu_1'])

        layerdict['block'+str(num)+'_relu_2'] = layers.ReLU(name='block'+str(num)+'_relu_2')(layerdict['block'+str(num)+'_conv2d_1'])

        layerdict['block'+str(num)+'_out'] = layers.Conv2D(filters=filter_sizes[num-1],kernel_size=(3,3),strides=(1,1),padding='same', use_bias=False,
                                    name='block'+str(num)+'_out',kernel_initializer='normal')(layerdict['block'+str(num)+'_relu_2'])

        layerdict['block'+str(num)+'_add_fitter'] = layers.Conv2D(filters=filter_sizes[num-1],kernel_size=(3,3),strides=(2,2),padding='same', use_bias=False,
                                    name='block'+str(num)+'_add_fitter',kernel_initializer='normal')(layerdict['block'+str(num-1)+'_add'])
        
        layerdict['block'+str(num)+'_add'] = layers.add([layerdict['block'+str(num)+'_add_fitter'],layerdict['block'+str(num)+'_out']])

    num += 1

    layerdict['block'+str(num)+'_relu_1'] = layers.ReLU(name='block'+str(num)+'_relu_1')(layerdict['block'+str(num-1)+'_add'])

    layerdict['block'+str(num)+'_conv2d_1'] = layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same', use_bias=False,
                                    name='block'+str(num)+'_conv2d_1',kernel_initializer='normal')(layerdict['block'+str(num)+'_relu_1'])

    layerdict['block'+str(num)+'_relu_2'] = layers.ReLU(name='block'+str(num)+'_relu_2')(layerdict['block'+str(num)+'_conv2d_1'])

    layerdict['block'+str(num)+'_out'] = layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same', use_bias=False,
                                    name='block'+str(num)+'_out',kernel_initializer='normal')(layerdict['block'+str(num)+'_relu_2'])

    layerdict['block'+str(num)+'_add_fitter'] = layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same', use_bias=False,
                                    name='block'+str(num)+'_add_fitter',kernel_initializer='normal')(layerdict['block'+str(num-1)+'_out'])

    layerdict['block'+str(num)+'_add'] = layers.add([layerdict['block'+str(num)+'_add_fitter'],layerdict['block'+str(num)+'_out']])


    final_relu = layers.ReLU(name='final_relu')(layerdict['block'+str(num)+'_add'])


    final_pooling = layers.GlobalAveragePooling2D()(final_relu)

    outputlayer = layers.Dense(output_shape,name='model_output', activation='softmax',kernel_initializer='he_uniform')(final_pooling)

    model = Model(layerdict['block0_add'],outputlayer)
    return model

if __name__ == '__main__':
    model = make_net()
    model.summary()