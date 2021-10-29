import Setup
import tensorflow as tf
import matplotlib.pyplot as plt
setup = Setup.Setup()

LATENT_SPACE_DIM = 100
num_examples_to_generate = setup.num_examples_to_generate

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10, 10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        if predictions.shape[-1] == 3:
            plt.imshow(predictions[i] * 0.5 + .5)  # scale image to [0, 1] floats (or you could also scale to [0, 255] ints) 
        else: 
            plt.imshow(predictions[i, :, :, 0] * 0.5 + .5, cmap='gray')  # scale image to [0, 1] floats (or you could also scale to [0, 255] ints) 
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.savefig(setup.output_dir + '/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)
    print('Generated Epoch: ' + str(epoch))

loaded_generator = tf.keras.models.load_model(setup.checkpoint_dir+"generator")

generate_and_save_images(loaded_generator,-1,tf.random.normal([num_examples_to_generate, LATENT_SPACE_DIM]))