class Setup:
    def __init__(self):
        self.jpg_path = "" #Datensatz path
        self.amount_files = 10000
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 128
        self.IMAGE_SIZE = 120 # reduce this to increase performance
        self.checkpoint_dir = "" #output safepoint
        self.EPOCHS = 10000
        self.num_examples_to_generate = 16
        self.output_dir = "" #output folder
