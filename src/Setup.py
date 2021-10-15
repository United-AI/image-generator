class Setup:
    def __init__(self):
        self.jpg_path = "" #Datensatz path
        self.jpg_file_start = 0
        self.jpg_file_end = 50000
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 256
        self.checkpoint_dir = "" #output safepoint
        self.EPOCHS = 300
        self.num_examples_to_generate = 16
        self.output_dir = "" #output folder
        self.gifmaker_input = "" #input gifmaker