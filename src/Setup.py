class Setup:
    def __init__(self):
        self.jpg_path = "" #Datensatz path
        self.amount_files = 493
        self.BUFFER_SIZE = 6000
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 120 # reduce this to increase performance
        self.checkpoint_dir = "./checkpoints/" #output safepoint
        self.EPOCHS = 80000
        self.num_examples_to_generate = 16
        self.output_dir = "./output" #output folder
        self.load_model = False
