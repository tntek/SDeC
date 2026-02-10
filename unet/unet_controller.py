import torch


class UNetController():
    # Static variables (Hyperparameters)

    def __init__(self):

        ## Variables (updated during inference) ##
        self.device = "cuda"
        self.current_unet_position = 'down'  # down, mid or up
        self.torch_dtype = torch.float16

        self.q_store = {}
        self.k_store = {}
        self.v_store = {}

        self.do_classifier_free_guidance = None

        ## Variables End ##
    

    def print_attributes(self):
        """
        Prints all attributes and their values of the object.
        """
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
