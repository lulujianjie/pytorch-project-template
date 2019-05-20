class Dataset():
    def __init__(self, data_dir = 'data_dir', verbose = True, using_val = True):
        super(Dataset, self).__init__()
        self.dataset_dir = data_dir
        if using_val:
            train, test = self.process_image_train_test(self.dataset_dir, relabel=True)
            if verbose:
                print("=> XXX loaded with train/val split")

            self.train = train
            self.test = test

        else:
            train = self.process_image_train(self.dataset_dir, relabel=True)

            if verbose:
                print("=> XXX loaded with train all")
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)


    def process_image_train_test(self, data_dir, relabel=True):
        train = [('img','label')]
        test = [('img','label')]
        return train, test
