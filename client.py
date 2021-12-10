import random
import warnings
from data.PruneH import prune,convert,disp,call_kmeans
import numpy as np

class Client:

    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data

    def train(self, num_epochs=1, batch_size=10 , minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        before=np.array(self.model.get_params())
        if minibatch is None:
            data = self.train_data
            comp, update = self.model.train(data, num_epochs, batch_size)
            # print(type(before),type(update))

        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}

            # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
            num_epochs = 1
            comp, update = self.model.train(data, num_epochs, num_data)
        np.save("keep",update)
        t1=np.array(update)
        t=prune(before, t1, 0.1)
        """clustered_delta """
        #c, clustered_updates, centeroids = call_kmeans( before , t)
        c, centroids, clustered_updates = call_kmeans(before , t)
        for i in range(8):
            print(len(clustered_updates[i]),centroids[i])
        '''np.save("data/first.npy",t1)
        np.save("data/before.npy",t)
        np.save("data/after.npy",clustered_updates)'''
        #disp(t)
        #disp(clustered_updates)
        # print(t==t1)
        """f1 = clustered_updates - before
        for j,k in zip(f1,clustered_delta):
            print(np.unique(j.flatten()))
            print(np.unique(k.flatten()))"""
        num_train_samples = len(data['y'])
        return comp, num_train_samples, clustered_updates  , centroids  #"""t => clustered_updates"""

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.

        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
